"""
Targeted ingestion script — ingest a specific wiki page and all its subpages.
Updates the existing ingestion_state.json so delta detection works correctly.

Usage:
    python ingest_page.py "/DRI"                          # Ingest /DRI and all subpages
    python ingest_page.py "/Engineering/Feature Flags"   # Ingest specific section
    python ingest_page.py "/DRI/TSG" --index-mode text     # Force text-only mode
    python ingest_page.py "/DRI" --force                   # Force re-ingest even if unchanged

Examples:
    python ingest_page.py "/"                              # Re-ingest everything (same as ingest.py)
    python ingest_page.py "/DRI/TSG/TSG Edge-HighLatency"  # Single page only (no subpages to recurse)
"""

import os
import sys
import argparse
import signal
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from utils import ConfigManager, WikiContentFetcher, SmartContentChunker, EmbeddingGenerator, ProcessingStats, ContentChunk, logger
from ingest import RAGIngestor, IngestionState
from azure_search_client import AzureSearchVectorStore


class TargetedRAGIngestor(RAGIngestor):
    """Ingest a specific wiki page path and all its subpages."""

    def __init__(self, config: Dict[str, str], target_path: str, force: bool = False):
        super().__init__(config)
        self.target_path = target_path.rstrip("/")
        self.force_reingest = force
        # Use separate state file so manual ingestion doesn't interfere with full ingestion
        self.state_file = Path("ingestion_state_manual.json")

    def should_process_page(self, page: Dict, state: IngestionState) -> bool:
        """Override: if --force, always process; otherwise use delta detection."""
        if self.force_reingest:
            return True
        return super().should_process_page(page, state)

    def run_targeted(self):
        """Fetch and ingest only the target page and its subpages, in batches."""
        logger.info(f"{'='*60}")
        logger.info(f"TARGETED INGESTION: {self.target_path}")
        logger.info(f"Force re-ingest: {self.force_reingest}")
        logger.info(f"Index: {self.config.get('INDEX_NAME', 'ado_wiki')}")
        logger.info(f"{'='*60}")

        try:
            # Load existing state
            state = self.load_state()
            self.state = state

            # Get wiki ID
            wiki_url = f"https://dev.azure.com/{self.config['ADO_ORG']}/{self.config['ADO_PROJECT']}/_apis/wiki/wikis/{self.config['ADO_WIKI']}?api-version=7.1-preview.1"
            logger.info(f"Fetching wiki details...")
            wiki_res = self.wiki_fetcher.session.get(wiki_url, headers=self.wiki_fetcher.headers)
            wiki_res.raise_for_status()
            wiki_data = wiki_res.json()

            if 'value' in wiki_data and len(wiki_data['value']) > 0:
                wiki_details = wiki_data['value'][0]
            else:
                wiki_details = wiki_data
            wiki_id = wiki_details.get('id') or wiki_details.get('wikiIdentifier') or self.config['ADO_WIKI']

            # Fetch page structure for the target path (lightweight — no content)
            base_url = f"https://dev.azure.com/{self.config['ADO_ORG']}/{self.config['ADO_PROJECT']}/_apis/wiki/wikis/{wiki_id}/pages"
            params = {
                "path": self.target_path,
                "recursionLevel": "Full",
                "includeContent": "False",
                "api-version": "7.1-preview.1"
            }

            logger.info(f"Fetching page structure for: {self.target_path}")
            structure_res = self.wiki_fetcher.session.get(base_url, headers=self.wiki_fetcher.headers, params=params)

            if structure_res.status_code == 404:
                logger.error(f"Page not found: {self.target_path}")
                logger.error("Check the path — it should match the wiki page hierarchy (e.g., '/DRI/TSG')")
                sys.exit(1)

            structure_res.raise_for_status()
            root_data = structure_res.json()

            # Pass 1: Collect just page paths (no content fetch — fast)
            page_paths = []
            self._collect_page_paths(root_data, page_paths)
            logger.info(f"Found {len(page_paths)} pages under {self.target_path}")

            if not page_paths:
                logger.warning("No pages to process")
                return

            # Pass 2: Process in batches of 25
            batch_size = 25
            processed = 0
            chunks_total = 0

            for batch_start in range(0, len(page_paths), batch_size):
                if self.interrupted:
                    logger.info("Interrupted")
                    break

                batch_paths = page_paths[batch_start:batch_start + batch_size]
                batch_num = (batch_start // batch_size) + 1
                total_batches = (len(page_paths) + batch_size - 1) // batch_size
                logger.info(f"\n=== Batch {batch_num}/{total_batches} ({len(batch_paths)} pages) ===")

                # Fetch content for this batch
                batch_pages = []
                for page_info in batch_paths:
                    if self.interrupted:
                        break
                    path = page_info["path"]
                    logger.info(f"Fetching content for: {path}")
                    try:
                        content_res = self.wiki_fetcher.session.get(
                            base_url,
                            headers=self.wiki_fetcher.headers,
                            params={"path": path, "includeContent": "true", "api-version": "7.1-preview.1"}
                        )
                        content_res.raise_for_status()
                        page_with_content = content_res.json()
                        page_info["content"] = page_with_content.get("content", "")
                        time.sleep(0.1)
                    except Exception as e:
                        logger.error(f"Error fetching content for {path}: {str(e)}")
                        page_info["content"] = ""
                    batch_pages.append(page_info)

                # Process each page in the batch (chunk → embed → upsert)
                batch_chunks = 0
                for page in batch_pages:
                    if self.interrupted:
                        break

                    if not self.should_process_page(page, state):
                        logger.info(f"Skipping unchanged: {page.get('path')}")
                        continue

                    chunks_created, success = self.process_page(page, state)
                    if success:
                        processed += 1
                        batch_chunks += chunks_created
                        chunks_total += chunks_created
                        self.stats.pages_processed += 1
                        self.stats.total_content_size += len(page.get("content", ""))

                # Save state after each batch
                self.save_state(state)
                logger.info(f"Batch {batch_num} complete: {batch_chunks} chunks upserted, state saved")

            # Final state update
            state.pages_processed = len(state.pages)
            state.total_chunks = sum(p.get("chunk_count", 0) for p in state.pages.values())
            self.save_state(state)

            # Summary
            logger.info(f"\n{'='*60}")
            logger.info(f"TARGETED INGESTION COMPLETE")
            logger.info(f"{'='*60}")
            logger.info(f"Target path: {self.target_path}")
            logger.info(f"Pages found: {len(page_paths)}")
            logger.info(f"Pages processed: {processed}")
            logger.info(f"Chunks upserted: {chunks_total}")
            logger.info(f"Errors: {self.stats.errors_encountered}")
            logger.info(f"Total pages in state: {len(state.pages)}")
            logger.info(f"{'='*60}")

        except Exception as e:
            logger.error(f"Fatal error: {str(e)}")
            raise
        finally:
            if hasattr(self, 'state'):
                self.save_state(self.state)

    def _collect_page_paths(self, page_data: Dict, collected: List[Dict]):
        """Recursively collect page paths and metadata WITHOUT fetching content."""
        path = page_data.get("path", "")
        page_id = page_data.get("id", path)

        # Compute parent path
        parts = path.strip('/').split('/')
        parent_path = '/' + '/'.join(parts[:-1]) if len(parts) > 1 else ""

        collected.append({
            "id": page_id,
            "path": path,
            "parent_path": parent_path,
            "lastModified": page_data.get("lastModified", ""),
        })

        for sub_page in page_data.get("subPages", []):
            self._collect_page_paths(sub_page, collected)


def main():
    parser = argparse.ArgumentParser(
        description="Ingest a specific wiki page and all its subpages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ingest_page.py "/DRI"                    # Ingest /DRI and all subpages
  python ingest_page.py "/DRI/TSG" --force        # Force re-ingest even if unchanged
  python ingest_page.py "/Engineering" --force   # Re-ingest the Engineering section
        """
    )
    parser.add_argument("path", help="Wiki page path to ingest (e.g., '/Onboarding', '/Engineering')")
    parser.add_argument("--force", action="store_true", help="Force re-ingest even if pages haven't changed")
    parser.add_argument("--index-mode", choices=["text", "multimodal"], default=None, help="Override INDEX_MODE from .env")
    args = parser.parse_args()

    try:
        os.makedirs("logs", exist_ok=True)
        config = ConfigManager.load_config()

        # Override index mode if specified
        if args.index_mode:
            if args.index_mode == "multimodal":
                config["INDEX_NAME"] = "ado_wiki_images"
            else:
                config["INDEX_NAME"] = config.get("INDEX_NAME", "ado_wiki")

        logger.info(f"Targeted ingestion: {args.path}")
        ingestor = TargetedRAGIngestor(config, target_path=args.path, force=args.force)
        ingestor.run_targeted()

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
