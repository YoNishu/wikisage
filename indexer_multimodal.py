"""
Multimodal RAG ingestion — text + images + mermaid diagrams.
Populates the 'ado_wiki_images' index with GPT-4.1 vision descriptions.

Usage: python indexer_multimodal.py

This is the multimodal variant of indexer.py. It:
  - Downloads wiki attachment images via ADO Git Items API
  - Describes each image using GPT-4.1 vision (with surrounding page context)
  - Describes mermaid diagrams using GPT-4.1 text
  - Stores image_url, image_type, mermaid_code in the index
  - Uses index name 'ado_wiki_images' (separate from text-only 'ado_wiki')

To use this index at query time, set INDEX_MODE=multimodal in .env
"""

import os
import sys
from pathlib import Path
from common import ConfigManager, logger
from indexer import RAGIngestor


class MultimodalRAGIngestor(RAGIngestor):
    """RAG ingestor with image and mermaid diagram processing enabled."""
    
    def __init__(self, config):
        # Override index name to use the images index
        config["INDEX_NAME"] = "ado_wiki_images"
        super().__init__(config)
        
        # Enable image/mermaid processing
        self.process_images = True
        
        # Use separate state file so text and multimodal don't interfere
        self.state_file = Path("ingestion_state_images.json")
        
        logger.info("Multimodal ingestion enabled — images and mermaid diagrams will be described by GPT-4.1")


def main():
    """Main entry point for multimodal ingestion."""
    try:
        os.makedirs("logs", exist_ok=True)
        
        config = ConfigManager.load_config()
        
        logger.info("="*60)
        logger.info("MULTIMODAL INGESTION (text + images + mermaid)")
        logger.info("Index: ado_wiki_images")
        logger.info("="*60)
        
        ingestor = MultimodalRAGIngestor(config)
        ingestor.run()
        
    except KeyboardInterrupt:
        logger.info("\nIngestion interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
