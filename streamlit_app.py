"""
Beautiful Streamlit interface for the ADO Wiki RAG system.
A modern, user-friendly web application for querying your Azure DevOps wiki content.
"""

import streamlit as st
import time
from typing import List, Dict, Any
from datetime import datetime

# Import your existing query components
from query import EnhancedQueryProcessor, QueryResult, QueryResponse
from utils import ConfigManager, logger

# Configure Streamlit page
st.set_page_config(
    page_title="ADO Wiki Agent",
    page_icon="�️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    
    .stTextInput > div > div > input {
        background-color: #f8f9fa;
        border: 2px solid #e9ecef;
        border-radius: 10px;
        padding: 12px;
        font-size: 16px;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #0078d4;
        box-shadow: 0 0 0 3px rgba(0, 120, 212, 0.1);
    }
    
    .query-container {
        background: linear-gradient(135deg, #0078d4 0%, #106ebe 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
    }
    
    .answer-container {
        background-color: #ffffff;
        border: 1px solid #e9ecef;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .context-card {
        background-color: #f8f9fa;
        border-left: 4px solid #0078d4;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .sidebar-content {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    .example-query {
        background-color: #e3f2fd;
        border: 1px solid #bbdefb;
        border-radius: 8px;
        padding: 0.8rem;
        margin: 0.3rem 0;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .example-query:hover {
        background-color: #bbdefb;
        transform: translateY(-2px);
    }
    
    .stats-container {
        display: flex;
        justify-content: space-around;
        margin: 1rem 0;
    }
    
    .highlight-text {
        background: linear-gradient(120deg, #a8edea 0%, #fed6e3 100%);
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-weight: bold;
    }
    
    /* Green color for search button */
    .stButton > button[kind="primary"] {
        background-color: #28a745 !important;
        border-color: #28a745 !important;
    }
    
    .stButton > button[kind="primary"]:hover {
        background-color: #218838 !important;
        border-color: #1e7e34 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []  # Chat messages: [{"role": "user"/"assistant", "content": "..."}]
if 'processor' not in st.session_state:
    st.session_state.processor = None
if 'total_queries' not in st.session_state:
    st.session_state.total_queries = 0
if 'last_contexts' not in st.session_state:
    st.session_state.last_contexts = []  # Store contexts from last query for expandable view
if 'query_metrics' not in st.session_state:
    st.session_state.query_metrics = []  # [{tokens, sources, avg_relevance}, ...]
if 'wiki_base_url' not in st.session_state:
    st.session_state.wiki_base_url = None  # Set after config loads

def build_wiki_url(page_path: str) -> str:
    """Build a clickable ADO wiki URL from a page path."""
    base = st.session_state.wiki_base_url
    if not base:
        return ""
    # ADO wiki URLs use the page path with / replaced by %2F
    import urllib.parse
    encoded_path = urllib.parse.quote(page_path, safe='')
    return f"{base}?pagePath={encoded_path}"

@st.cache_resource
def get_query_processor():
    """Initialize and cache the query processor."""
    try:
        config = ConfigManager.load_config()
        processor = EnhancedQueryProcessor(config)
        # Build wiki base URL for linking to source pages
        org = config.get('ADO_ORG', '')
        project = config.get('ADO_PROJECT', '')
        wiki = config.get('ADO_WIKI', '')
        if org and project and wiki:
            st.session_state.wiki_base_url = f"https://dev.azure.com/{org}/{project}/_wiki/wikis/{wiki}"
        return processor
    except Exception as e:
        st.error(f"Failed to initialize query processor: {str(e)}")
        return None

def main():
    """Main Streamlit chat application."""
    
    # Header
    st.markdown("""
    <div class="query-container" style="display: flex; align-items: center; gap: 1rem;">
        <img src="https://raw.githubusercontent.com/David-Summers/Azure-Design/master/SVG_Azure_All/Azure%20Front%20Door.svg" style="height: 50px;" />
        <div>
            <h1 style="margin: 0; font-size: 2.2rem;">ADO Wiki Agent</h1>
            <p style="margin: 0.5rem 0 0 0; font-size: 1rem; opacity: 0.9;">
                Chat with your Azure DevOps wiki &mdash; multi-turn conversations with RAG
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize processor
    if st.session_state.processor is None:
        with st.spinner("🚀 Initializing ADO Wiki Agent..."):
            st.session_state.processor = get_query_processor()
    
    if st.session_state.processor is None:
        st.error("❌ Failed to initialize the system. Please check your configuration.")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 1rem;">
            <img src="https://raw.githubusercontent.com/David-Summers/Azure-Design/master/SVG_Azure_All/Azure%20Front%20Door.svg" style="height: 40px;" />
            <div style="font-weight: bold; font-size: 1.1rem;">ADO Wiki Agent</div>
            <div style="font-size: 0.8rem; color: #888;">GPT-4.1 + Azure AI Search</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"**Session:** {st.session_state.total_queries} queries | {len(st.session_state.messages)//2} turns")
        
        if st.button("🗑️ New Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_contexts = []
            st.session_state.total_queries = 0
            st.session_state.query_metrics = []
            st.rerun()
        
        # Session metrics
        if st.session_state.query_metrics:
            st.markdown("---")
            st.markdown("**📊 Session Metrics**")
            total_tokens = sum(m["tokens"] for m in st.session_state.query_metrics)
            total_sources = sum(m["sources"] for m in st.session_state.query_metrics)
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Total Tokens", f"{total_tokens:,}")
            with col_b:
                st.metric("Sources Found", total_sources)
            
            # Per-query breakdown
            st.markdown("**Per-query:**")
            for i, m in enumerate(st.session_state.query_metrics, 1):
                st.caption(f"Q{i}: {m['tokens']:,} tok | {m['sources']} sources")
        
        # Show sources from last answer
        if st.session_state.last_contexts:
            st.markdown("---")
            st.markdown(f"**📚 Sources ({len(st.session_state.last_contexts)})**")
            for i, ctx in enumerate(st.session_state.last_contexts, 1):
                icon = "📐" if ctx.image_type == "mermaid" else "🖼️" if ctx.image_type in ("png","jpg","gif","svg") else "📄"
                page_name = ctx.page_path.split('/')[-1][:30] if ctx.page_path else "?"
                wiki_url = build_wiki_url(ctx.page_path)
                with st.expander(f"{icon} Source {i}: {page_name}"):
                    if wiki_url:
                        st.markdown(f"🔗 **[Open in ADO Wiki]({wiki_url})**")
                    st.caption(f"Page: {ctx.page_path}")
                    st.caption(f"Type: {ctx.chunk_type} | Tokens: {ctx.token_count}")
                    # Render mermaid diagrams
                    if ctx.image_type == "mermaid" and ctx.mermaid_code:
                        st.code(ctx.mermaid_code, language="mermaid")
                    # Render images
                    elif ctx.image_type in ("png", "jpg", "gif", "svg") and ctx.image_url:
                        try:
                            st.image(ctx.image_url, caption=ctx.page_path.split('/')[-1])
                        except Exception:
                            st.caption(f"📷 Image: {ctx.image_url}")
                    st.text(ctx.content[:500] + ("..." if len(ctx.content) > 500 else ""))
    
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Welcome message if empty
    if not st.session_state.messages:
        with st.chat_message("assistant"):
            st.markdown("👋 Hi! I'm your ADO Wiki Agent. Ask me anything covered in your team's wiki. I'll search, retrieve, and synthesize answers from your documentation. You can ask follow-up questions and I'll keep context across our conversation.")
    
    # Chat input
    user_input = st.chat_input("Ask about architecture, runbooks, deployments...")
    query = user_input
    
    if query:
        st.session_state.total_queries += 1
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.status("🔍 Searching wiki...", expanded=True) as status:
                st.write("Contextualizing query from chat history...")
                st.write("Extracting search keywords...")
                st.write("**Pass 1:** Multi-query + HyDE + hybrid semantic search...")
                st.write("**Pass 1:** LLM reranking first-pass results...")
                st.write("**Pass 2:** Generating refined query from first-pass chunks...")
                st.write("**Pass 2:** Retrieving additional chunks...")
                st.write("Merging, deduplicating & final reranking...")
                st.write("Stitching neighbor chunks...")
                st.write("Generating answer with ≥10 source chunks...")
                
                # Build chat history for the processor (exclude current message)
                chat_history = st.session_state.messages[:-1]  # all messages except the one just added
                
                # Process query with full chat history
                response = st.session_state.processor.process_chat_query(query, chat_history=chat_history)
                
                status.update(label=f"✅ Found {len(response.contexts_used)} sources, used {response.total_tokens_used:,} tokens", state="complete", expanded=False)
            
            # Display answer
            st.markdown(response.answer)
            
            # Display source links after the answer — every source with wiki link
            if response.contexts_used:
                st.markdown("---")
                st.markdown(f"**📚 Sources ({len(response.contexts_used)} chunks from wiki):**")
                for i, ctx in enumerate(response.contexts_used, 1):
                    wiki_url = build_wiki_url(ctx.page_path)
                    page_name = ctx.page_path.split('/')[-1] if ctx.page_path else "Unknown"
                    if wiki_url:
                        st.markdown(f"- **[Source {i}]** [{page_name}]({wiki_url})")
                    else:
                        st.markdown(f"- **[Source {i}]** {ctx.page_path}")
            
            # Display related diagrams and images inline
            diagrams = [c for c in response.contexts_used if c.image_type != "none"]
            if diagrams:
                st.markdown("---")
                st.markdown("**📊 Related Diagrams & Images**")
                for ctx in diagrams:
                    if ctx.image_type == "mermaid" and ctx.mermaid_code:
                        with st.expander(f"📐 Mermaid: {ctx.page_path.split('/')[-1][:40]}", expanded=True):
                            st.code(ctx.mermaid_code, language="mermaid")
                    elif ctx.image_url and ctx.image_type in ("png", "jpg", "gif", "svg"):
                        with st.expander(f"🖼️ Image: {ctx.page_path.split('/')[-1][:40]}", expanded=True):
                            try:
                                st.image(ctx.image_url)
                            except Exception:
                                st.caption(f"Image: {ctx.image_url}")
            
            # Store contexts and metrics for sidebar
            st.session_state.last_contexts = response.contexts_used
            avg_rel = sum(1 - c.distance for c in response.contexts_used) / len(response.contexts_used) if response.contexts_used else 0
            st.session_state.query_metrics.append({
                "tokens": response.total_tokens_used,
                "sources": len(response.contexts_used),
                "avg_relevance": avg_rel
            })
        
        # Add assistant message to history
        st.session_state.messages.append({"role": "assistant", "content": response.answer})
        
        st.rerun()  # Rerun to update sidebar with new sources
    
    # Footer
    st.markdown("""
    <div style="text-align: center; color: #888; padding: 0.5rem; font-size: 0.8rem;">
        GPT-4.1 + text-embedding-3-large + Azure AI Search | ADO Wiki Agent with HyDE + RRF + LLM Reranking
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
