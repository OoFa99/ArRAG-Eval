"""
Qdrant Vector Store Implementation.

A production-grade vector database that runs locally (Docker) or in the cloud.
The sweet spot between ChromaDB (simple) and Pinecone (expensive).

Why Qdrant:
  ✓ Production-ready (built for scale)
  ✓ Local Docker support (no API keys)
  ✓ Fast & efficient (better than ChromaDB on large indexes)
  ✓ Cloud option when you're ready (no rewrite needed)
  ✓ Built-in filtering & metadata
  ✓ Same interface as Pinecone/ChromaDB (drop-in replacement)

Use cases:
  - Week 1-2: Run locally via Docker (free)
  - Week 5-8: Switch to cloud if needed (still drop-in replacement)
  - Production: Use cloud Qdrant with uptime SLA

Setup (Docker):
  docker run -d -p 6333:6333 qdrant/qdrant:latest
  Then initialize: store = VectorStoreQdrant()
"""

import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import uuid

logger = logging.getLogger(__name__)


class VectorStoreQdrant:
    """
    Qdrant-based vector store with BM25 hybrid search.
    
    Same interface as PineconeStore and VectorStoreChroma, but uses Qdrant
    which can run locally (Docker) or in the cloud.
    
    Architecture:
      Dense Index:  Qdrant (vector embeddings with filtering)
      Sparse Index: BM25 (keyword matching)
      Fusion:       RRF (Reciprocal Rank Fusion)
    
    Example (Local Docker):
        >>> # First, start Qdrant: docker run -p 6333:6333 qdrant/qdrant
        >>> store = VectorStoreQdrant(
        ...     collection_name="arrag-eval",
        ...     url="http://localhost:6333"
        ... )
        >>> store.add_documents(chunks, document_id="wiki_001")
        >>> results = store.hybrid_query("What is ML?", top_k=5)
    
    Example (Cloud Qdrant):
        >>> store = VectorStoreQdrant(
        ...     collection_name="arrag-eval",
        ...     url="https://your-qdrant-instance.qdrant.io",
        ...     api_key="your-qdrant-api-key"
        ... )
    """
    
    def __init__(self,
                 collection_name: str = "arrag-eval",
                 model_name: str = "intfloat/multilingual-e5-small",
                 url: str = "http://localhost:6333",
                 api_key: Optional[str] = None,
                 prefer_grpc: bool = False):
        """
        Initialize Qdrant vector store.
        
        Args:
            collection_name (str): Name of collection. Default: "arrag-eval"
            model_name (str): HuggingFace embedding model.
                            Default: "intfloat/multilingual-e5-small"
            url (str): Qdrant server URL.
                      - Local: "http://localhost:6333"
                      - Cloud: "https://your-instance.qdrant.io"
                      Default: "http://localhost:6333"
            api_key (str): API key for cloud Qdrant (optional for local)
            prefer_grpc (bool): Use gRPC instead of REST (faster but needs Docker config)
        
        Raises:
            ConnectionError: If cannot connect to Qdrant server
        
        Setup Instructions:
            
            LOCAL (Docker):
              1. Install Docker: https://docs.docker.com/get-docker/
              2. Start Qdrant:
                 docker run -d -p 6333:6333 qdrant/qdrant:latest
              3. Initialize store:
                 store = VectorStoreQdrant()
              4. Done! (data persists in Docker volume)
            
            CLOUD:
              1. Sign up: https://cloud.qdrant.io
              2. Create cluster (get URL + API key)
              3. Initialize store:
                 store = VectorStoreQdrant(
                     url="https://your-instance.qdrant.io",
                     api_key="your-api-key"
                 )
        
        Example:
            >>> # Local development
            >>> store = VectorStoreQdrant(
            ...     collection_name="arrag-eval-dev",
            ...     url="http://localhost:6333"
            ... )
            >>>
            >>> # Production cloud
            >>> store = VectorStoreQdrant(
            ...     collection_name="arrag-eval-prod",
            ...     url="https://my-qdrant-cluster.qdrant.io",
            ...     api_key=os.getenv("QDRANT_API_KEY")
            ... )
        """
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_embedding_dimension()
        self.model_name = model_name
        
        # Initialize Qdrant client
        print(f"Connecting to Qdrant at {url}...")
        try:
            self.client = QdrantClient(
                url=url,
                api_key=api_key,
                prefer_grpc=prefer_grpc
            )
            # Test connection
            self.client.get_collections()
            logger.info("✓ Successfully connected to Qdrant")
        except Exception as e:
            raise ConnectionError(
                f"Could not connect to Qdrant at {url}. "
                f"Make sure Qdrant is running. "
                f"Start with: docker run -p 6333:6333 qdrant/qdrant:latest\n"
                f"Error: {e}"
            )
        
        self.collection_name = collection_name
        self.url = url
        
        # Create or get collection
        self._init_collection()
        
        # BM25 components (in-memory)
        self.bm25 = None
        self.corpus = []
        self.corpus_ids = []
        self.added_documents = set()
        
        # Load existing documents from Qdrant
        self._load_existing_documents()
    
    def _init_collection(self):
        """Create collection if it doesn't exist."""
        try:
            self.client.get_collection(self.collection_name)
            logger.info(f"✓ Loaded existing collection: {self.collection_name}")
        except Exception:
            # Collection doesn't exist, create it
            print(f"Creating new collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"✓ Created collection: {self.collection_name}")
    
    def _load_existing_documents(self):
        """Load existing documents from Qdrant into BM25."""
        try:
            # Get all points in collection
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000  # Adjust if you have more than 10K chunks
            )
            
            if points:
                print(f"Loading {len(points)} existing points into BM25...")
                
                for point in points:
                    # Reconstruct document from payload
                    chunk_id = point.id
                    chunk_text = point.payload.get('text', '')
                    
                    self.corpus.append(chunk_text)
                    self.corpus_ids.append(chunk_id)
                    
                    # Track document source
                    source = point.payload.get('source_document', '')
                    if source:
                        self.added_documents.add(source)
                
                # Build BM25
                if self.corpus:
                    tokenized = [self._tokenize_arabic(c) for c in self.corpus]
                    self.bm25 = BM25Okapi(tokenized)
                    logger.info(f"✓ BM25 built from {len(self.corpus)} chunks")
        except Exception as e:
            logger.debug(f"No existing documents: {e}")
    
    def add_documents(self, chunks: List[str], document_id: str):
        """
        Add document chunks to both Qdrant (dense) and BM25 (sparse) indexes.
        
        Dual indexing:
          1. Qdrant: Vector embeddings with metadata filtering
          2. BM25: Tokenized text for keyword search
          3. RRF: Combine at query time
        
        Idempotency:
          - Checks if document already exists
          - Skips if present (safe for re-runs)
        
        Args:
            chunks (List[str]): Text chunks
            document_id (str): Source document identifier
        
        Returns:
            None
        
        Example:
            >>> chunks = ["Chapter 1...", "Chapter 2...", "Chapter 3..."]
            >>> store.add_documents(chunks, document_id="book_001")
        """
        if not chunks:
            logger.warning(f"Empty chunk list for: {document_id}")
            return
        
        # Check idempotency
        if document_id in self.added_documents:
            logger.info(f"✓ Document '{document_id}' already indexed. Skipping.")
            return
        
        print(f"Adding {len(chunks)} chunks for document: {document_id}")
        
        # ============ Step 1: Embed with SentenceTransformer ============
        embeddings = self.model.encode(chunks, normalize_embeddings=True)
        
        # ============ Step 2: Prepare points for Qdrant ============
        points = []
        chunk_ids = []
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Generate unique ID (can also use f"{document_id}_chunk_{i}")
            chunk_id = f"{document_id}_chunk_{i}"
            chunk_ids.append(chunk_id)
            
            point = PointStruct(
                id=hash(chunk_id) % (2**63),  # Convert string to int64
                vector=embedding.tolist(),
                payload={
                    "text": chunk,
                    "source_document": document_id,
                    "chunk_index": i,
                    "chunk_id": chunk_id  # Store original ID in payload
                }
            )
            points.append(point)
        
        # ============ Step 3: Upsert to Qdrant (batched) ============
        batch_size = 100
        for batch_idx in range(0, len(points), batch_size):
            batch = points[batch_idx:batch_idx + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch
            )
            logger.debug(f"Upserted batch {batch_idx//batch_size + 1}")
        
        # ============ Step 4: Update BM25 index ============
        self.corpus.extend(chunks)
        self.corpus_ids.extend(chunk_ids)
        
        tokenized = [self._tokenize_arabic(c) for c in self.corpus]
        self.bm25 = BM25Okapi(tokenized)
        
        # Track document
        self.added_documents.add(document_id)
        
        print(f"✅ Added {len(chunks)} chunks")
    
    def _tokenize_arabic(self, text: str) -> List[str]:
        """
        Tokenize text (Arabic and English).
        
        Simple but effective:
          - Lowercase
          - Remove newlines/tabs
          - Split on whitespace
        
        Args:
            text (str): Input text
        
        Returns:
            List[str]: Tokens
        
        Example:
            >>> tokens = store._tokenize_arabic("مرحبا بك")
            ['مرحبا', 'بك']
        """
        tokens = text.lower().replace('\n', ' ').replace('\t', ' ').split()
        return tokens
    
    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Semantic search using Qdrant (dense embeddings).
        
        Strengths:
          ✓ Captures semantic meaning
          ✓ Good for paraphrases
          ✓ Works across languages
        
        Args:
            query_text (str): Natural language query
            top_k (int): Results to return. Default: 5
        
        Returns:
            List[Dict] with:
              - id: Chunk identifier
              - score: Similarity (0-1, higher better)
              - text: Original text
              - source: Source document
        
        Example:
            >>> results = store.query("What is ML?", top_k=3)
            >>> for r in results:
            ...     print(f"{r['score']:.3f}: {r['text'][:50]}...")
        """
        # Embed query
        query_vector = self.model.encode([query_text], normalize_embeddings=True)[0].tolist()
        
        # Search in Qdrant
        search_results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
            with_vectors=False
        )
        
        formatted = []
        for hit in search_results.points:
            # ScoredPoint object
            point_id = hit.id
            score = hit.score
            payload = hit.payload or {}
            
            formatted.append({
                "id": payload.get('chunk_id', str(point_id)),
                "score": float(score),
                "text": payload.get('text', ''),
                "source": payload.get('source_document', '')
            })
            
        # # Format results
        # formatted = []
        # for hit in search_results:
        #     formatted.append({
        #         "id": hit.payload.get('chunk_id', str(hit.id)),
        #         "score": hit.score,
        #         "text": hit.payload.get('text', ''),
        #         "source": hit.payload.get('source_document', '')
        #     })
        
        return formatted
    
    def bm25_query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Keyword search using BM25 (sparse retrieval).
        
        Strengths:
          ✓ Exact keyword matching
          ✓ Good for technical terms
          ✓ Interpretable
        
        Args:
            query_text (str): Keywords
            top_k (int): Results. Default: 5
        
        Returns:
            List[Dict] with:
              - id: Chunk ID
              - score: BM25 score
              - text: Text
              - source: "bm25"
        
        Example:
            >>> results = store.bm25_query("neural networks", top_k=5)
        """
        if not self.bm25:
            logger.warning("BM25 not initialized")
            return []
        
        # Tokenize and score
        tokenized_query = self._tokenize_arabic(query_text)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        return [
            {
                "id": self.corpus_ids[i],
                "score": float(scores[i]),
                "text": self.corpus[i],
                "source": "bm25"
            }
            for i in top_indices
        ]
    
    def hybrid_query(self, query_text: str, top_k: int = 10, k: int = 60) -> List[Dict[str, Any]]:
        """
        Hybrid search (semantic + keyword) with RRF fusion.
        
        Combines Qdrant (dense) + BM25 (sparse) using Reciprocal Rank Fusion.
        
        RRF Score: 1 / (rank + k)
          - Documents in both rankings get boosted
          - Parameter k balances importance of ranking position
        
        Args:
            query_text (str): Query (natural language or keywords)
            top_k (int): Final results. Default: 10
            k (int): RRF parameter. Default: 60
        
        Returns:
            List[Dict] with:
              - id: Chunk ID
              - score: RRF combined score
              - text: Text
        
        Example:
            >>> results = store.hybrid_query("ML algorithms", top_k=5)
        """
        # Get results from both methods
        semantic = self.query(query_text, top_k=top_k * 2)
        bm25 = self.bm25_query(query_text, top_k=top_k * 2)
        
        # RRF fusion
        rrf_scores = {}
        
        # Add semantic results
        for rank, result in enumerate(semantic):
            doc_id = result['id']
            score = 1 / (rank + k + 1)
            rrf_scores[doc_id] = {"text": result['text'], "score": score}
        
        # Add/combine BM25
        for rank, result in enumerate(bm25):
            doc_id = result['id']
            score = 1 / (rank + k + 1)
            
            if doc_id in rrf_scores:
                rrf_scores[doc_id]["score"] += score
            else:
                rrf_scores[doc_id] = {"text": result['text'], "score": score}
        
        # Sort and return top-k
        sorted_results = sorted(
            rrf_scores.items(),
            key=lambda x: x[1]["score"],
            reverse=True
        )
        
        return [
            {
                "id": doc_id,
                "score": data["score"],
                "text": data["text"]
            }
            for doc_id, data in sorted_results[:top_k]
        ]
    
    def delete_collection(self):
        """Delete entire collection (careful!)."""
        try:
            self.client.delete_collection(self.collection_name)
            self.corpus = []
            self.corpus_ids = []
            self.bm25 = None
            self.added_documents.clear()
            self._init_collection()
            logger.info(f"Deleted and recreated: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "num_documents": info.points_count,
                "num_unique_sources": len(self.added_documents),
                "model_name": self.model_name,
                "url": self.url,
                "collection_name": self.collection_name
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}
    
    def __repr__(self) -> str:
        """String representation."""
        stats = self.get_stats()
        return (
            f"VectorStoreQdrant("
            f"collection='{stats.get('collection_name')}', "
            f"docs={stats.get('num_documents', 0)}, "
            f"url='{self.url}'"
            f")"
        )


# ============================================================================
# Helper: Create store with sensible defaults
# ============================================================================

def create_qdrant_store(collection_name: str = "arrag-eval",
                       model_name: str = "intfloat/multilingual-e5-small",
                       local: bool = True,
                       url: Optional[str] = None,
                       api_key: Optional[str] = None) -> VectorStoreQdrant:
    """
    Convenience function to create a Qdrant store.
    
    Args:
        collection_name: Collection name
        model_name: Embedding model
        local: Use local Docker. Default: True
        url: Custom Qdrant URL (overrides local)
        api_key: API key for cloud Qdrant
    
    Returns:
        VectorStoreQdrant instance
    
    Example:
        >>> # Local (Docker)
        >>> store = create_qdrant_store()
        >>>
        >>> # Cloud
        >>> store = create_qdrant_store(
        ...     local=False,
        ...     url="https://my-instance.qdrant.io",
        ...     api_key="my-api-key"
        ... )
    """
    if url is None:
        url = "http://localhost:6333" if local else None
    
    return VectorStoreQdrant(
        collection_name=collection_name,
        model_name=model_name,
        url=url or "http://localhost:6333",
        api_key=api_key
    )


# ============================================================================
# Docker startup helper
# ============================================================================

def start_qdrant_docker():
    """
    Print Docker command to start Qdrant locally.
    
    Run this command in your terminal:
        docker run -d -p 6333:6333 qdrant/qdrant:latest
    
    Verify it's running:
        curl http://localhost:6333/health
    
    Stop it:
        docker ps  # find container ID
        docker stop <container-id>
    """
    cmd = "docker run -d -p 6333:6333 qdrant/qdrant:latest"
    print(f"\n{'='*60}")
    print("START QDRANT LOCAL")
    print(f"{'='*60}")
    print(f"\nRun this command in your terminal:\n")
    print(f"  {cmd}\n")
    print(f"Verify it's running:\n")
    print(f"  curl http://localhost:6333/health\n")
    print(f"Then initialize store:\n")
    print(f"  store = VectorStoreQdrant()\n")
    print(f"{'='*60}\n")
