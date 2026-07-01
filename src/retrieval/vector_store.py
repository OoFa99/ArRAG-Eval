from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from rank_bm25 import BM25Okapi

class PineConeStore:
    def __init__(self, api_key: str, index_name: str, model_name: str = "intfloat/multilingual-e5-small"):
        """
        Initialize the PineConeStore with both semantic (vector) and keyword (BM25) search capabilities.
        
        This constructor sets up:
        1. A SentenceTransformer model for generating dense embeddings (semantic search)
        2. A Pinecone cloud index for storing and querying vectors
        3. BM25 components for keyword-based retrieval
        
        Args:
            api_key (str): Pinecone API key for authentication
            index_name (str): Name of the Pinecone index to use/create
            model_name (str): HuggingFace model name for embeddings. Defaults to 
                            'intfloat/multilingual-e5-small' (768-dimensional, multilingual)
        
        Raises:
            Exception: If Pinecone API authentication fails
        
        Example:
            >>> store = PineConeStore(
            ...     api_key="your-pinecone-key",
            ...     index_name="arrag-eval",
            ...     model_name="intfloat/multilingual-e5-small"
            ... )
        """
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_embedding_dimension()
        
        print("Connecting to Pinecone...")
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        
        # Check if the index exists; if not, create it
        existing_indexes = [index_info["name"] for index_info in self.pc.list_indexes()]
        if self.index_name not in existing_indexes:
            print(f"Creating Pinecone index '{self.index_name}'...")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1" # Update this to match your Pinecone project region
                )
            )
            
        self.index = self.pc.Index(self.index_name)
        
        # BM25 components
        self.bm25 = None
        self.corpus = []          # Store raw chunks for BM25
        self.corpus_ids = []      # Store corresponding ids for BM25
        
    def add_documents(self, chunks: List[str], document_id: str):
        """
        Add document chunks to both semantic (Pinecone) and keyword (BM25) indexes.
        
        This function performs dual indexing:
        - Creates embeddings for semantic search in Pinecone
        - Tokenizes and indexes chunks for keyword search via BM25
        
        Both indexes maintain the same chunk IDs for result correlation in hybrid queries.
        
        Args:
            chunks (List[str]): List of text chunks (typically from document splitting)
            document_id (str): Unique identifier for the source document (used in chunk IDs)
        
        Returns:
            None
        
        Notes:
            - Empty chunk lists are silently ignored
            - Vectors are batched in groups of 100 for Pinecone efficiency
            - BM25 index is rebuilt from scratch after each addition (suitable for medium corpora)
            - Chunk IDs follow the pattern: f"{document_id}_chunk_{index}"
        
        Example:
            >>> chunks = ["Text of chunk 1", "Text of chunk 2", "Text of chunk 3"]
            >>> store.add_documents(chunks, document_id="document_001")
        """
        if not chunks:
            return

        print(f"Adding {len(chunks)} chunks for document: {document_id}")
        
        # === IDEMPOTENCY CHECK ===
        try:
            stats = self.index.describe_index_stats()
            total_vectors = stats.get('total_vector_count', 0)
            
            # Simple but effective check for this corpus
            # You can make it more robust by querying a few IDs
            if total_vectors > 0:
                # Optional: check if this specific document already exists
                sample_id = f"{document_id}_chunk_0"
                try:
                    fetch_result = self.index.fetch(ids=[sample_id])
                    if sample_id in fetch_result.get('vectors', {}):
                        print(f"✅ Document '{document_id}' already exists in index. Skipping add_documents.")
                        return
                except Exception:
                    pass  # fallback to total count
                    
            print(f"Index currently has {total_vectors} vectors. Proceeding with upsert...")
        except Exception as e:
            print(f"Warning: Could not check index stats: {e}")

        # 1. Semantic Indexing (Pinecone)
        embeddings = self.model.encode(chunks, normalize_embeddings=True)
        
        vectors_to_upsert = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{document_id}_chunk_{i}"
            metadata = {"text": chunk, "source_document": document_id}
            vectors_to_upsert.append((chunk_id, emb.tolist(), metadata))

        # Batch upsert
        batch_size = 100
        for i in range(0, len(vectors_to_upsert), batch_size):
            self.index.upsert(vectors=vectors_to_upsert[i:i + batch_size])

        # 2. BM25 Indexing
        self.corpus.extend(chunks)
        self.corpus_ids.extend([f"{document_id}_chunk_{i}" for i in range(len(chunks))])
        
        # Rebuild BM25 (simple approach)
        tokenized_corpus = [self._tokenize_arabic(chunk) for chunk in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

        print(f"✅ Successfully added {len(chunks)} chunks.")
        
    def _tokenize_arabic(self, text: str) -> List[str]:
        """
        Tokenize text supporting both Arabic and English languages.
        
        This is a simple yet effective tokenizer that handles common preprocessing:
        - Converts to lowercase
        - Removes newlines
        - Splits on whitespace
        
        Args:
            text (str): Input text in Arabic, English, or mixed
        
        Returns:
            List[str]: List of lowercase tokens
        
        Notes:
            - This basic implementation handles general use cases well
            - For production with complex Arabic morphology, consider libraries like:
              * Farasa (Fast and Accurate Arabic Segmentation Toolkit)
              * Camel Tools (Comprehensive Computational Linguistics Toolkit)
              * MADAMIRA (Morphological Analysis and Disambiguation for Arabic)
        
        Example:
            >>> tokens = store._tokenize_arabic("مرحبا بك في النظام")
            >>> tokens
            ['مرحبا', 'بك', 'في', 'النظام']
        """
        # Basic cleanup: lowercase, remove newlines, split on whitespace
        tokens = text.lower().replace('\n', ' ').split()
        return tokens
        
    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform semantic search using dense vector embeddings (Pinecone).
        
        This method finds semantically similar chunks using cosine similarity in the vector space.
        It captures meaning and context, making it effective for paraphrases and synonyms.
        
        Args:
            query_text (str): The search query (natural language)
            top_k (int): Number of top results to return. Defaults to 5
        
        Returns:
            List[Dict[str, Any]]: List of matching chunks with metadata, sorted by relevance score
                - id (str): Unique chunk identifier
                - score (float): Cosine similarity score (0-1, higher is better)
                - text (str): The original chunk text
                - source (str): Source document ID
        
        Example:
            >>> results = store.query("What is machine learning?", top_k=3)
            >>> for result in results:
            ...     print(f"Score: {result['score']:.3f} | {result['text'][:50]}...")
        """
        # Encode the query text into a 768-dimensional embedding vector
        # normalize_embeddings=True ensures vectors are unit length for cosine similarity
        query_vector = self.model.encode([query_text], normalize_embeddings=True)[0].tolist()
        
        # Query Pinecone index with the embedding vector
        # Returns top_k most similar vectors based on cosine similarity
        response = self.index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True  # Include chunk text and source document info
        )
        
        # Format results: extract id, score, text, and source from Pinecone response
        # Results are already sorted by similarity score (descending)
        return [
            {
                "id": match['id'],
                "score": match['score'],  # Cosine similarity (0-1)
                "text": match['metadata']['text'],  # Original chunk text
                "source": match['metadata'].get('source_document')  # Source document
            }
            for match in response['matches']
        ]

    def bm25_query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform keyword-based search using BM25 algorithm (local, in-memory index).
        
        BM25 (Best Matching 25) is a probabilistic ranking function that excels at:
        - Exact keyword matching
        - Document relevance based on term frequency and inverse document frequency (IDF)
        - Handling queries with specific technical terms
        
        Args:
            query_text (str): The search query (keywords or phrases)
            top_k (int): Number of top results to return. Defaults to 5
        
        Returns:
            List[Dict[str, Any]]: List of matching chunks ranked by BM25 score
                - id (str): Unique chunk identifier
                - score (float): BM25 relevance score (higher is better, unbounded)
                - text (str): The original chunk text
                - source (str): Always "bm25" to indicate search method
        
        Example:
            >>> results = store.bm25_query("neural networks deep learning", top_k=5)
            >>> for result in results:
            ...     print(f"BM25 Score: {result['score']:.2f} | {result['text'][:50]}...")
        """
        # Return empty list if BM25 index hasn't been initialized (no documents added yet)
        if not self.bm25:
            return []
        
        # Tokenize the query using the same tokenizer as the corpus
        # Ensures consistency between query and indexed documents
        tokenized_query = self._tokenize_arabic(query_text)
        
        # Calculate BM25 scores for all chunks in the corpus
        # Each score represents relevance: accounts for term frequency and IDF
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get indices of top_k highest scoring chunks
        # sorted() returns indices sorted by score in descending order
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        # Build result list with chunk metadata and BM25 scores
        # Maintains order (best matches first)
        return [
            {
                "id": self.corpus_ids[i],  # Retrieve ID from parallel list
                "score": float(scores[i]),  # BM25 relevance score
                "text": self.corpus[i],  # Retrieve original text from corpus
                "source": "bm25"  # Indicate this result came from keyword search
            }
            for i in top_indices
        ]

    def hybrid_query(self, query_text: str, top_k: int = 10, k: int = 60) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining semantic and keyword-based retrieval.
        
        This method uses Reciprocal Rank Fusion (RRF) to combine results from:
        1. Semantic search (Pinecone): Captures meaning and context
        2. Keyword search (BM25): Captures exact terms and technical specificity
        
        RRF scoring formula: score = 1 / (rank + k) for each result
        - Scores from both methods are summed for documents appearing in both
        - Parameter 'k' helps balance results (higher k reduces rank impact)
        
        Args:
            query_text (str): The search query (natural language or keywords)
            top_k (int): Final number of results to return. Defaults to 10
            k (int): RRF parameter controlling rank weighting. Defaults to 60
                   - Higher k: Reduces the impact of ranking position
                   - Lower k: Emphasizes high-ranked results more
        
        Returns:
            List[Dict[str, Any]]: Final combined results sorted by RRF score
                - id (str): Unique chunk identifier
                - score (float): Combined RRF score from both methods
                - text (str): The original chunk text
        
        Notes:
            - Retrieves 2*top_k results from each method before fusion
            - Documents appearing in both rankings get boosted scores
            - Final output limited to top_k results
        
        Example:
            >>> results = store.hybrid_query("machine learning algorithms", top_k=5)
            >>> for result in results:
            ...     print(f"RRF Score: {result['score']:.4f} | {result['text'][:50]}...")
        """
        # Step 1: Retrieve candidate results from both search methods
        # Use 2*top_k to ensure good coverage for fusion (some results may overlap)
        semantic_results = self.query(query_text, top_k=top_k * 2)
        bm25_results = self.bm25_query(query_text, top_k=top_k * 2)

        # Step 2: Apply Reciprocal Rank Fusion (RRF) scoring
        # Dictionary to store combined scores and text for each unique chunk
        rrf_scores = {}
        
        # Process semantic search results: assign RRF scores based on rank position
        # Formula: 1 / (rank + k) where rank starts at 0
        # Higher-ranked results (lower rank number) get higher scores
        for rank, result in enumerate(semantic_results):
            doc_id = result['id']
            rrf_score = 1 / (rank + k + 1)  # RRF score for this ranking position
            rrf_scores[doc_id] = {"text": result['text'], "score": rrf_score}

        # Process BM25 results: add scores to existing entries or create new ones
        # If a document appears in both rankings, scores are summed (boosting effect)
        for rank, result in enumerate(bm25_results):
            doc_id = result['id']
            rrf_score = 1 / (rank + k)  # RRF score for this ranking position
            
            if doc_id in rrf_scores:
                # Document appeared in both semantic AND keyword search: boost score
                rrf_scores[doc_id]["score"] += rrf_score
            else:
                # Document only appeared in BM25 results: create new entry
                rrf_scores[doc_id] = {
                    "text": result['text'],
                    "score": rrf_score
                }

        # Step 3: Sort by combined RRF score (descending) and return top_k results
        # Results with high scores in both methods rank highest
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1]["score"], reverse=True)
        
        # Format final results: extract ID, score, and text
        # Apply final top_k limit to return exactly the requested number of results
        return [
            {
                "id": doc_id,
                "score": data["score"],  # Combined RRF score
                "text": data["text"]
            }
            for doc_id, data in sorted_results[:top_k]
        ]