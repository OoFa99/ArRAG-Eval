from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from rank_bm25 import BM25Okapi

class PineConeStore:
    def __init__(self, api_key: str, index_name: str, model_name: str = "intfloat/multilingual-e5-small"):
        """
        Initialize the embedding model and connect to the Pinecone index + BM25 support.
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
        """Add documents to both Pinecone (semantic) and BM25 (keyword)."""
        if not chunks:
            return

        print(f"Adding {len(chunks)} chunks for document: {document_id}")

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
        """Simple Arabic tokenizer (can be improved with more sophisticated libraries)."""
        # Basic cleanup and splitting - can be enhanced with libraries like Farasa or Camel Tools
        tokens = text.lower().replace('\n', ' ').split()
        return tokens
        
    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Embed the query and search Pinecone for the most similar chunks.
        """
        # Embed the query
        query_vector = self.model.encode([query_text])[0].tolist()
        
        # Search Pinecone and request metadata back
        response = self.index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True
        )
        
        # Extract and format the results
        results = []
        for match in response['matches']:
            results.append({
                "score": match['score'], # The similarity score
                "text": match['metadata']['text'], # The original chunk
                "id": match['id']
            })
            
        return results