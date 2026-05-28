from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

class PineConeStore:
    def __init__(self, api_key: str, index_name: str, model_name: str):
        """
        Initialize the embedding model and connect to the Pinecone index.
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
        
    def add_documents(self, chunks: List[str], document_id: str):
        """
            Convert chunks to vectors and upsert them to Pinecone with metadata.
        """
        if not chunks:
            return
                
        print(f"Generating embeddings for {len(chunks)} chunks...")
        embeddings = self.model.encode(chunks)
        
        # Format data for Pinecone: list of tuples (id, vector, metadata)
        vectors_to_upsert = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{document_id}_chunk_{i}"
            metadata = {"text": chunk, "source_document": document_id}
            
            # Embeddings must be converted from numpy arrays to standard Python lists
            vectors_to_upsert.append((chunk_id, emb.tolist(), metadata))
        
        # Upsert in batches (recommended practice for APIs)
        batch_size = 100
        print("Upserting vectors to Pinecone...")
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i:i + batch_size]
            self.index.upsert(vectors=batch)
            
        print("Indexing complete.")
        
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