from typing import List
import numpy as np
from numpy.typing import NDArray

def chunk_hybrid_semantic(
    text: str,
    embedding_model=None,
    similarity_threshold: float = 0.72,
    max_chunk_size: int = 512,      # tokens (approximate)
    min_chunk_size: int = 3,        # sentences
    tokenizer=None
) -> List[str]:
    """
    Hybrid Chunking (Best of B + C):
    1. Split using Arabic sentence boundary (Strategy B)
    2. Group sentences semantically (Strategy C)
    3. Respect maximum chunk size
    """
    
    # ====================== Step 1: Sentence Splitting (Strategy B) ======================
    try:
        from camel_tools.sentence_splitter import simple_sentence_splitter
        sentences = simple_sentence_splitter(text)
    except ImportError:
        # Fallback splitter
        import re
        sentences = re.split(r'[.!؟؟\n]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    
    if len(sentences) <= 1:
        return [text]
    
    # ====================== Step 2: Get Embeddings (Strategy C) ======================
    if embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            # Good models for Arabic:
            embedding_model = SentenceTransformer('intfloat/multilingual-e5-small')
            # Alternative: 'intfloat/e5-mistral-7b-instruct' (better but heavier)
        except ImportError:
            raise ImportError("Please install: pip install sentence-transformers")
    
    embeddings: NDArray = embedding_model.encode(sentences, normalize_embeddings=True)
    
    # ====================== Step 3: Semantic Grouping with Size Control ======================
    chunks = []
    start = 0
    
    for i in range(1, len(sentences)):
        similarity = float(np.dot(embeddings[i-1], embeddings[i]))
        
        # Check if we should start a new chunk
        should_split = (
            similarity < similarity_threshold or
            _estimate_token_count(sentences[start:i+1], tokenizer) > max_chunk_size
        )
        
        if should_split:
            chunk_sentences = sentences[start:i]
            if len(chunk_sentences) >= min_chunk_size:
                chunk_text = ' '.join(chunk_sentences)
                chunks.append(chunk_text.strip())
            else:
                # If chunk is too small, merge with next one later
                continue
            start = i
    
    # Add the last chunk
    if start < len(sentences):
        chunk_text = ' '.join(sentences[start:])
        if chunk_text.strip():
            chunks.append(chunk_text.strip())
    
    return chunks


# ====================== Helper Function ======================
def _estimate_token_count(sentences: List[str], tokenizer=None) -> int:
    """Rough token estimation (works well for Arabic)"""
    text = ' '.join(sentences)
    if tokenizer is None:
        # Simple whitespace + Arabic characters estimation
        return len(text.split()) + int(len(text) * 0.3)  # Arabic tokens are longer
    else:
        if hasattr(tokenizer, 'encode'):
            return len(tokenizer.encode(text))
        return len(text.split())
    
    
    
    
text = """your long Arabic text here..."""

chunks = chunk_hybrid_semantic(
    text,
    similarity_threshold=0.70,   # Lower = more chunks, Higher = bigger chunks
    max_chunk_size=512,
    min_chunk_size=2
)

print(f"Created {len(chunks)} hybrid chunks\n")
for i, chunk in enumerate(chunks[:5]):
    print(f"Chunk {i+1} ({len(chunk.split())} words):")
    print(chunk[:300] + "..." if len(chunk) > 300 else chunk)
    print("-" * 80)