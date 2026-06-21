from typing import List
import re
import numpy as np
from numpy.typing import NDArray

# ========================= STRATEGY A =========================
def chunk_fixed_size(text: str, 
                     chunk_size: int = 512, 
                     overlap: int = 50, 
                     tokenizer=None) -> List[str]:
    """ Splits the input text into chunks of a specified size with optional overlap."""
    if tokenizer is None:
        # Default to simple whitespace tokenization if no tokenizer is provided
        tokens = text.split()
    else:
        tokens = tokenizer.encode(text) if hasattr(tokenizer, 'encode') else text.split()
    
    chunks = []
    start = 0
    
    while start < len(tokens):
        chunk_tokens = tokens[start:start + chunk_size]
        # Convert back to text
        if hasattr(tokenizer, 'decode'):
            chunk_text = tokenizer.decode(chunk_tokens)
        else:
            chunk_text = ' '.join(chunk_tokens)
        
        chunks.append(chunk_text.strip())
        start += chunk_size - overlap  # slide with overlap
    
    return chunks

# ========================= STRATEGY B =========================
def chunk_semantic(text: str,
                    embedding_model=None,
                    similarity_threshold: float = 0.7,
                    min_chunk_size: int = 3) -> List[str]:
    """Semantic chunking: embed sentences and split when similarity drops or size exceeds limit."""
    
    if embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            embedding_model = SentenceTransformer('intfloat/multilingual-e5-small')  # Good for Arabic
        except ImportError:
            raise ImportError("Install sentence-transformers: pip install sentence-transformers")
        
    # 1. Split into sentences (fallback to simple splitter if no Arabic-specific tool)
    sentences = re.split(r'[.!؟\n]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) < 2:
        return [text]
    
    # 2. Compute embeddings
    embeddings: NDArray = embedding_model.encode(sentences, normalize_embeddings=True)
    
    # 3. Find breakpoints using cosine similarity
    chunks = []
    start = 0
    
    for i in range(1, len(sentences)):
        similarity = float(np.dot(embeddings[i-1], embeddings[i]))
        
        if similarity < similarity_threshold:
            # Big semantic shift → create chunk
            chunk_text = ' '.join(sentences[start:i])
            if len(chunk_text.split()) >= min_chunk_size or i - start >= min_chunk_size:
                chunks.append(chunk_text)
            start = i
    
    # Last chunk
    if start < len(sentences):
        chunks.append(' '.join(sentences[start:]))
    
    return chunks

# ========================= STRATEGY C =========================
def chunk_document_structure(text: str) -> List[str]:
    """
    Structure-aware chunking (headings, sections).
    Works especially well with Wikipedia / Markdown / HTML-like text.
    """
    # Common Arabic + English heading patterns
    heading_pattern = re.compile(
        r'^(#{1,6})\s+(.+)$|^(\d+\.)\s+(.+)$|^([أ-يa-zA-Z].{0,100}?[:：])\s*$',
        re.MULTILINE
    )
    
    # Find all structural breaks
    positions = []
    for match in heading_pattern.finditer(text):
        positions.append(match.start())
    
    positions = sorted(set([0] + positions + [len(text)]))
    
    chunks = []
    for i in range(len(positions) - 1):
        start = positions[i]
        end = positions[i + 1]
        chunk = text[start:end].strip()
        if chunk and len(chunk) > 50:  # avoid tiny fragments
            chunks.append(chunk)
    
    # Fallback if no structure detected
    if not chunks:
        return [text]
    
    return chunks

# ========================= STRATEGY D =========================
def chunk_hybrid_semantic():
    pass