# run_baseline.py
import os
import json
from dotenv import load_dotenv
from src.config import BASELINE_CONFIG
from src.data.corpus_download import load_arcd, load_wikipedia
from src.data.preprocessing import preprocess_corpus
from src.chunking.strategies import chunk_fixed_size
from src.agent.pipeline import ArRAGPipeline

# New imports for Qdrant
from src.retrieval.qdrant_store import create_qdrant_store

# Load environment variables
load_dotenv()
GEMINI_KEY = os.getenv('GOOGLE_GENAI_API_KEY')

# 1. Load data
arcd = load_arcd()
wiki = load_wikipedia(corpus_size=500)  # Small for testing

# 2. Preprocess
wiki_cleaned = preprocess_corpus(wiki['text'])

# 3. Chunk
chunks = []
for doc in wiki_cleaned:
    doc_chunks = chunk_fixed_size(doc, chunk_size=512, overlap=50)
    chunks.extend(doc_chunks)

# 4. Create Qdrant Store (using config)
config = BASELINE_CONFIG
config.corpus_size = 500
config.num_test_samples = 20

vector_store = create_qdrant_store(
    collection_name=config.qdrant_collection_name,
    model_name=config.embedding_model,
    url=config.qdrant_url,
    api_key=config.qdrant_api_key or os.getenv("QDRANT_API_KEY")
)

# Add documents (idempotent)
vector_store.add_documents(chunks, document_id="wikipedia_corpus")

# 5. Create pipeline
pipeline = ArRAGPipeline(
    config=config,
    vector_store=vector_store,
    api_key=GEMINI_KEY
)

# 6. Run evaluation
test_data = arcd["validation"][:20]
results = pipeline.run_batch(test_data["question"])

# 7. Save outputs
with open("baseline_outputs.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("✓ Complete. Review baseline_outputs.json manually.")
print("Qdrant Stats:", vector_store.get_stats())