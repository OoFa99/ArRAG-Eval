from datasets import load_dataset
import os

# Define local data path
DATA_PATH = "~/arrag-eval/data"
os.makedirs(DATA_PATH, exist_ok=True)


def load_arcd():
    """Load ARCD dataset from disk or download and cache it."""
    arcd_path = os.path.join(DATA_PATH, "arcd")
    
    if os.path.exists(arcd_path):
        # Load from disk if already cached
        from datasets import load_from_disk
        return load_from_disk(arcd_path)
    else:
        # Download and cache
        print("Downloading ARCD dataset...")
        arcd = load_dataset("hsseinmz/arcd")
        arcd.save_to_disk(arcd_path)
        print("ARCD dataset saved to:", arcd_path)
        return arcd


def load_wikipedia(corpus_size=500):
    """Load Arabic Wikipedia dataset from disk or download and cache it."""
    wiki_path = os.path.join(DATA_PATH, "wiki_ar")
    
    if os.path.exists(wiki_path):
        # Load from disk if already cached
        from datasets import load_from_disk
        wiki = load_from_disk(wiki_path)
        # Return only the requested corpus size
        if corpus_size < len(wiki):
            return wiki.select(range(corpus_size))
        return wiki
    else:
        # Download and cache full dataset (5000 samples)
        print(f"Downloading Arabic Wikipedia ({corpus_size} samples)...")
        wiki_ar = load_dataset(
            "wikimedia/wikipedia", 
            "20231101.ar", 
            split="train[:5000]"
        )
        wiki_ar.save_to_disk(wiki_path)
        print("Arabic Wikipedia dataset saved to:", wiki_path)
        # Return only the requested corpus size
        if corpus_size < len(wiki_ar):
            return wiki_ar.select(range(corpus_size))
        return wiki_ar


if __name__ == "__main__":
    # Download and cache both datasets on first run
    print("Downloading datasets...")
    load_arcd()
    load_wikipedia()
    print("✓ All datasets cached locally")