import os

# Define local data path
DATA_PATH = "~/arrag-eval/data"
ARCD_CACHE_NAME = "arcd"
WIKIPEDIA_CACHE_NAME = "wiki_ar"


def load_arcd():
    """Load ARCD dataset from disk or download and cache it."""
    from datasets import load_dataset, load_from_disk

    arcd_path = os.path.join(DATA_PATH, ARCD_CACHE_NAME)
    
    if os.path.exists(arcd_path):
        # Load from disk if already cached
        return load_from_disk(arcd_path)
    else:
        # Download and cache
        print("Downloading ARCD dataset...")
        arcd = load_dataset("hsseinmz/arcd")
        os.makedirs(DATA_PATH, exist_ok=True)
        arcd.save_to_disk(arcd_path)
        print("ARCD dataset saved to:", arcd_path)
        return arcd


def load_wikipedia(corpus_size=500):
    """Load Arabic Wikipedia dataset from disk or download and cache it."""
    from datasets import load_dataset, load_from_disk

    wiki_path = os.path.join(DATA_PATH, WIKIPEDIA_CACHE_NAME)
    
    if os.path.exists(wiki_path):
        # Load from disk if already cached
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
        os.makedirs(DATA_PATH, exist_ok=True)
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
