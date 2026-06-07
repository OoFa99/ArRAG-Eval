from datasets import load_dataset
import os

# Define local data path
DATA_PATH = "F:\\arrag-eval\\data"
os.makedirs(DATA_PATH, exist_ok=True)

# Load and save ARCD dataset
arcd = load_dataset("hsseinmz/arcd") # your QA test set source
arcd.save_to_disk(os.path.join(DATA_PATH, "arcd"))
print("ARCD dataset saved to:", os.path.join(DATA_PATH, "arcd"))

# Load and save Arabic Wikipedia dataset
wiki_ar = load_dataset(
    "wikimedia/wikipedia", 
    "20231101.ar", 
    split="train[:5000]"
)
wiki_ar.save_to_disk(os.path.join(DATA_PATH, "wiki_ar"))
print("Arabic Wikipedia dataset saved to:", os.path.join(DATA_PATH, "wiki_ar"))