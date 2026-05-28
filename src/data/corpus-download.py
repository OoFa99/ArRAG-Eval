from datasets import load_dataset

arcd = load_dataset("hsseinmz/arcd") # your QA test set source

wiki_ar = load_dataset(
    "wikimedia/wikipedia", 
    "20231101.ar", 
    split="train[:5000]"
)