from datasets import load_dataset
from transformers import DistilBertTokenizer

raw_dataset = load_dataset("imdb")
print (raw_dataset["test"])


tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
