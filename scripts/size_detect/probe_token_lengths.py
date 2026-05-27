# save as probe_token_lengths.py
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np
import pandas as pd
import sys

# load your sample CSV/df instead of this placeholder
df = pd.read_excel(r"C:\Users\12932\Desktop\nus\BAP\data\combined\combined_strategy_reviews.xlsx")
texts = df['review_content_clean'].fillna('').astype(str).tolist()


model_name = "all-MiniLM-L6-v2"
model = SentenceTransformer(model_name, device='cpu')  # tokenization only; CPU ok
tokenizer = model._first_module().tokenizer  # HF tokenizer

lengths = []
for t in tqdm(texts):
    tokens = tokenizer.encode(t, add_special_tokens=True)
    lengths.append(len(tokens))

lengths = np.array(lengths)
for p in [50, 75, 90, 95, 99]:
    print(f"{p}th percentile token length: {np.percentile(lengths, p):.0f}")
print("Suggested max_length (round to 64/128/192/256):", int(np.percentile(lengths, 90)))