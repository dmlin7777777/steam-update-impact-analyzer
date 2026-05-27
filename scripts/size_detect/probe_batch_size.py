# probe_batch_size_real.py
import argparse
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import pandas as pd


def truncate_texts_with_tokenizer(tokenizer, texts, max_length):
    """Truncate texts to max_length tokens (excluding special tokens) using the tokenizer."""
    out = []
    for t in texts:
        try:
            toks = tokenizer.encode(t, add_special_tokens=False)
        except Exception:
            # fallback: simple whitespace-based truncation
            tokens = t.split()
            if len(tokens) > max_length:
                out.append(" ".join(tokens[:max_length]))
            else:
                out.append(t)
            continue

        if len(toks) > max_length:
            toks = toks[:max_length]
            truncated = tokenizer.decode(toks, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            out.append(truncated)
        else:
            out.append(t)
    return out


def try_batch(model, tokenizer, texts, batch_size, max_length):
    try:
        truncated = truncate_texts_with_tokenizer(tokenizer, texts, max_length)
        with torch.no_grad():
            emb = model.encode(truncated, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=False)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True, None
    except RuntimeError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)


def main(sample_csv, text_col='review_content_clean', model_name='all-MiniLM-L6-v2', max_length=128, candidates=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    model = SentenceTransformer(model_name, device=device)
    # load tokenizer separately to perform truncation
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # support csv or excel
    if str(sample_csv).lower().endswith(('.xls', '.xlsx')):
        df = pd.read_excel(sample_csv)
    else:
        df = pd.read_csv(sample_csv, encoding='utf-8')
    texts = df[text_col].fillna('').astype(str).tolist()

    # build a probe set that includes some long and some short samples
    texts_sorted = sorted(texts, key=lambda s: len(s.split()), reverse=True)
    n = min(len(texts), 512)
    probe_texts = texts_sorted[:50] + texts_sorted[-max(0, min(n - 50, len(texts_sorted) - 50)):] if len(texts_sorted) > 50 else texts_sorted[:n]
    # dedupe and limit to n
    seen = set(); final_texts = []
    for t in probe_texts:
        if len(final_texts) >= n:
            break
        if t in seen:
            continue
        seen.add(t); final_texts.append(t)

    print(f"Using {len(final_texts)} probe texts; max_length={max_length}")
    if candidates is None:
        candidates = [512, 256, 128, 64, 32, 16, 8]

    for b in candidates:
        ok, err = try_batch(model, tokenizer, final_texts, batch_size=b, max_length=max_length)
        print(f"batch {b}: {'OK' if ok else 'FAIL'}{'' if err is None else ' -- ' + str(err)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', required=True, help='CSV file path with a text column')
    parser.add_argument('--col', default='review_content_clean')
    parser.add_argument('--model', default='all-MiniLM-L6-v2')
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--candidates', default='512,256,128,64,32,16,8')
    args = parser.parse_args()
    cand = [int(x) for x in args.candidates.split(',') if x.strip()]
    main(args.sample, text_col=args.col, model_name=args.model, max_length=args.max_length, candidates=cand)
