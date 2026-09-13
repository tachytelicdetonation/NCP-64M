import os, sys, json, argparse

__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from transformers import AutoTokenizer

def iter_texts(args):
    if args.hf_dataset:
        from datasets import load_dataset
        ds = load_dataset(args.hf_dataset, split=args.hf_split, streaming=True)
        for i, row in enumerate(ds):
            if i >= args.max_docs: break
            text = row.get(args.text_field, '')
            if text: yield text
    else:
        with open(args.data_path, 'r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f):
                if i >= args.max_docs: break
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = row.get(args.text_field) or row.get('text') or ''
                if text: yield text

def main():
    parser = argparse.ArgumentParser(description="Tokenize text data into a packed uint16 .bin stream")
    parser.add_argument('--data_path', type=str, default='../dataset/pretrain.jsonl', help='jsonl with a text field')
    parser.add_argument('--hf_dataset', type=str, default='', help='e.g. roneneldan/TinyStories (streams, no full download)')
    parser.add_argument('--hf_split', type=str, default='train')
    parser.add_argument('--text_field', type=str, default='text')
    parser.add_argument('--max_docs', type=int, default=100000)
    parser.add_argument('--tokenizer_dir', type=str, default='../model')
    parser.add_argument('--out', type=str, default='../dataset/pretrain.bin')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    eos = tokenizer.eos_token_id
    assert eos is not None and eos < 65536

    path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    buf, n_doc, n_tok = [], 0, 0
    with open(path, 'wb') as f:
        for text in iter_texts(args):
            ids = tokenizer(text, add_special_tokens=False).input_ids
            if not ids: continue
            buf.extend(ids + [eos])
            n_doc += 1
            if len(buf) >= 1 << 20:
                f.write(np.asarray(buf, dtype=np.uint16).tobytes())
                n_tok += len(buf); buf = []
                print(f'{n_doc} docs, {n_tok / 1e6:.1f}M tokens', flush=True)
        if buf:
            f.write(np.asarray(buf, dtype=np.uint16).tobytes())
            n_tok += len(buf)
    print(f'done: {n_doc} docs -> {n_tok} tokens -> {path}')

if __name__ == '__main__':
    main()
