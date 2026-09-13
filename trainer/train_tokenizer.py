# Train a small BPE tokenizer on your own corpus (optional; the repo ships a
# ready tokenizer under model/). Mirrors minimind's approach via `tokenizers`.
import os, sys, json, argparse

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tokenizers import decoders, models, pre_tokenizers, trainers, Tokenizer

SPECIAL_TOKENS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]

def get_texts(data_path, text_field='text', max_docs=20000):
    with open(data_path, 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            if i >= max_docs: break
            try:
                row = json.loads(line)
                text = row.get(text_field) or row.get('text') or ''
                if text: yield text
            except json.JSONDecodeError:
                continue

def train_tokenizer(data_path, tokenizer_dir, vocab_size=6400):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=SPECIAL_TOKENS,
    )
    tokenizer.train_from_iterator(get_texts(data_path), trainer=trainer)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens(SPECIAL_TOKENS)

    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))

    config = {
        "add_bos_token": False, "add_eos_token": False, "add_prefix_space": False,
        "bos_token": "<|im_start|>", "eos_token": "<|im_end|>",
        "pad_token": "<|endoftext|>", "unk_token": "<|endoftext|>",
        "model_max_length": 131072, "tokenizer_class": "PreTrainedTokenizerFast",
    }
    with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
    print("Tokenizer training completed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./dataset/pretrain.jsonl')
    parser.add_argument('--tokenizer_dir', type=str, default='./model')
    parser.add_argument('--vocab_size', type=int, default=6400)
    args = parser.parse_args()
    train_tokenizer(args.data_path, args.tokenizer_dir, args.vocab_size)
