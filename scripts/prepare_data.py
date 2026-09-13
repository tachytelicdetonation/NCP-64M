import os, sys, json, argparse, random, threading, multiprocessing as mp
from collections import defaultdict

__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

_TOK = None
_EOS = None

def _init_tok(tok_dir):
    """Load the lightweight Rust tokenizer in workers — no torch/transformers,
    so RSS stays ~50MB per process instead of ~2GB."""
    global _TOK, _EOS
    from tokenizers import Tokenizer
    cfg = json.load(open(os.path.join(tok_dir, 'tokenizer_config.json')))
    _TOK = Tokenizer.from_file(os.path.join(tok_dir, 'tokenizer.json'))
    _EOS = _TOK.token_to_id(cfg['eos_token'])

def _tokenize(job):
    """(src, is_val, texts) -> (src, is_val, [uint16 ids per doc])."""
    src, is_val, texts = job
    out = []
    for text in texts:
        ids = _TOK.encode(text, add_special_tokens=False).ids
        if ids:
            ids.append(_EOS)
            out.append(np.asarray(ids, dtype=np.uint16))
    return src, is_val, out


def parse_spec(spec):
    """NAME=WEIGHT:hf:DATASET:SPLIT[:FIELD] | NAME=WEIGHT:file:PATH[:FIELD]"""
    name, rest = spec.split('=', 1)
    parts = rest.split(':')
    weight, kind = float(parts[0]), parts[1]
    if kind == 'hf':
        ds = parts[2].split('@', 1)  # DATASET[@CONFIG]
        split, field = parts[3], (parts[4] if len(parts) > 4 else 'text')
        return name, weight, ('hf', ds, split, field)
    if kind == 'file':
        return name, weight, ('file', parts[2], 'train', parts[3] if len(parts) > 3 else 'text')
    raise ValueError(f'unknown source kind in {spec!r}')


def parse_ood(spec):
    """NAME:hf:DATASET:SPLIT[:FIELD] | NAME:file:PATH[:FIELD]"""
    return parse_spec(spec.replace(':', '=0:', 1))


def open_iter(locator):
    kind, a, split, field = locator
    if kind == 'hf':
        from datasets import load_dataset
        rows = iter(load_dataset(*a, split=split, streaming=True))
    else:
        rows = (json.loads(l) for l in open(a, encoding='utf-8', errors='ignore'))
    for row in rows:
        text = row.get(field) or row.get('text') or ''
        if text:
            yield text


class BinWriter:
    def __init__(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.f = open(path, 'wb')
        self.buf, self.n_tok = [], 0

    def write(self, ids):
        self.buf.extend(ids.tolist())
        if len(self.buf) >= 1 << 20:
            self._flush()

    def _flush(self):
        self.f.write(np.asarray(self.buf, dtype=np.uint16).tobytes())
        self.n_tok += len(self.buf)
        self.buf = []

    def close(self):
        if self.buf:
            self._flush()
        self.f.close()


def main():
    p = argparse.ArgumentParser(description="Parallel tokenizer + weighted multi-source mixer -> packed uint16 .bin")
    p.add_argument('--source', action='append', required=True,
                   help='NAME=WEIGHT:hf:DATASET:SPLIT[:FIELD] or NAME=WEIGHT:file:PATH[:FIELD]; repeat per source')
    p.add_argument('--ood', action='append', default=[],
                   help='NAME:hf:DATASET:SPLIT[:FIELD] — val-only out-of-distribution bin')
    p.add_argument('--ood_tokens', type=int, default=300_000)
    p.add_argument('--target_tokens', type=int, required=True)
    p.add_argument('--val_frac', type=float, default=0.002)
    p.add_argument('--val_cap', type=int, default=400_000, help='max val tokens per source')
    p.add_argument('--chunk', type=int, default=48, help='docs per tokenize task')
    p.add_argument('--workers', type=int, default=min(mp.cpu_count() - 1, 8))
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--tokenizer_dir', type=str, default='./model')
    p.add_argument('--out', type=str, default='./dataset/pretrain.bin')
    p.add_argument('--val_dir', type=str, default='./dataset/val')
    args = p.parse_args()

    rng = random.Random(args.seed)
    specs = [parse_spec(s) for s in args.source]
    ood_specs = [parse_ood(s) for s in args.ood]

    srcs = [[name, w, open_iter(loc)] for name, w, loc in specs]
    train_w = BinWriter(args.out)
    val_w = {name: BinWriter(os.path.join(args.val_dir, f'{name}.bin')) for name, _, _ in specs}
    for name, _, loc in ood_specs:
        val_w[name] = BinWriter(os.path.join(args.val_dir, f'ood_{name}.bin'))

    emitted = defaultdict(int)
    sched = defaultdict(int)      # estimated tokens scheduled but not yet emitted
    avg_len = {name: 800.0 for name, _, _ in specs}  # running mean doc length
    docs = defaultdict(int)
    val_tok = defaultdict(int)
    state = {'total': 0}

    # Bound job lookahead — imap_unordered would otherwise drain the generator
    # far ahead of results. The WFQ pick uses sched (incl. in-flight estimates),
    # so balance holds regardless of this bound.
    inflight = threading.Semaphore(args.workers * 2)

    def jobs():
        """Weighted-fair generator on estimated scheduled tokens -> (src, is_val, [texts])."""
        while state['total'] < args.target_tokens:
            live = [s for s in srcs if s[2] is not None]
            if not live:
                return
            wsum = sum(s[1] for s in live)
            T = state['total'] + sum(sched.values())  # emitted + in-flight estimate
            s = max(live, key=lambda s: (s[1] / wsum) * (T + 1)
                    - (emitted[s[0]] + sched[s[0]]))
            groups = defaultdict(list)
            chars = 0
            while sum(len(t) for t in groups.values()) < args.chunk and chars < 400_000:
                try:
                    text = next(s[2])
                except StopIteration:
                    s[2] = None
                    break
                chars += len(text)
                is_val = rng.random() < args.val_frac and val_tok[s[0]] < args.val_cap
                groups[is_val].append(text)
            if not groups:
                continue  # exhausted source dropped; pick among remaining live ones
            for is_val, texts in groups.items():
                if not is_val:
                    sched[s[0]] += int(avg_len[s[0]] * len(texts))
                inflight.acquire()
                yield (s[0], is_val, texts)

    last_report = 0
    with mp.Pool(args.workers, initializer=_init_tok, initargs=(args.tokenizer_dir,)) as pool:
        for src, is_val, arrays in pool.imap_unordered(_tokenize, jobs(), chunksize=1):
            inflight.release()
            if not is_val:
                sched[src] -= int(avg_len[src] * len(arrays))
            actual = 0
            for ids in arrays:
                (val_w[src] if is_val else train_w).write(ids)
                if is_val:
                    val_tok[src] += len(ids)
                else:
                    emitted[src] += len(ids)
                    actual += len(ids)
                    docs[src] += 1
                    state['total'] += len(ids)
            if not is_val and actual:
                avg_len[src] += (actual / len(arrays) - avg_len[src]) * 0.1
            if state['total'] - last_report >= 50_000_000:
                last_report = state['total']
                shares = {n: f'{emitted[n] / max(state["total"], 1):.1%}' for n, _, _ in specs}
                print(f'{state["total"] / 1e6:.0f}M tokens | shares {shares}', flush=True)

    # OOD bins — small, tokenize inline
    _init_tok(args.tokenizer_dir)
    for name, _, loc in ood_specs:
        n = 0
        for text in open_iter(loc):
            ids = _TOK.encode(text, add_special_tokens=False).ids
            if not ids:
                continue
            ids.append(_EOS)
            arr = np.asarray(ids, dtype=np.uint16)
            val_w[name].write(arr)
            n += len(arr)
            if n >= args.ood_tokens:
                break
        print(f'ood/{name}: {n} tokens', flush=True)

    train_w.close()
    for w in val_w.values():
        w.close()

    total = state['total']
    print(f'\ndone: {total} train tokens -> {args.out}')
    for name, _, _ in specs:
        print(f'  {name}: {docs[name]} docs, {emitted[name]} tok ({emitted[name] / max(total, 1):.1%}), '
              f'val {val_tok[name]} tok')
    for name, _, _ in ood_specs:
        print(f'  ood_{name}: {val_w[name].n_tok} tok (val only)')


if __name__ == '__main__':
    main()
