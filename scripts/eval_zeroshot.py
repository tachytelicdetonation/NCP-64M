"""Zero-shot likelihood-scored benchmarks: BoolQ, PIQA, ARC-Easy, OpenBookQA,
WinoGrande, HellaSwag. Scores each choice by sum log P(choice | context);
reports acc and acc_norm (per-token normalized, matching lm-eval convention).

Works on NCP checkpoints (--ckpt out/pretrain_768.pth) and on any HF causal
LM (--hf_model jingyaogong/minimind3) for apples-to-apples comparison.
"""
import os, sys, argparse, json

__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F

TASKS = {}


def task(name):
    def deco(fn):
        TASKS[name] = fn
        return fn
    return deco


def _hf(ds_id, *cfgs, split='validation'):
    from datasets import load_dataset
    for cfg in cfgs or [None]:
        try:
            return load_dataset(ds_id, cfg, split=split)
        except Exception:
            continue
    return load_dataset(ds_id, split=split)


@task('boolq')
def boolq():
    for r in _hf('google/boolq', 'super_glue'):
        yield r['passage'] + '\n' + r['question'] + '?', [' yes', ' no'], int(r['answer'])


@task('piqa')
def piqa():
    from datasets import load_dataset
    for r in load_dataset('ybisk/piqa', revision='refs/convert/parquet', split='validation'):
        yield 'Goal: ' + r['goal'] + '\n', [' ' + r['sol1'], ' ' + r['sol2']], int(r['label'])


@task('arc_easy')
def arc_easy():
    for r in _hf('allenai/ai2_arc', 'ARC-Easy'):
        gold = r['choices']['label'].index(r['answerKey'])
        yield 'Question: ' + r['question'] + '\nAnswer:', [' ' + t for t in r['choices']['text']], gold


@task('openbookqa')
def openbookqa():
    for r in _hf('allenai/openbookqa', 'main'):
        gold = r['choices']['label'].index(r['answerKey'])
        yield 'Question: ' + r['question_stem'] + '\nAnswer:', [' ' + t for t in r['choices']['text']], gold


@task('winogrande')
def winogrande():
    for r in _hf('allenai/winogrande', 'winogrande_xl'):
        pre, post = r['sentence'].split('_')
        opt = r['option' + r['answer']]
        other = r['option1'] if r['answer'] == '2' else r['option2']
        yield pre, [opt + post, other + post], 0


@task('hellaswag')
def hellaswag():
    for r in _hf('Rowan/hellaswag'):
        ctx = r['ctx'].replace('[header]', '').replace('[step]', '').strip()
        yield ctx + '\n', [' ' + e.strip() for e in r['endings']], int(r['label'])


def encode(tok, text):
    return tok(text, add_special_tokens=False).input_ids


@torch.no_grad()
def choice_logprobs(model_forward, tok, ctx, choices, device, max_len=2048):
    """-> [(sum_logprob, n_tokens) per choice]"""
    ctx_ids = encode(tok, ctx)
    out = []
    for ch in choices:
        ch_ids = encode(tok, ch)
        ids = (ctx_ids + ch_ids)[-max_len:]
        if len(ids) < 2:
            out.append((-1e9, 1))
            continue
        n_ctx = min(len(ctx_ids), len(ids) - len(ch_ids))
        x = torch.tensor([ids], device=device)
        logits = model_forward(x).float()
        lp = F.log_softmax(logits[0, n_ctx - 1:-1], dim=-1)
        tgt = torch.tensor(ids[n_ctx:], device=device)
        ll = lp.gather(-1, tgt.unsqueeze(-1)).sum().item()
        out.append((ll, len(tgt)))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, default='', help='NCP checkpoint .pth')
    p.add_argument('--hf_model', type=str, default='', help='HF model id/path for comparison')
    p.add_argument('--tasks', nargs='*', default=list(TASKS))
    p.add_argument('--limit', type=int, default=400)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else
                  ('mps' if torch.backends.mps.is_available() else 'cpu'))
    p.add_argument('--save_dir', default='out')
    p.add_argument('--weight', default='pretrain')
    p.add_argument('--hidden_size', default=768, type=int)
    p.add_argument('--out_json', type=str, default='')
    args = p.parse_args()

    from transformers import AutoTokenizer
    if args.hf_model:
        tok = AutoTokenizer.from_pretrained(args.hf_model)
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            args.hf_model, trust_remote_code=True, torch_dtype=torch.bfloat16).to(args.device).eval()
        fwd = lambda x: model(x).logits
    else:
        from model.model_ncp import NCPConfig, NCPForCausalLM
        tok = AutoTokenizer.from_pretrained('./model')
        model = NCPForCausalLM(NCPConfig(hidden_size=args.hidden_size))
        ckpt = args.ckpt or f'./{args.save_dir}/{args.weight}_{args.hidden_size}.pth'
        model.load_state_dict(torch.load(ckpt, map_location=args.device), strict=True)
        model = model.to(args.device).eval()
        fwd = lambda x: model(x)['logits']

    results = {}
    for name in args.tasks:
        n = n_acc = n_norm = 0
        for i, (ctx, choices, gold) in enumerate(TASKS[name]()):
            if i >= args.limit:
                break
            scores = choice_logprobs(fwd, tok, ctx, choices, args.device)
            pred = max(range(len(scores)), key=lambda j: scores[j][0])
            pred_n = max(range(len(scores)), key=lambda j: scores[j][0] / max(scores[j][1], 1))
            n_acc += pred == gold
            n_norm += pred_n == gold
            n += 1
        results[name] = {'acc': n_acc / max(n, 1), 'acc_norm': n_norm / max(n, 1), 'n': n}
        print(f"{name:12s} acc={results[name]['acc']:.3f} acc_norm={results[name]['acc_norm']:.3f} (n={n})", flush=True)

    print(json.dumps(results, indent=2))
    if args.out_json:
        json.dump(results, open(args.out_json, 'w'), indent=2)


if __name__ == '__main__':
    main()
