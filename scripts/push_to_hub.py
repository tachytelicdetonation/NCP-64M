"""Push a trained NCP checkpoint + tokenizer + model code to the HF hub.

Sets config.auto_map so the repo loads via trust_remote_code:
    AutoModelForCausalLM.from_pretrained('user/repo', trust_remote_code=True)
"""
import os, sys, argparse, shutil, tempfile

__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, default='out/pretrain_768.pth')
    p.add_argument('--repo', type=str, required=True, help='e.g. user/ncp-64m')
    p.add_argument('--private', action='store_true')
    p.add_argument('--commit_message', type=str, default='NCP-64M pretrained checkpoint')
    args = p.parse_args()

    from model.model_ncp import NCPConfig, NCPForCausalLM
    from huggingface_hub import HfApi

    cfg = NCPConfig()
    cfg.auto_map = {
        'AutoConfig': 'model_ncp.NCPConfig',
        'AutoModelForCausalLM': 'model_ncp.NCPForCausalLM',
    }
    model = NCPForCausalLM(cfg)
    model.load_state_dict(torch.load(args.ckpt, map_location='cpu'), strict=True)
    n = sum(p.numel() for p in model.parameters())
    print(f'loaded {args.ckpt}: {n / 1e6:.2f}M params')

    api = HfApi()
    api.create_repo(repo_id=args.repo, repo_type='model', exist_ok=True, private=args.private)
    with tempfile.TemporaryDirectory() as d:
        model.save_pretrained(d, safe_serialization=True)
        shutil.copy('model/model_ncp.py', os.path.join(d, 'model_ncp.py'))
        for f in ('tokenizer.json', 'tokenizer_config.json'):
            shutil.copy(os.path.join('model', f), os.path.join(d, f))
        api.upload_folder(folder_path=d, repo_id=args.repo,
                          commit_message=args.commit_message)
    print(f'uploaded -> https://huggingface.co/{args.repo}')


if __name__ == '__main__':
    main()
