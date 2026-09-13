import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import torch
from contextlib import nullcontext
from torch.utils.data import DataLoader
from model.model_ncp import NCPConfig
from dataset.lm_dataset import PretrainDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, setup_seed, init_model, build_optimizers


def train_epoch(epoch, loader, iters, optimizers, scaler, autocast_ctx, start_step=0):
    start_time = time.time()
    last_step = start_step
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        last_step = step
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for opt in optimizers:
            for param_group in opt.param_groups:
                param_group['lr'] = lr

        with autocast_ctx:
            res = model(input_ids, labels=labels)
            loss = res['loss'] / args.accumulation_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if step % args.accumulation_steps == 0:
            for opt in optimizers:
                if scaler is not None: scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            for opt in optimizers:
                if scaler is not None:
                    scaler.step(opt)
                else:
                    opt.step()
            if scaler is not None: scaler.update()
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_lr = optimizers[-1].param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            msg = (f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                   f'loss: {res["loss"].item():.4f}, ntp: {res["loss_ntp"].item():.4f}, '
                   f'lr: {current_lr:.8f}, eta: {eta_min:.1f}min')
            if res['loss_ncp'] is not None:
                msg += f', ncp: {res["loss_ncp"].item():.4f}, vq: {res["loss_vq"].item():.4f}'
            Logger(msg)

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}.pth'
            state_dict = {k: v.half().cpu() for k, v in model.state_dict().items()}
            torch.save(state_dict, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizers=optimizers,
                          epoch=epoch, step=step, save_dir=args.ckpt_dir)
            model.train()
            del state_dict

        del input_ids, labels, res, loss

        if args.max_steps > 0 and step >= args.max_steps:
            return last_step

    if last_step > start_step and last_step % args.accumulation_steps != 0:
        for opt in optimizers:
            if scaler is not None: scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        for opt in optimizers:
            (scaler.step(opt) if scaler is not None else opt.step())
        if scaler is not None: scaler.update()
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
    return last_step


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NCP Pretraining")
    parser.add_argument("--save_dir", type=str, default="./out")
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints")
    parser.add_argument('--save_weight', default='pretrain', type=str)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=0, help=">0 caps total optimizer-visible steps per epoch")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "muon"])
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--save_interval", type=int, default=1000)
    # architecture
    parser.add_argument('--arch', default='ncp', choices=['ncp', 'vanilla'], help="vanilla = parameter-matched plain transformer baseline")
    parser.add_argument('--hidden_size', default=768, type=int)
    parser.add_argument('--n_enc_layers', default=3, type=int)
    parser.add_argument('--n_concept_layers', default=2, type=int)
    parser.add_argument('--n_dec_layers', default=3, type=int)
    parser.add_argument('--concept_chunk', default=4, type=int)
    parser.add_argument('--pq_segments', default=6, type=int)
    parser.add_argument('--pq_codewords', default=128, type=int)
    parser.add_argument('--ncp_loss_weight', default=1.0, type=float)
    parser.add_argument('--vq_loss_weight', default=1.0, type=float)
    parser.add_argument('--use_irc', default=1, type=int, choices=[0, 1])
    parser.add_argument('--use_crc', default=1, type=int, choices=[0, 1])
    parser.add_argument('--max_seq_len', default=512, type=int, help="must be a multiple of concept_chunk")
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument("--data_path", type=str, default="./dataset/pretrain.bin")
    parser.add_argument('--from_weight', default='none', type=str)
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1])
    args = parser.parse_args()

    # ========== 1. seed ==========
    setup_seed(args.seed)

    # ========== 2. config ==========
    os.makedirs(args.save_dir, exist_ok=True)
    assert args.max_seq_len % args.concept_chunk == 0, "max_seq_len must be divisible by concept_chunk"
    lm_config = NCPConfig(
        hidden_size=args.hidden_size, n_enc_layers=args.n_enc_layers,
        n_concept_layers=args.n_concept_layers, n_dec_layers=args.n_dec_layers,
        arch=args.arch, concept_chunk=args.concept_chunk,
        pq_segments=args.pq_segments, pq_codewords=args.pq_codewords,
        ncp_loss_weight=args.ncp_loss_weight, vq_loss_weight=args.vq_loss_weight,
        use_irc=bool(args.use_irc),  # for a pure vanilla baseline pass --use_irc 0
        use_crc=bool(args.use_crc) if args.arch == 'ncp' else False,
        max_position_embeddings=args.max_seq_len,
    )
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir=args.ckpt_dir) if args.from_resume == 1 else None

    # ========== 3. mixed precision ==========
    device_type = "cuda" if "cuda" in args.device else ("mps" if "mps" in args.device else "cpu")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = torch.autocast(device_type=device_type, dtype=dtype) if device_type in ("cuda", "mps") else nullcontext()
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16')) if device_type == "cuda" else None

    # ========== 4. model, data, optimizers ==========
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device, save_dir=args.save_dir)
    train_ds = PretrainDataset(args.data_path, seq_len=args.max_seq_len)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=False, drop_last=True)
    optimizers = build_optimizers(model, args)

    # ========== 5. resume ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        if ckp_data.get('optimizers'):
            for opt, sd in zip(optimizers, ckp_data['optimizers']):
                opt.load_state_dict(sd)
        start_epoch, start_step = ckp_data['epoch'], ckp_data.get('step', 0)
        Logger(f'Resumed from epoch {start_epoch} step {start_step}')

    # ========== 6. train ==========
    iters = len(loader)
    for epoch in range(start_epoch, args.epochs):
        last = train_epoch(epoch, loader, iters, optimizers, scaler, autocast_ctx,
                           start_step if epoch == start_epoch else 0)
        if args.max_steps > 0 and last >= args.max_steps:
            break
    Logger('Training done.')
