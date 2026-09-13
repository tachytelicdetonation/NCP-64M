import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import torch
from contextlib import nullcontext
from torch.utils.data import DataLoader, Subset
from model.model_ncp import NCPConfig
from dataset.lm_dataset import PretrainDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, setup_seed, init_model, build_optimizers
from trainer.metrics import WandbMonitor


def train_epoch(epoch, loader, iters, optimizers, scaler, autocast_ctx, start_step=0, monitor=None):
    start_time = time.time()
    last_step = start_step
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        last_step = step
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate, args.warmup_ratio)
        for opt in optimizers:
            for param_group in opt.param_groups:
                param_group['lr'] = lr

        if monitor is not None:
            monitor.note_micro(input_ids, epoch * iters + step)

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
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if monitor is not None:
                monitor.on_boundary(raw_model, grad_norm.item())
            for opt in optimizers:
                if scaler is not None:
                    scaler.step(opt)
                else:
                    opt.step()
            if scaler is not None: scaler.update()
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)
            if monitor is not None:
                monitor.on_opt_step(input_ids, labels, lr)

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
            if monitor is not None:
                monitor.log_scalars(res, current_lr)

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            raw_model.eval()
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}.pth'
            state_dict = {k: v.half().cpu() for k, v in raw_model.state_dict().items()}
            torch.save(state_dict, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=raw_model, optimizers=optimizers,
                          epoch=epoch, step=step, save_dir=args.ckpt_dir,
                          wandb_run_id=monitor.run.id if monitor is not None else None)
            raw_model.train()
            del state_dict

        last_ids, last_labels = input_ids, labels   # tail-accumulation flush uses these
        del input_ids, labels, res, loss

        if args.max_steps > 0 and step >= args.max_steps:
            return last_step

    if last_step > start_step and last_step % args.accumulation_steps != 0:
        for opt in optimizers:
            if scaler is not None: scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if monitor is not None:
            monitor.on_boundary(raw_model, grad_norm.item())
        for opt in optimizers:
            (scaler.step(opt) if scaler is not None else opt.step())
        if scaler is not None: scaler.update()
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        if monitor is not None:
            monitor.on_opt_step(last_ids, last_labels, optimizers[-1].param_groups[-1]['lr'])
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
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.02,
                        help="fraction of total steps spent in linear lr warmup")
    parser.add_argument("--optimizer", type=str, default="muon", choices=["adamw", "muon"])
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
    parser.add_argument('--train_vq_only', default=0, type=int, choices=[0, 1],
                        help="Sec. 5.1: freeze the backbone and train only VQ codebooks + concept-prediction heads (use with --from_weight)")
    # wandb monitoring (see trainer/metrics.py for the metric catalog)
    parser.add_argument('--use_wandb', default=0, type=int, choices=[0, 1])
    parser.add_argument('--wandb_project', default='ncp-pretrain', type=str)
    parser.add_argument('--wandb_run_name', default='', type=str)
    parser.add_argument('--wandb_mode', default='online', choices=['online', 'offline', 'disabled'])
    parser.add_argument('--wandb_watch', default=1, type=int, choices=[0, 1],
                        help="wandb.watch per-layer grad/param histograms")
    parser.add_argument('--watch_freq', default=500, type=int)
    parser.add_argument('--diag_interval', default=250, type=int,
                        help="optimizer steps between diagnostics (update ratios, attn entropy, GNS)")
    parser.add_argument('--eval_interval', default=500, type=int,
                        help="optimizer steps between val-split evals")
    parser.add_argument('--eval_batches', default=16, type=int,
                        help="batches in the held-out val slice (0 disables the holdout)")
    parser.add_argument('--showcase_interval', default=1000, type=int,
                        help="optimizer steps between showcase tables/images (generations, codebook, routing)")
    parser.add_argument('--gen_tokens', default=64, type=int)
    parser.add_argument('--gen_prompts', default=4, type=int)
    parser.add_argument('--peak_flops', default=0.0, type=float,
                        help="device peak FLOPS for MFU logging (0 = skip)")
    parser.add_argument('--compile', default=1, type=int, choices=[0, 1],
                        help="torch.compile the model on CUDA")
    parser.add_argument('--compile_mode', default='default', type=str,
                        choices=['default', 'max-autotune', 'max-autotune-no-cudagraphs',
                                 'reduce-overhead'])
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
    if device_type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.cuda.memory._set_allocator_settings("expandable_segments:True")
        except Exception:
            pass

    # ========== 4. model, data, optimizers ==========
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device, save_dir=args.save_dir)
    raw_model = model
    if args.compile and device_type == "cuda":
        try:
            import torch._inductor.config as _inductor_cfg
            _inductor_cfg.triton.cudagraphs = False  # pools pin VRAM and starve diag/eval probes
        except Exception:
            pass
        model = torch.compile(model, mode=args.compile_mode)
        Logger(f'torch.compile enabled (mode={args.compile_mode})')
    # compiled graphs don't fire module forward hooks; monitor + checkpoints use
    # raw_model (same parameter objects, so optimizer/grad views are unaffected)
    if args.train_vq_only:
        assert args.arch == 'ncp', '--train_vq_only requires --arch ncp'
        for name, p in model.named_parameters():
            p.requires_grad = 'quantizer' in name
        Logger(f'VQ-only adaptation: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M '
               'trainable (codebooks + prediction heads)')
    train_ds = PretrainDataset(args.data_path, seq_len=args.max_seq_len)
    eval_idx = []
    if args.use_wandb and args.eval_batches > 0:
        eval_windows = args.eval_batches * args.batch_size
        if train_ds.n_samples > eval_windows + args.batch_size:
            eval_idx = list(range(train_ds.n_samples - eval_windows, train_ds.n_samples))
            train_ds = Subset(train_ds, range(train_ds.n_samples - eval_windows))
        else:
            Logger(f'Skipping val holdout: dataset too small ({train_ds.n_samples} windows)')
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=(device_type == "cuda"),
                        persistent_workers=(args.num_workers > 0), drop_last=True)
    optimizers = build_optimizers(model, args)

    # ========== 5. resume ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        raw_model.load_state_dict(ckp_data['model'])
        if ckp_data.get('optimizers'):
            for opt, sd in zip(optimizers, ckp_data['optimizers']):
                opt.load_state_dict(sd)
        start_epoch, start_step = ckp_data['epoch'], ckp_data.get('step', 0)
        Logger(f'Resumed from epoch {start_epoch} step {start_step}')

    # ========== 6. wandb monitor ==========
    monitor = None
    if args.use_wandb and is_main_process():
        monitor = WandbMonitor(
            raw_model, tokenizer, optimizers, args, lm_config, autocast_ctx, scaler,
            val_ds=train_ds.dataset if isinstance(train_ds, Subset) else train_ds,
            eval_idx=eval_idx,
            run_id=ckp_data.get('wandb_run_id') if ckp_data else None)

    # ========== 7. train ==========
    iters = len(loader)
    if monitor is not None:
        monitor.micro_step = start_epoch * iters + start_step
        monitor.opt_step = monitor.micro_step // args.accumulation_steps
        monitor.tokens_seen = monitor.opt_step * args.batch_size * args.max_seq_len * args.accumulation_steps
        monitor.updates.mark(monitor.opt_step)
    try:
        for epoch in range(start_epoch, args.epochs):
            last = train_epoch(epoch, loader, iters, optimizers, scaler, autocast_ctx,
                               start_step if epoch == start_epoch else 0, monitor=monitor)
            if args.max_steps > 0 and last >= args.max_steps:
                break
        Logger('Training done.')
    finally:
        if monitor is not None:
            monitor.finish()
