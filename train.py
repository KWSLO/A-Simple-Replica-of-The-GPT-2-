# train_gpt.py
import math
import os
import sys
import time

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from GPT.GPT2.GPT2.model import GPTConfig, MyDataset, GPT

sys.path.append('model.py')


# -------------- 训练配置 --------------
class TrainArgs:
    data_path = "data/sample.jsonl"  # 小数据文件路径（jsonl，每行含 "text" 字段）
    out_dir = "./checkpoints"
    epochs = 3
    batch_size = 4  # 如果显存小，调小
    learning_rate = 5e-4
    weight_decay = 0.01
    betas = (0.9, 0.95)
    max_grad_norm = 1.0
    warmup_ratio = 0.03  # warmup 占总 steps 比例
    gradient_accumulation_steps = 1  # 如果显存小，可用 >1 来累积
    save_every_steps = 200  # 每 N step 保存一次 checkpoint
    log_every_steps = 20
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = 2
    seed = 1024
    model_cfg = dict(block_size=128, batch_size=8, n_layer=4, n_head=4, n_embd=256, dropout=0.1, vocab_size=50274)


args = TrainArgs()


# -------------- 工具函数：保存 / 加载 checkpoint --------------
def save_checkpoint(model, optimizer, scheduler, scaler, step, epoch, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "step": step,
        "epoch": epoch,
    }
    path = os.path.join(out_dir, f"ckpt_step{step}.pt")
    torch.save(ckpt, path)
    print(f"[checkpoint] saved {path}")


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, device=args.device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler is not None and ckpt.get("scaler_state_dict") is not None:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    step = ckpt.get("step", 0)
    epoch = ckpt.get("epoch", 0)
    print(f"[checkpoint] loaded {path} (step={step}, epoch={epoch})")
    return step, epoch


# -------------- 学习率 schedule helper --------------
def build_lr_scheduler(optimizer, total_steps, warmup_steps):
    """
    Linear warmup -> linear decay to 0.
    lr = base_lr * min(step / warmup_steps, max(0, (total_steps - step) / (total_steps - warmup_steps)))
    """

    def lr_lambda(current_step):
        if current_step < warmup_steps and warmup_steps > 0:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            denom = max(1, total_steps - warmup_steps)
            return max(0.0, float(total_steps - current_step) / denom)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler


# -------------- 主训练函数 --------------
def train():
    # reproducibility
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)

    # model & dataset
    cfg = GPTConfig(**args.model_cfg)
    model = GPT(cfg)
    model.to(args.device)

    # dataset
    dataset = MyDataset(args.data_path, block_size=cfg.block_size, max_lines=1000)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                            pin_memory=True)

    # optimizer / scheduler / scaler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, betas=args.betas, weight_decay=args.weight_decay)
    # total steps estimate
    steps_per_epoch = math.ceil(len(dataloader) / max(1, args.gradient_accumulation_steps))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = build_lr_scheduler(optimizer, total_steps, warmup_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.device.startswith("cuda")))

    # optionally resume checkpoint (uncomment to use)
    # last_ckpt = "./checkpoints/ckpt_stepXXXX.pt"
    # step0, start_epoch = load_checkpoint(last_ckpt, model, optimizer, scheduler, scaler)
    step = 0
    model.train()

    # training loop
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        running_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, (x, y) in enumerate(dataloader):
            x = x.to(args.device, non_blocking=True)
            y = y.to(args.device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(args.device.startswith("cuda"))):
                logits, loss = model(x, targets=y)
                # forward returns (logits, loss) if targets provided
                loss = loss / args.gradient_accumulation_steps

            # backward with scaler
            scaler.scale(loss).backward()

            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                # unscale before clipping
                scaler.unscale_(optimizer)
                # gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                step += 1

                running_loss += loss.item() * args.gradient_accumulation_steps  # undo division

                # logging
                if step % args.log_every_steps == 0:
                    avg_loss = running_loss / args.log_every_steps
                    lr = scheduler.get_last_lr()[0]
                    print(f"[epoch {epoch}] step {step}/{total_steps} lr={lr:.3e} loss={avg_loss:.4f}")
                    running_loss = 0.0

                # checkpoint
                if step % args.save_every_steps == 0:
                    save_checkpoint(model, optimizer, scheduler, scaler, step, epoch, args.out_dir)

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch} finished in {epoch_time:.1f}s")

        # validation pass (optional): compute avg loss over small subset
        model.eval()
        val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            # iterate a few batches for quick validation
            for vi, (vx, vy) in enumerate(dataloader):
                if vi >= 10:
                    break
                vx = vx.to(args.device)
                vy = vy.to(args.device)
                with torch.cuda.amp.autocast(enabled=(args.device.startswith("cuda"))):
                    _, vl = model(vx, targets=vy)
                    val_loss += vl.item()
                    val_steps += 1
        model.train()
        if val_steps > 0:
            print(f"Validation loss (sampled) after epoch {epoch}: {val_loss / val_steps:.4f}")

    # end training: save final checkpoint
    save_checkpoint(model, optimizer, scheduler, scaler, step, args.epochs, args.out_dir)
    print("Training complete.")


# -------------- 运行 --------------
if __name__ == "__main__":
    train()
