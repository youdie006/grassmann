import json
import random
import time
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import WikiText2Dataset
from model import LMConfig, Transformer


def select_device(device_name: str) -> str:
    if device_name != "auto":
        return device_name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(cfg: DictConfig):
    train_dataset = WikiText2Dataset(
        split="train",
        sequence_len=cfg.data.max_context_len,
        tokenizer_name=cfg.data.tokenizer_name,
        source=cfg.data.source,
        data_root=cfg.data.data_root,
        max_samples=cfg.data.max_samples_train,
    )
    val_dataset = WikiText2Dataset(
        split="validation",
        sequence_len=cfg.data.max_context_len,
        tokenizer_name=cfg.data.tokenizer_name,
        source=cfg.data.source,
        data_root=cfg.data.data_root,
        max_samples=cfg.data.max_samples_val,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
    )
    return train_loader, val_loader, train_dataset.vocab_size


def create_model(vocab_size: int, cfg: DictConfig, device: str):
    lm_config = LMConfig(
        vocab_size=vocab_size,
        max_context_len=cfg.data.max_context_len,
        num_layers=cfg.model.num_layers,
        tensor_lifting_strategy=cfg.model.tensor_lifting_strategy,
        lags=list(cfg.model.lags),
        d_model=cfg.model.d_model,
        num_heads=cfg.model.num_heads,
        expansion_ratio=cfg.model.expansion_ratio,
        dropout_rate=cfg.train.dropout_rate,
        weight_tying=cfg.model.weight_tying,
        d_low=cfg.model.d_low,
        pre_norm=cfg.model.pre_norm,
    )
    model = Transformer(lm_config).to(device)
    return model, lm_config


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(model, dataloader, optimizer, scaler, criterion, scheduler, cfg, device):
    model.train()
    total_loss = 0.0
    total_tokens = 0
    start = time.time()
    use_amp = bool(cfg.train.amp and device == "cuda")

    pbar = tqdm(dataloader, desc="train", leave=False)
    for inputs, targets in pbar:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(inputs)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        if use_amp:
            scaler.scale(loss).backward()
            if cfg.train.grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if cfg.train.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()

        scheduler.step()
        total_loss += loss.item()
        total_tokens += targets.numel()
        elapsed = max(time.time() - start, 1e-6)
        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            ppl=f"{np.exp(loss.item()):.2f}",
            tok_s=f"{int(total_tokens / elapsed)}",
        )
    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(model, dataloader, criterion, cfg, device):
    model.eval()
    total_loss = 0.0
    use_amp = bool(cfg.train.amp and device == "cuda")

    pbar = tqdm(dataloader, desc="val", leave=False)
    for inputs, targets in pbar:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(inputs)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}", ppl=f"{np.exp(loss.item()):.2f}")

    avg_loss = total_loss / len(dataloader)
    return avg_loss


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    set_seed(int(cfg.seed))
    device = select_device(cfg.train.device)
    print(f"device={device}")
    print(f"config=\n{OmegaConf.to_yaml(cfg)}")

    train_loader, val_loader, vocab_size = load_data(cfg)
    model, lm_config = create_model(vocab_size, cfg, device)
    num_params = count_parameters(model)

    optimizer = AdamW(
        model.parameters(),
        lr=float(cfg.train.learning_rate),
        weight_decay=float(cfg.train.weight_decay),
    )
    total_steps = len(train_loader) * int(cfg.train.num_epochs)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg.train.amp and device == "cuda"))

    best_val_loss = float("inf")
    best_val_ppl = float("inf")
    train_losses = []
    val_losses = []

    for epoch in range(1, int(cfg.train.num_epochs) + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, scaler, criterion, scheduler, cfg, device
        )
        val_loss = evaluate(model, val_loader, criterion, cfg, device)
        train_ppl = float(np.exp(train_loss))
        val_ppl = float(np.exp(val_loss))
        train_losses.append(float(train_loss))
        val_losses.append(float(val_loss))

        print(
            f"epoch {epoch:02d}/{int(cfg.train.num_epochs)} "
            f"train_loss={train_loss:.4f} train_ppl={train_ppl:.2f} "
            f"val_loss={val_loss:.4f} val_ppl={val_ppl:.2f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = float(val_loss)
            best_val_ppl = float(val_ppl)
            output_dir = Path(HydraConfig.get().runtime.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output_dir / "best.pt")

    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "run_dir": str(output_dir),
        "experiment_name": str(cfg.experiment.name),
        "strategy": str(cfg.model.tensor_lifting_strategy),
        "num_params": int(num_params),
        "best_val_loss": float(best_val_loss),
        "best_val_ppl": float(best_val_ppl),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "lm_config": vars(lm_config),
    }
    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(
        f"saved={output_dir/'results.json'} "
        f"strategy={results['strategy']} params={results['num_params']} "
        f"best_val_ppl={results['best_val_ppl']:.2f}"
    )


if __name__ == "__main__":
    main()
