#!/usr/bin/env python3
"""
Finetune WiLoR on POV-Surgery dataset.

Usage:
    python train.py --data-dir ../pov_surgery_data --output-dir ./output

    # With custom config
    python train.py --data-dir ../pov_surgery_data --config configs/finetune_pov_surgery.yaml

    # Resume from checkpoint
    python train.py --data-dir ../pov_surgery_data --resume-from ./output/checkpoints/last.ckpt
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

# Add WiLoR to path
WILOR_ROOT = Path(__file__).resolve().parents[2] / "WiLoR"
sys.path.insert(0, str(WILOR_ROOT))

from wilor.configs import get_config

# Add finetuning package
sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.wilor_finetune import WiLoRFinetune
from datasets.pov_surgery_dataset import POVSurgeryDataset


def find_wilor_checkpoint():
    """Find the pretrained wilor_final.ckpt from wilor_mini package."""
    try:
        import wilor_mini
        ckpt = Path(wilor_mini.__file__).parent / "pretrained_models" / "wilor_final.ckpt"
        if ckpt.exists():
            return str(ckpt)
    except ImportError:
        pass

    # Fallback: check WiLoR directory
    ckpt = WILOR_ROOT / "pretrained_models" / "wilor_final.ckpt"
    if ckpt.exists():
        return str(ckpt)

    return None


def find_mano_files():
    """Find MANO model files needed by WiLoR."""
    try:
        import wilor_mini
        mano_dir = Path(wilor_mini.__file__).parent / "pretrained_models"
        model_path = mano_dir / "MANO_RIGHT.pkl"
        mean_params = mano_dir / "mano_mean_params.npz"
        if model_path.exists() and mean_params.exists():
            return str(mano_dir), str(model_path), str(mean_params)
    except ImportError:
        pass

    # Fallback: WiLoR mano_data
    mano_dir = WILOR_ROOT / "mano_data"
    return str(mano_dir), str(mano_dir / "MANO_RIGHT.pkl"), str(mano_dir / "mano_mean_params.npz")


def setup_config(args):
    """Load and configure the training config."""
    cfg = get_config(str(args.config), merge=True)

    # Update MANO paths
    mano_dir, model_path, mean_params = find_mano_files()
    cfg.defrost()
    cfg.MANO.DATA_DIR = mano_dir
    cfg.MANO.MODEL_PATH = model_path
    cfg.MANO.MEAN_PARAMS = mean_params

    # Remove pretrained backbone weights (loaded from checkpoint)
    if 'PRETRAINED_WEIGHTS' in cfg.MODEL.BACKBONE:
        cfg.MODEL.BACKBONE.pop('PRETRAINED_WEIGHTS')

    cfg.freeze()
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Finetune WiLoR on POV-Surgery")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to pov_surgery_data/ (parent of demo_data/)")
    parser.add_argument("--config", type=str,
                        default=str(Path(__file__).parent / "configs" / "finetune_pov_surgery.yaml"),
                        help="Training config YAML")
    parser.add_argument("--output-dir", type=str, default="./output",
                        help="Output directory for checkpoints and logs")
    parser.add_argument("--ckpt-path", type=str, default=None,
                        help="Pretrained WiLoR checkpoint (auto-detected if not specified)")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Resume training from this checkpoint")
    parser.add_argument("--devices", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--precision", type=str, default="16-mixed",
                        choices=["32", "16-mixed", "bf16-mixed"])
    args = parser.parse_args()

    # Paths
    data_dir = Path(args.data_dir).resolve()
    data_root = data_dir / "demo_data" / "POV_Surgery_data"
    mano_model_dir = data_dir / "data" / "bodymodel"
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    assert data_root.exists(), f"Data root not found: {data_root}"
    assert mano_model_dir.exists(), f"MANO model dir not found: {mano_model_dir}"

    # Config
    cfg = setup_config(args)
    print(f"Config loaded: LR={cfg.TRAIN.LR}, BS={cfg.TRAIN.BATCH_SIZE}, steps={cfg.GENERAL.TOTAL_STEPS}")

    # Model
    print("Creating WiLoRFinetune model...")
    model = WiLoRFinetune(cfg, init_renderer=False)

    # Load pretrained weights
    ckpt_path = args.ckpt_path or find_wilor_checkpoint()
    if ckpt_path and Path(ckpt_path).exists():
        print(f"Loading pretrained weights from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        missing, unexpected = model.load_state_dict(ckpt['state_dict'], strict=False)
        print(f"  Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
        if missing:
            print(f"  Missing (first 5): {missing[:5]}")
    else:
        print("WARNING: No pretrained checkpoint found! Training from scratch.")

    # Datasets
    aug_config = {
        "SCALE_FACTOR": cfg.DATASETS.CONFIG.SCALE_FACTOR,
        "ROT_FACTOR": cfg.DATASETS.CONFIG.ROT_FACTOR,
        "ROT_AUG_RATE": cfg.DATASETS.CONFIG.ROT_AUG_RATE,
        "TRANS_FACTOR": cfg.DATASETS.CONFIG.TRANS_FACTOR,
        "COLOR_SCALE": cfg.DATASETS.CONFIG.COLOR_SCALE,
    }

    print("\nCreating datasets...")
    train_dataset = POVSurgeryDataset(
        data_root=str(data_root),
        mano_model_dir=str(mano_model_dir),
        split="train",
        img_size=cfg.MODEL.IMAGE_SIZE,
        augment=True,
        aug_config=aug_config,
    )
    val_dataset = POVSurgeryDataset(
        data_root=str(data_root),
        mano_model_dir=str(mano_model_dir),
        split="test",
        img_size=cfg.MODEL.IMAGE_SIZE,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.GENERAL.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=cfg.GENERAL.NUM_WORKERS > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.GENERAL.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=cfg.GENERAL.NUM_WORKERS > 0,
    )

    print(f"Train: {len(train_dataset)} samples, {len(train_loader)} batches/epoch")
    print(f"Val:   {len(val_dataset)} samples, {len(val_loader)} batches/epoch")

    # Callbacks
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="wilor_ft_{step:06d}_{val/loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=cfg.GENERAL.get("CHECKPOINT_SAVE_TOP_K", 3),
        every_n_train_steps=cfg.GENERAL.CHECKPOINT_STEPS,
        save_last=True,
    )

    callbacks = [checkpoint_cb]

    # Logger
    logger = TensorBoardLogger(
        save_dir=str(output_dir),
        name="wilor_pov_surgery_ft",
    )

    # Trainer
    trainer = pl.Trainer(
        max_steps=cfg.GENERAL.TOTAL_STEPS,
        val_check_interval=cfg.GENERAL.VAL_STEPS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.devices,
        precision=args.precision,
        logger=logger,
        callbacks=callbacks,
        # gradient clipping is handled manually in training_step (manual optimization)
        log_every_n_steps=cfg.GENERAL.LOG_STEPS,
        num_sanity_val_steps=2,
    )

    print(f"\nStarting training for {cfg.GENERAL.TOTAL_STEPS} steps...")
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=args.resume_from,
    )

    print(f"\nTraining complete! Checkpoints saved to {output_dir / 'checkpoints'}")


if __name__ == "__main__":
    main()
