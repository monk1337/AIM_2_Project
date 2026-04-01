"""
WiLoRFinetune: Subclass of WiLoR for finetuning on POV-Surgery.

Changes from original WiLoR:
1. Simplified training_step (no discriminator, no mocap batch)
2. configure_optimizers returns single optimizer (backbone + refine_net)
3. Added vertex loss following CPCI paper's recipe
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Tuple

# Add WiLoR to path
WILOR_ROOT = Path(__file__).resolve().parents[3] / "WiLoR"
sys.path.insert(0, str(WILOR_ROOT))

from wilor.models.wilor import WiLoR
from wilor.models.losses import Keypoint3DLoss
from wilor.utils.geometry import aa_to_rotmat


class WiLoRFinetune(WiLoR):
    """WiLoR model adapted for finetuning on POV-Surgery."""

    def __init__(self, cfg, init_renderer=False):
        super().__init__(cfg, init_renderer=init_renderer)

        # Additional vertex loss (L2, following CPCI paper)
        self.vertex_3d_loss = Keypoint3DLoss(loss_type='l1')

    def get_parameters(self):
        """Return all trainable parameters (backbone + refine_net)."""
        all_params = list(self.backbone.parameters()) + list(self.refine_net.parameters())
        return all_params

    def configure_optimizers(self):
        """Single AdamW optimizer, no discriminator."""
        param_groups = [{
            'params': filter(lambda p: p.requires_grad, self.get_parameters()),
            'lr': self.cfg.TRAIN.LR,
        }]
        optimizer = torch.optim.AdamW(
            params=param_groups,
            weight_decay=self.cfg.TRAIN.WEIGHT_DECAY,
        )
        return optimizer

    def compute_loss(self, batch: Dict, output: Dict, train: bool = True) -> torch.Tensor:
        """Compute loss with added vertex distance loss.

        Loss = WiLoR original losses (3D joints L1 + 2D L1 + MANO param MSE)
               + vertex distance loss (L1, root-centered)
        """
        pred_mano_params = output['pred_mano_params']
        pred_keypoints_2d = output['pred_keypoints_2d']
        pred_keypoints_3d = output['pred_keypoints_3d']
        pred_vertices = output['pred_vertices']

        batch_size = pred_mano_params['hand_pose'].shape[0]
        device = pred_mano_params['hand_pose'].device

        gt_keypoints_2d = batch['keypoints_2d']
        gt_keypoints_3d = batch['keypoints_3d']
        gt_mano_params = batch['mano_params']
        has_mano_params = batch['has_mano_params']
        is_axis_angle = batch['mano_params_is_axis_angle']

        # 2D keypoint loss
        loss_keypoints_2d = self.keypoint_2d_loss(pred_keypoints_2d, gt_keypoints_2d)

        # 3D keypoint loss (root-centered)
        loss_keypoints_3d = self.keypoint_3d_loss(pred_keypoints_3d, gt_keypoints_3d, pelvis_id=0)

        # MANO parameter loss
        loss_mano_params = {}
        for k, pred in pred_mano_params.items():
            gt = gt_mano_params[k].view(batch_size, -1).to(device)
            if is_axis_angle[k].all():
                gt = aa_to_rotmat(gt.reshape(-1, 3)).view(batch_size, -1, 3, 3)
            has_gt = has_mano_params[k].to(device)
            loss_mano_params[k] = self.mano_parameter_loss(
                pred.reshape(batch_size, -1), gt.reshape(batch_size, -1), has_gt
            )

        # Vertex loss (root-centered using wrist joint, same as 3D keypoint loss)
        # pred_vertices: (B, 778, 3), pred_keypoints_3d: (B, 21, 3)
        # Root-center vertices using the wrist joint (joint 0)
        gt_verts = batch.get('gt_vertices')
        loss_vertices = torch.tensor(0.0, device=device)
        if gt_verts is not None:
            pred_verts_centered = pred_vertices - pred_keypoints_3d[:, 0:1, :]
            gt_verts_centered = gt_verts - gt_keypoints_3d[:, 0:1, :3]
            loss_vertices = nn.functional.l1_loss(pred_verts_centered, gt_verts_centered, reduction='sum')

        # Total loss
        loss = (
            self.cfg.LOSS_WEIGHTS['KEYPOINTS_3D'] * loss_keypoints_3d
            + self.cfg.LOSS_WEIGHTS['KEYPOINTS_2D'] * loss_keypoints_2d
            + sum(loss_mano_params[k] * self.cfg.LOSS_WEIGHTS[k.upper()] for k in loss_mano_params)
            + self.cfg.LOSS_WEIGHTS.get('VERTICES_3D', 0.0) * loss_vertices
        )

        losses = dict(
            loss=loss.detach(),
            loss_keypoints_2d=loss_keypoints_2d.detach(),
            loss_keypoints_3d=loss_keypoints_3d.detach(),
            loss_vertices=loss_vertices.detach(),
        )
        for k, v in loss_mano_params.items():
            losses['loss_' + k] = v.detach()

        output['losses'] = losses
        return loss

    def training_step(self, batch: Dict, batch_idx: int) -> Dict:
        """Simplified training step: no discriminator, no mocap batch."""
        optimizer = self.optimizers(use_pl_optimizer=True)

        output = self.forward_step(batch, train=True)
        loss = self.compute_loss(batch, output, train=True)

        if torch.isnan(loss):
            raise ValueError('Loss is NaN')

        optimizer.zero_grad()
        self.manual_backward(loss)

        if self.cfg.TRAIN.get('GRAD_CLIP_VAL', 0) > 0:
            gn = torch.nn.utils.clip_grad_norm_(
                self.get_parameters(), self.cfg.TRAIN.GRAD_CLIP_VAL, error_if_nonfinite=False
            )
            self.log('train/grad_norm', gn, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        optimizer.step()

        if self.global_step > 0 and self.global_step % self.cfg.GENERAL.LOG_STEPS == 0:
            if self.mesh_renderer is not None:
                self.tensorboard_logging(batch, output, self.global_step, train=True)

        self.log('train/loss', output['losses']['loss'], on_step=True, on_epoch=True, prog_bar=True, logger=False)
        self.log('train/loss_kp3d', output['losses']['loss_keypoints_3d'], on_step=True, on_epoch=False, prog_bar=False, logger=True)
        self.log('train/loss_kp2d', output['losses']['loss_keypoints_2d'], on_step=True, on_epoch=False, prog_bar=False, logger=True)

        return output

    def validation_step(self, batch: Dict, batch_idx: int, dataloader_idx=0) -> Dict:
        """Validation step with loss computation."""
        output = self.forward_step(batch, train=False)
        loss = self.compute_loss(batch, output, train=False)
        output['loss'] = loss

        self.log('val/loss', output['losses']['loss'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val/loss_keypoints_3d', output['losses']['loss_keypoints_3d'], on_step=False, on_epoch=True)
        self.log('val/loss_keypoints_2d', output['losses']['loss_keypoints_2d'], on_step=False, on_epoch=True)

        if self.global_step > 0 and self.mesh_renderer is not None:
            self.tensorboard_logging(batch, output, self.global_step, train=False)

        return output
