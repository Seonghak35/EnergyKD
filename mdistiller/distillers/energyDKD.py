import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller

import pdb

def dkd_loss(logits_student, logits_teacher, target, alpha, beta, kd_temp):
    
    batch_num, class_num = logits_student.shape
    T_mat = kd_temp.unsqueeze(1).repeat(1, class_num)

    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / T_mat, dim=1)
    pred_teacher = F.softmax(logits_teacher / T_mat, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    
    tckd_loss = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1)
    tckd_loss = (tckd_loss*kd_temp*kd_temp).mean()

    pred_teacher_part2 = F.softmax(
        logits_teacher / T_mat - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / T_mat - 1000.0 * gt_mask, dim=1
    )

    nckd_loss = F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction="none").sum(1)
    nckd_loss = (nckd_loss*kd_temp*kd_temp).mean()

    return alpha * tckd_loss + beta * nckd_loss


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt


class energyDKD(Distiller):
    """Maximizing discrimination capability of knowledge distillation with energy function 
    (Knowledge-Based Systems 2024)"""

    def __init__(self, student, teacher, cfg, args, energy_gather):
        super(energyDKD, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.energyDKD.CE_WEIGHT
        self.alpha = cfg.energyDKD.ALPHA
        self.beta = cfg.energyDKD.BETA
        self.warmup = cfg.energyDKD.WARMUP

        self.high_T = cfg.energyDKD.T.HIGH
        self.mid_T = cfg.energyDKD.T.MID
        self.low_T = cfg.energyDKD.T.LOW

        self.energy_T = 4.0
        self.low_standard = energy_gather[0]
        self.high_standard = energy_gather[1]

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)

        energy_teacher = -self.energy_T*torch.log(torch.sum(torch.exp(logits_teacher/self.energy_T), dim=1))
        mask4low = torch.lt(energy_teacher, self.low_standard)
        mask4high = torch.gt(energy_teacher, self.high_standard)

        low_energy_T = mask4low*self.low_T
        high_energy_T = mask4high*self.high_T

        kd_temp = low_energy_T + high_energy_T
        kd_temp = torch.where(kd_temp == 0, torch.tensor(self.mid_T).cuda(), kd_temp)

        loss_dkd = min(kwargs["epoch"] / self.warmup, 1.0) * dkd_loss(
            logits_student,
            logits_teacher,
            target,
            self.alpha,
            self.beta,
            kd_temp,
        )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_dkd,
        }
        return logits_student, losses_dict
