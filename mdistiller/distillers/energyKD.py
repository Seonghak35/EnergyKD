import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller

import pdb

def kd_loss(logits_student, logits_teacher, temperature):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd

def kd_t_loss(logits_student, logits_teacher, temperature):
    batch_num, class_num = logits_student.shape
    T_mat = temperature.unsqueeze(1).repeat(1, class_num)
    log_pred_student = F.log_softmax(logits_student / T_mat, dim=1)
    pred_teacher = F.softmax(logits_teacher / T_mat, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1)
    loss_kd = (loss_kd*temperature*temperature).mean()
    return loss_kd

class energyKD(Distiller):
    """Maximizing discrimination capability of knowledge distillation with energy function 
    (Knowledge-Based Systems 2024)"""

    def __init__(self, student, teacher, cfg, args, energy_gather):
        super(energyKD, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.energyKD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.energyKD.LOSS.KD_WEIGHT

        self.high_T = cfg.energyKD.T.HIGH
        self.mid_T = cfg.energyKD.T.MID
        self.low_T = cfg.energyKD.T.LOW

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

        loss_kd = self.kd_loss_weight * kd_t_loss(
                logits_student, logits_teacher, kd_temp
                )

        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict
