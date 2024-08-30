import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ._base import Distiller
import pdb

def kd_loss(logits_student, logits_teacher, temperature):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd

class energyAUG(Distiller):
    """Maximizing discrimination capability of knowledge distillation with energy function 
    (Knowledge-Based Systems 2024)"""

    def __init__(self, student, teacher, cfg, args, energy_gather):
        super(energyAUG, self).__init__(student, teacher)
        self.temperature = cfg.energyAUG.TEMPERATURE
        self.ce_loss_weight = cfg.energyAUG.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.energyAUG.LOSS.KD_WEIGHT
        self.aug = cfg.energyAUG.AUGMENTATION

        self.energy_T = 4.0
        self.low_standard = energy_gather[0]
        self.high_standard = energy_gather[1]

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_kd = self.kd_loss_weight * kd_loss(
                logits_student, logits_teacher, self.temperature
        )

        ### revised by hakk ###

        # augmentation data
        energy_teacher = -self.energy_T*torch.log(torch.sum(torch.exp(logits_teacher/self.energy_T), dim=1))
        mask4low = torch.lt(energy_teacher, self.low_standard)
        mask4high = torch.gt(energy_teacher, self.high_standard)
        pdb.set_trace()

        aug4target_high = torch.mul(target, mask4high.cuda())
        mask4high = mask4high.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        mask4high = mask4high.repeat(1,3,32,32)
        aug4img_high = torch.mul(image, mask4high.cuda())
 
        aug4target_low = torch.mul(target, mask4low.cuda())
        mask4low = mask4low.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        mask4low = mask4low.repeat(1,3,32,32)
        aug4img_low = torch.mul(image, mask4low.cuda())
        
        aug4target = aug4target_high + aug4target_low
        aug4img = aug4img_high + aug4img_low

        img0 = image[0,:,:,:]
        img1 = image[1,:,:,:]

        tar0 = target[0]
        tar1 = target[1]

        new_img = torch.stack([img0, img1], dim=0)
        new_target = torch.stack([tar0, tar1], dim=0)
        
        batch_num, cls_num = logits_teacher.shape
        for i in range(len(target)):
            if (aug4img[i,:,:,:].sum() != 0) and (aug4target[i] < cls_num):
                add_image = aug4img[i,:,:,:].unsqueeze(0)
                add_target = aug4target[i].unsqueeze(0)

                new_img = torch.cat([new_img, add_image], 0)
                new_target = torch.cat([new_target, add_target], 0)
        
        if self.aug == 'MIXUP':
            mixed_image, original_target, mixing_target, lam = mixup_data(new_img, new_target)
        elif self.aug == 'CUTMIX':
            mixed_image, original_target, mixing_target, lam = cutmix_data(new_img, new_target)
        
        aug_logits_student, _ = self.student(mixed_image)
        with torch.no_grad():
            aug_logits_teacher, _ = self.teacher(mixed_image)
        
        loss_aug = self.kd_loss_weight * kd_loss(
                aug_logits_student, aug_logits_teacher, self.temperature
                )

        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
            "loss_aug": loss_aug,
        }
        return logits_student, losses_dict

def mixup_data(image, target, aug_alpha=0.2):
    lam = np.random.beta(aug_alpha, aug_alpha)
    index = torch.randperm(image.size(0)).cuda()
    mixed_image = lam*image + (1.0 - lam)*image[index]
    original_target, mixing_target = target, target[index]
    return mixed_image, original_target, mixing_target, lam

def cutmix_data(image, target, aug_alpha=1.0):
    image = image.clone()
    def rand_bbox(size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = np.int(W*cut_rat)
        cut_h = np.int(H*cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        return bbx1, bby1, bbx2, bby2
    lam = np.random.beta(aug_alpha, aug_alpha)
    index = torch.randperm(image.size(0)).cuda()
    original_target, mixing_target = target, target[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(image.size(), lam)
    image[:, :, bbx1:bbx2, bby1:bby2] = image[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1)*(bby2 - bby1))/(image.size()[-1]*image.size()[-2])
    return image, original_target, mixing_target, lam


