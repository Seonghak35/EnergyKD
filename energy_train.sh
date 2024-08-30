
### Energy KD ###

CUDA_VISIBLE_DEVICES=0 python tools/train.py --cfg configs/cifar100/energy_kd/wrn40_2_wrn16_2.yaml --energy

CUDA_VISIBLE_DEVICES=0 python tools/train.py --cfg configs/cifar100/energy_kd/res56_res20.yaml --energy

CUDA_VISIBLE_DEVICES=0 python tools/train.py --cfg configs/cifar100/energy_kd/vgg13_vgg8.yaml --energy

CUDA_VISIBLE_DEVICES=0 python tools/train.py --cfg configs/cifar100/energy_kd/res32x4_shuv2.yaml --energy

CUDA_VISIBLE_DEVICES=0 python tools/train.py --cfg configs/cifar100/energy_kd/vgg13_mv2.yaml --energy


### Energy DKD ###

CUDA_VISIBLE_DEVICES=0 python tools/train.py --cfg configs/cifar100/energy_dkd/vgg13_vgg8.yaml --energy


### HE-DA ###

CUDA_VISIBLE_DEVICES=0 python tools/train.py --cfg configs/cifar100/energy_aug/wrn40_2_wrn16_2.yaml --energy

CUDA_VISIBLE_DEVICES=0 python tools/train.py --cfg configs/cifar100/energy_aug/res56_res20.yaml --energy

CUDA_VISIBLE_DEVICES=0 python tools/train.py --cfg configs/cifar100/energy_aug/res32x4_res8x4.yaml --energy

CUDA_VISIBLE_DEVICES=0 python tools/train.py --cfg configs/cifar100/energy_aug/vgg13_vgg8.yaml --energy

CUDA_VISIBLE_DEVICES=0 python tools/train.py --cfg configs/cifar100/energy_aug/res32x4_shuv2.yaml --energy

CUDA_VISIBLE_DEVICES=0 python tools/train.py --cfg configs/cifar100/energy_aug/vgg13_mv2.yaml --energy

