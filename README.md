
# Energy KD
**Maximizing discrimination capability of knowledge distillation with energy function**

Seonghak Kim, Gyeongdo Ham, Suin Lee, Donggon Jang, Daeshik Kim

This provides an implementation of the code for "[Maximizing discrimination capability of knowledge distillation with energy function](https://doi.org/10.1016/j.knosys.2024.111911)", as published in the _Knowledge-Based Systems_.

## Installation

Environments:

- Python 3.8
- PyTorch 1.10.1
- torchvision 0.11.2

Install the package:

```
pip install -r requirements.txt
python setup.py develop
```

## Getting started

1. Evaluation

- You can see `energy_eval.sh`.


  ```bash 
  # Energy KD
  python tools/eval.py -m wrn_16_2 -c ./best_results/energykd/wrn16_2_student_best
  python tools/eval.py -m resnet20 -c ./best_results/energykd/res20_student_best
  python tools/eval.py -m vgg8 -c ./best_results/energykd/vgg8_student_best
  python tools/eval.py -m ShuffleV2 -c ./best_results/energykd/shuv2_student_best
  python tools/eval.py -m MobileNetV2 -c ./best_results/energykd/mv2_student_best

  # Energy DKD
  python tools/eval.py -m vgg8 -c ./best_results/energydkd/vgg8_student_best

  # KD w/ HE-DA
  python tools/eval.py -m wrn_16_2 -c ./best_results/energyaug/wrn16_2_student_best
  python tools/eval.py -m resnet20 -c ./best_results/energyaug/res20_student_best
  python tools/eval.py -m resnet8x4 -c ./best_results/energyaug/res8x4_student_best
  python tools/eval.py -m vgg8 -c ./best_results/energyaug/vgg8_student_best
  python tools/eval.py -m ShuffleV2 -c ./best_results/energyaug/shuv2_student_best
  python tools/eval.py -m MobileNetV2 -c ./best_results/energyaug/mv2_student_best
  ```


2. Training

- The weights of the teacher models can be created before implementing Energy KD. (Refer to `teacher.sh`)

  ```bash
  # Teacher models
  python tools/train.py --cfg configs/cifar100/teacher/wrn40_2.yaml
  python tools/train.py --cfg configs/cifar100/teacher/res32x4.yaml
  python tools/train.py --cfg configs/cifar100/teacher/vgg13.yaml

  # Move `student_best` into `download_ckpts/cifar_teachers/`
  mv output/cifar100_teacher/teacher,wrn40_2/student_best download_ckpts/cifar_teachers/wrn_40_2_vanilla/ckpt_epoch_240.pth
  mv output/cifar100_teacher/teacher,res32x4/student_best download_ckpts/cifar_teachers/resnet32x4_vanilla/ckpt_epoch_240.pth
  mv output/cifar100_teacher/teacher,vgg13/student_best download_ckpts/cifar_teachers/vgg13_vanilla/ckpt_epoch_240.pth
  ```

- If the teacher models are prepared, you can train the student models using Energy KD, as demonstrated in the `energy_train.sh`.

  ```bash
  # Energy KD
  python tools/train.py --cfg configs/cifar100/energy_kd/wrn40_2_wrn16_2.yaml --energy
  python tools/train.py --cfg configs/cifar100/energy_kd/res56_res20.yaml --energy
  python tools/train.py --cfg configs/cifar100/energy_kd/vgg13_vgg8.yaml --energy
  python tools/train.py --cfg configs/cifar100/energy_kd/res32x4_shuv2.yaml --energy
  python tools/train.py --cfg configs/cifar100/energy_kd/vgg13_mv2.yaml --energy

  # Energy DKD
  python tools/train.py --cfg configs/cifar100/energy_dkd/vgg13_vgg8.yaml --energy

  # KD w/ HE-DA
  python tools/train.py --cfg configs/cifar100/energy_aug/wrn40_2_wrn16_2.yaml --energy
  python tools/train.py --cfg configs/cifar100/energy_aug/res56_res20.yaml --energy
  python tools/train.py --cfg configs/cifar100/energy_aug/res32x4_res8x4.yaml --energy
  python tools/train.py --cfg configs/cifar100/energy_aug/vgg13_vgg8.yaml --energy
  python tools/train.py --cfg configs/cifar100/energy_aug/res32x4_shuv2.yaml --energy
  python tools/train.py --cfg configs/cifar100/energy_aug/vgg13_mv2.yaml --energy
  ```

## Citation

Please consider citing **Energy KD** and **HE-DA** in your publications if it helps your research.

```bib
@article{kim2024maximizing,
  title={Maximizing discrimination capability of knowledge distillation with energy function},
  author={Kim, Seonghak and Ham, Gyeongdo and Lee, Suin and Jang, Donggon and Kim, Daeshik},
  journal={Knowledge-Based Systems},
  volume={296},
  pages={111911},
  year={2024},
  publisher={Elsevier}
}
```

## Acknowledgement

This code is built on [mdistiller](<https://github.com/megvii-research/mdistiller>).

Thanks to the contributors of mdistiller for your exceptional efforts.
