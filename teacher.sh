
### teacher model ###

CUDA_VISIBLE_DEVICES=0 python tools/train.py --cfg configs/cifar100/teacher/wrn40_2.yaml

CUDA_VISIBLE_DEVICES=0 python tools/train.py --cfg configs/cifar100/teacher/res32x4.yaml

CUDA_VISIBLE_DEVICES=0 python tools/train.py --cfg configs/cifar100/teacher/vgg13.yaml

### move files ###

mv output/cifar100_teacher/teacher,wrn40_2/student_best download_ckpts/cifar_teachers/wrn_40_2_vanilla/ckpt_epoch_240.pth
mv output/cifar100_teacher/teacher,res32x4/student_best download_ckpts/cifar_teachers/resnet32x4_vanilla/ckpt_epoch_240.pth
mv output/cifar100_teacher/teacher,vgg13/student_best download_ckpts/cifar_teachers/vgg13_vanilla/ckpt_epoch_240.pth
