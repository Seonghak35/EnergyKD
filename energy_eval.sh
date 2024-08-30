
### Energy KD ###

CUDA_VISIBLE_DEVICES=0 python tools/eval.py -m wrn_16_2 -c ./best_results/energykd/wrn16_2_student_best

CUDA_VISIBLE_DEVICES=0 python tools/eval.py -m resnet20 -c ./best_results/energykd/res20_student_best

CUDA_VISIBLE_DEVICES=0 python tools/eval.py -m vgg8 -c ./best_results/energykd/vgg8_student_best

CUDA_VISIBLE_DEVICES=0 python tools/eval.py -m ShuffleV2 -c ./best_results/energykd/shuv2_student_best

CUDA_VISIBLE_DEVICES=0 python tools/eval.py -m MobileNetV2 -c ./best_results/energykd/mv2_student_best


### Energy DKD ###

CUDA_VISIBLE_DEVICES=0 python tools/eval.py -m vgg8 -c ./best_results/energydkd/vgg8_student_best


### HE-DA ###

CUDA_VISIBLE_DEVICES=0 python tools/eval.py -m wrn_16_2 -c ./best_results/energyaug/wrn16_2_student_best

CUDA_VISIBLE_DEVICES=0 python tools/eval.py -m resnet20 -c ./best_results/energyaug/res20_student_best

CUDA_VISIBLE_DEVICES=0 python tools/eval.py -m resnet8x4 -c ./best_results/energyaug/res8x4_student_best

CUDA_VISIBLE_DEVICES=0 python tools/eval.py -m vgg8 -c ./best_results/energyaug/vgg8_student_best

CUDA_VISIBLE_DEVICES=0 python tools/eval.py -m ShuffleV2 -c ./best_results/energyaug/shuv2_student_best

CUDA_VISIBLE_DEVICES=0 python tools/eval.py -m MobileNetV2 -c ./best_results/energyaug/mv2_student_best
