import os

# for i in range(2):
#     cmd = f'python Old_CIFAR_benchmark_final_26.47_93.36_25.py --dataset=cifar100 --model=densenet_dice --auxiliary_trans=ood --train_trans=train --num_component=25 --acquisition=ucb --kappa=2.576 --lower_percentile=65.001 --upper_percentile=100 --ash_percentile=64.999 --metric=fpr --percent=15 > nmf_cifar100_comp25_seed{i+8}.log'
#     print(cmd)
#     os.system(cmd)
#     print(i) 

# for i in range(5):
#     cmd = f'python Old_CIFAR_benchmark_final_26.47_93.36_30.py --dataset=cifar100 --model=densenet_dice --auxiliary_trans=ood --train_trans=train --num_component=30 --acquisition=ucb --kappa=2.576 --lower_percentile=65.001 --upper_percentile=100 --ash_percentile=64.999 --metric=fpr --percent=15 > nmf_cifar100_comp30_seed{i+5}.log'
#     print(cmd)
#     os.system(cmd)
#     print(i) 

# for i in range(5):
#     cmd = f'python Old_CIFAR_benchmark_final_26.47_93.36_35.py --dataset=cifar100 --model=densenet_dice --auxiliary_trans=ood --train_trans=train --num_component=35 --acquisition=ucb --kappa=2.576 --lower_percentile=65.001 --upper_percentile=100 --ash_percentile=64.999 --metric=fpr --percent=15 > nmf_cifar100_comp35_seed{i+5}.log'
#     print(cmd)
#     os.system(cmd)
#     print(i)

# for i in range(10):
#     cmd = f'python Old_CIFAR_benchmark_final_26.47_93.36_60.py --dataset=cifar100 --model=densenet_dice --auxiliary_trans=ood --train_trans=train --num_component=60 --acquisition=ucb --kappa=2.576 --lower_percentile=65.001 --upper_percentile=100 --ash_percentile=64.999 --metric=fpr --percent=15 > nmf_cifar100_comp60_seed{i}.log'
#     print(cmd)
#     os.system(cmd)
#     print(i) 

# for i in range(10):
#     cmd = f'python Old_CIFAR_benchmark_final_26.47_93.36_65.py --dataset=cifar100 --model=densenet_dice --auxiliary_trans=ood --train_trans=train --num_component=65 --acquisition=ucb --kappa=2.576 --lower_percentile=65.001 --upper_percentile=100 --ash_percentile=64.999 --metric=fpr --percent=15 > nmf_cifar100_comp65_seed{i}.log'
#     print(cmd)
#     os.system(cmd)
#     print(i) 

# for i in range(7):
#     cmd = f'python Old_CIFAR_benchmark_final_26.47_93.36_70.py --dataset=cifar100 --model=densenet_dice --auxiliary_trans=ood --train_trans=train --num_component=70 --acquisition=ucb --kappa=2.576 --lower_percentile=65.001 --upper_percentile=100 --ash_percentile=64.999 --metric=fpr --percent=15 > nmf_cifar100_comp70_seed{i+2}.log'
#     print(cmd)
#     os.system(cmd)
#     print(i)

# for i in range(9):
#     cmd = f'python Old_CIFAR_benchmark_final_26.47_93.36_75.py --dataset=cifar100 --model=densenet_dice --auxiliary_trans=ood --train_trans=train --num_component=75 --acquisition=ucb --kappa=2.576 --lower_percentile=65.001 --upper_percentile=100 --ash_percentile=64.999 --metric=fpr --percent=15 > nmf_cifar100_comp75_seed{i+1}.log'
#     print(cmd)
#     os.system(cmd)
#     print(i)

# for i in range(10):
#     cmd = f'python Old_CIFAR_benchmark_final_26.47_93.36_100.py --dataset=cifar100 --model=densenet_dice --auxiliary_trans=ood --train_trans=train --num_component=100 --acquisition=ucb --kappa=2.576 --lower_percentile=65.001 --upper_percentile=100 --ash_percentile=64.999 --metric=fpr --percent=15 > nmf_cifar100_comp100_seed{i}.log'
#     print(cmd)
#     os.system(cmd)
#     print(i) 

# for i in range(8):
#     cmd = f'python nmf_bo_clamping_densenet161.py --model=densenet161 --percent=5 --ash_bound=74.999 --react_lower_bound=75.001 --react_upper_bound=100 --num_component=50 --seed={i+2} > save_imagenet_densenet161_{i+2}.log'
#     print(cmd)
#     os.system(cmd)
#     print(i) 

# for i in range(20):
#     cmd = f'python nmf_bo_clamping_densenet161.py --model=wide_resnet50_2 --percent=10 --ash_bound=74.999 --react_lower_bound=75.001 --react_upper_bound=100 --num_component=50 --seed={i} > save_imagenet_wide_resnet50_2_{i}.log'
#     print(cmd)
#     os.system(cmd)
#     print(i) 

# cmd = f'python nmf_bo_clamping_densenet161.py --model=mobilenet_v2 --percent=10 --ash_bound=74.999 --react_lower_bound=75.001 --react_upper_bound=100 --num_component=50 --seed=16 > save_imagenet_mobilenet_v2_16.log'
# print(cmd)
# os.system(cmd)

# cmd = f'python nmf_bo_clamping_densenet161.py --model=wide_resnet50_2 --percent=10 --ash_bound=74.999 --react_lower_bound=75.001 --react_upper_bound=100 --num_component=50 --seed=16 > save_imagenet_wide_resnet50_2_16.log'
# print(cmd)
# os.system(cmd)

# cmd = f'python nmf_bo_clamping_densenet161.py --model=mobilenet_v2 --percent=5 --ash_bound=74.999 --react_lower_bound=75.001 --react_upper_bound=100 --num_component=50 --seed=16 > save_imagenet_mobilenet_v2_16_5.log'
# print(cmd)
# os.system(cmd)

# cmd = f'python nmf_bo_clamping_densenet161.py --model=mobilenet_v2 --percent=15 --ash_bound=74.999 --react_lower_bound=75.001 --react_upper_bound=100 --num_component=50 --seed=16 > save_imagenet_mobilenet_v2_16_15.log'
# print(cmd)
# os.system(cmd)

# cmd = f'python dice_final.py --dataset=imagenet --model=resnet50'
# print(cmd)
# os.system(cmd)

# cmd = f'python dice_final.py --dataset=imagenet --model=densenet161'
# print(cmd)
# os.system(cmd)

# cmd = f'python dice_final.py --dataset=imagenet --model=wide_resnet50_2'
# print(cmd)
# os.system(cmd)

# cmd = f'python dice_final.py --dataset=imagenet --model=mobilenet_v2'
# print(cmd)
# os.system(cmd)

# cmd = f'python baseline_final.py --dataset=imagenet --model=resnet50'
# print(cmd)
# os.system(cmd)

# cmd = f'python baseline_final.py --dataset=imagenet --model=densenet161'
# print(cmd)
# os.system(cmd)

# cmd = f'python baseline_final.py --dataset=imagenet --model=wide_resnet50_2'
# print(cmd)
# os.system(cmd)

# cmd = f'python baseline_final.py --dataset=imagenet --model=mobilenet_v2'
# print(cmd)
# os.system(cmd)

# for i in range(10):
#     cmd = f'python Old_CIFAR_benchmark_final_26.47_93.36_55.py --dataset=cifar10 --model=densenet_dice --auxiliary_trans=ood --train_trans=train --num_component=55 --acquisition=ucb --kappa=2.576 --lower_percentile=65.001 --upper_percentile=100 --ash_percentile=64.999 --metric=fpr --percent=5 > nmf_cifar10_comp55_seed{i}.log'
#     print(cmd)
#     os.system(cmd)
#     print(i) 

# for i in range(10):
#     cmd = f'python Old_CIFAR_benchmark_final_26.47_93.36_60.py --dataset=cifar10 --model=densenet_dice --auxiliary_trans=ood --train_trans=train --num_component=60 --acquisition=ucb --kappa=2.576 --lower_percentile=65.001 --upper_percentile=100 --ash_percentile=64.999 --metric=fpr --percent=5 > nmf_cifar10_comp60_seed{i}.log'
#     print(cmd)
#     os.system(cmd)
#     print(i) 

# for i in range(10):
#     cmd = f'python Old_CIFAR_benchmark_final_26.47_93.36_65.py --dataset=cifar10 --model=densenet_dice --auxiliary_trans=ood --train_trans=train --num_component=65 --acquisition=ucb --kappa=2.576 --lower_percentile=65.001 --upper_percentile=100 --ash_percentile=64.999 --metric=fpr --percent=5 > nmf_cifar10_comp65_seed{i}.log'
#     print(cmd)
#     os.system(cmd)
#     print(i) 

# for i in range(10):
#     cmd = f'python Old_CIFAR_benchmark_final_26.47_93.36_5.py --dataset=cifar10 --model=densenet_dice --auxiliary_trans=ood --train_trans=train --num_component=5 --acquisition=ucb --kappa=2.576 --lower_percentile=65.001 --upper_percentile=100 --ash_percentile=64.999 --metric=fpr --percent=5 --seed={i} > nmf_cifar10_comp5_seed{i}_final.log'
#     print(cmd)
#     os.system(cmd)
#     print(i) 

# for i in range(10):
#     cmd = f'python Old_CIFAR_benchmark_final_26.47_93.36_10.py --dataset=cifar10 --model=densenet_dice --auxiliary_trans=ood --train_trans=train --num_component=10 --acquisition=ucb --kappa=2.576 --lower_percentile=65.001 --upper_percentile=100 --ash_percentile=64.999 --metric=fpr --percent=5 --seed={i} > nmf_cifar10_comp10_seed{i}_final.log'
#     print(cmd)
#     os.system(cmd)
#     print(i) 

# for i in range(10):
#     cmd = f'python Old_CIFAR_benchmark_final_26.47_93.36_20.py --dataset=cifar10 --model=densenet_dice --auxiliary_trans=ood --train_trans=train --num_component=20 --acquisition=ucb --kappa=2.576 --lower_percentile=65.001 --upper_percentile=100 --ash_percentile=64.999 --metric=fpr --percent=5 --seed={i} > nmf_cifar10_comp20_seed{i}_final.log'
#     print(cmd)
#     os.system(cmd)
#     print(i) 

# for i in range(10):
#     cmd = f'python Old_CIFAR_benchmark_final_26.47_93.36_50.py --dataset=cifar10 --model=densenet_dice --auxiliary_trans=ood --train_trans=train --num_component=50 --acquisition=ucb --kappa=2.576 --lower_percentile=65.001 --upper_percentile=100 --ash_percentile=64.999 --metric=fpr --percent=5 --seed={i} > nmf_cifar10_comp50_seed{i}_final.log'
#     print(cmd)
#     os.system(cmd)
#     print(i)

# for i in range(10):
#     cmd = f'python Old_CIFAR_benchmark_final_26.47_93.36_5.py --dataset=cifar100 --model=densenet_dice --auxiliary_trans=ood --train_trans=train --num_component=5 --acquisition=ucb --kappa=2.576 --lower_percentile=65.001 --upper_percentile=100 --ash_percentile=64.999 --metric=fpr --percent=15 --seed={i} > nmf_cifar100_comp5_seed{i}_final.log'
#     print(cmd)
#     os.system(cmd)
#     print(i) 

# for i in range(10):
#     cmd = f'python Old_CIFAR_benchmark_final_26.47_93.36_10.py --dataset=cifar100 --model=densenet_dice --auxiliary_trans=ood --train_trans=train --num_component=10 --acquisition=ucb --kappa=2.576 --lower_percentile=65.001 --upper_percentile=100 --ash_percentile=64.999 --metric=fpr --percent=15 --seed={i} > nmf_cifar100_comp10_seed{i}_final.log'
#     print(cmd)
#     os.system(cmd)
#     print(i) 

# for i in range(10):
#     cmd = f'python Old_CIFAR_benchmark_final_26.47_93.36_20.py --dataset=cifar100 --model=densenet_dice --auxiliary_trans=ood --train_trans=train --num_component=20 --acquisition=ucb --kappa=2.576 --lower_percentile=65.001 --upper_percentile=100 --ash_percentile=64.999 --metric=fpr --percent=15 --seed={i} > nmf_cifar100_comp20_seed{i}_final.log'
#     print(cmd)
#     os.system(cmd)
#     print(i) 

# for i in range(10):
#     cmd = f'python Old_CIFAR_benchmark_final_26.47_93.36_50.py --dataset=cifar100 --model=densenet_dice --auxiliary_trans=ood --train_trans=train --num_component=50 --acquisition=ucb --kappa=2.576 --lower_percentile=65.001 --upper_percentile=100 --ash_percentile=64.999 --metric=fpr --percent=15 --seed={i} > nmf_cifar100_comp50_seed{i}_final.log'
#     print(cmd)
#     os.system(cmd)
#     print(i) 

# for i in range(10):
#     cmd = f'python Old_CIFAR_benchmark_final_26.47_93.36_80.py --dataset=cifar100 --model=densenet_dice --auxiliary_trans=ood --train_trans=train --num_component=80 --acquisition=ucb --kappa=2.576 --lower_percentile=65.001 --upper_percentile=100 --ash_percentile=64.999 --metric=fpr --percent=15 --seed={i} > nmf_cifar100_comp80_seed{i}_final.log'
#     print(cmd)
#     os.system(cmd)
#     print(i) 

# for i in range(10):
#     cmd = f'python Old_CIFAR_benchmark_final_26.47_93.36_100.py --dataset=cifar100 --model=densenet_dice --auxiliary_trans=ood --train_trans=train --num_component=100 --acquisition=ucb --kappa=2.576 --lower_percentile=65.001 --upper_percentile=100 --ash_percentile=64.999 --metric=fpr --percent=15 --seed={i} > nmf_cifar100_comp100_seed{i}_final.log'
#     print(cmd)
#     os.system(cmd)
#     print(i)

# cmd = f'python Old_CIFAR_benchmark_final_26.47_93.36_10.py --dataset=cifar10 --model=densenet_dice --auxiliary_trans=ood --train_trans=train --num_component=10 --acquisition=ucb --kappa=2.576 --lower_percentile=65.001 --upper_percentile=100 --ash_percentile=64.999 --metric=fpr --percent=5 --seed=1 --score=msp > nmf_cifar10_comp10_seed1_msp_final.log'
# print(cmd)
# os.system(cmd)

# cmd = f'python Old_CIFAR_benchmark_final_26.47_93.36_10.py --dataset=cifar10 --model=densenet_dice --auxiliary_trans=ood --train_trans=train --num_component=10 --acquisition=ucb --kappa=2.576 --lower_percentile=65.001 --upper_percentile=100 --ash_percentile=64.999 --metric=fpr --percent=5 --seed=1 --score=maxlogit > nmf_cifar10_comp10_seed1_maxlogit_final.log'
# print(cmd)
# os.system(cmd)

# for i in [50, 55]:
#     cmd = f'python Old_CIFAR_benchmark_final_26.47_93.36_10.py --dataset=cifar10 --model=densenet_dice --auxiliary_trans=ood --train_trans=train --num_component=10 --acquisition=ucb --kappa=2.576 --lower_percentile={i}.001 --upper_percentile=100 --ash_percentile={i-1}.999 --metric=fpr --percent=5 --seed=1 --score=energy > nmf_cifar10_comp10_seed1_{i}_final.log'
#     print(cmd)
#     os.system(cmd)
#     print(i) 

# for i in [60, 65]:
#     cmd = f'python Old_CIFAR_benchmark_final_26.47_93.36_10.py --dataset=cifar10 --model=densenet_dice --auxiliary_trans=ood --train_trans=train --num_component=10 --acquisition=ucb --kappa=2.576 --lower_percentile={i}.001 --upper_percentile=100 --ash_percentile={i-1}.999 --metric=fpr --percent=5 --seed=1 --score=energy > nmf_cifar10_comp10_seed1_{i}_final.log'
#     print(cmd)
#     os.system(cmd)
#     print(i) 

# for i in [70, 75]:
#     cmd = f'python Old_CIFAR_benchmark_final_26.47_93.36_10.py --dataset=cifar10 --model=densenet_dice --auxiliary_trans=ood --train_trans=train --num_component=10 --acquisition=ucb --kappa=2.576 --lower_percentile={i}.001 --upper_percentile=100 --ash_percentile={i-1}.999 --metric=fpr --percent=5 --seed=1 --score=energy > nmf_cifar10_comp10_seed1_{i}_final.log'
#     print(cmd)
#     os.system(cmd)
#     print(i) 

# for i in [80, 85]:
#     cmd = f'python Old_CIFAR_benchmark_final_26.47_93.36_10.py --dataset=cifar10 --model=densenet_dice --auxiliary_trans=ood --train_trans=train --num_component=10 --acquisition=ucb --kappa=2.576 --lower_percentile={i}.001 --upper_percentile=100 --ash_percentile={i-1}.999 --metric=fpr --percent=5 --seed=1 --score=energy > nmf_cifar10_comp10_seed1_{i}_final.log'
#     print(cmd)
#     os.system(cmd)
#     print(i) 


# for i in [50, 55]:
#     cmd = f'python Old_CIFAR_benchmark_final_26.47_93.36_50.py --dataset=cifar100 --model=densenet_dice --auxiliary_trans=ood --train_trans=train --num_component=50 --acquisition=ucb --kappa=2.576 --lower_percentile={i}.001 --upper_percentile=100 --ash_percentile={i-1}.999 --metric=fpr --percent=15 --seed=1 --score=energy > nmf_cifar100_comp50_seed1_{i}_final.log'
#     print(cmd)
#     os.system(cmd)
#     print(i) 

# for i in [60, 65]:
#     cmd = f'python Old_CIFAR_benchmark_final_26.47_93.36_50.py --dataset=cifar100 --model=densenet_dice --auxiliary_trans=ood --train_trans=train --num_component=50 --acquisition=ucb --kappa=2.576 --lower_percentile={i}.001 --upper_percentile=100 --ash_percentile={i-1}.999 --metric=fpr --percent=15 --seed=1 --score=energy > nmf_cifar100_comp50_seed1_{i}_final.log'
#     print(cmd)
#     os.system(cmd)
#     print(i) 

# for i in [70, 75]:
#     cmd = f'python Old_CIFAR_benchmark_final_26.47_93.36_50.py --dataset=cifar100 --model=densenet_dice --auxiliary_trans=ood --train_trans=train --num_component=50 --acquisition=ucb --kappa=2.576 --lower_percentile={i}.001 --upper_percentile=100 --ash_percentile={i-1}.999 --metric=fpr --percent=15 --seed=1 --score=energy > nmf_cifar100_comp50_seed1_{i}_final.log'
#     print(cmd)
#     os.system(cmd)
#     print(i) 

# for i in [80, 85]:
#     cmd = f'python Old_CIFAR_benchmark_final_26.47_93.36_50.py --dataset=cifar100 --model=densenet_dice --auxiliary_trans=ood --train_trans=train --num_component=50 --acquisition=ucb --kappa=2.576 --lower_percentile={i}.001 --upper_percentile=100 --ash_percentile={i-1}.999 --metric=fpr --percent=15 --seed=1 --score=energy > nmf_cifar100_comp50_seed1_{i}_final.log'
#     print(cmd)
#     os.system(cmd)
#     print(i) 

# cmd = f'python Old_CIFAR_benchmark_final_26.47_93.36_50.py --dataset=cifar100 --model=densenet_dice --auxiliary_trans=ood --train_trans=train --num_component=50 --acquisition=ucb --kappa=2.576 --lower_percentile=65.001 --upper_percentile=100 --ash_percentile=64.999 --metric=fpr --percent=15 --seed=1 --score=energy > nmf_cifar100_comp50_seed1_r_final.log'
# print(cmd)
# os.system(cmd)

# cmd = f'python Old_CIFAR_benchmark_final_26.47_93.36_50.py --dataset=cifar100 --model=densenet_dice --auxiliary_trans=ood --train_trans=train --num_component=50 --acquisition=ucb --kappa=2.576 --lower_percentile=65.001 --upper_percentile=100 --ash_percentile=64.999 --metric=fpr --percent=15 --seed=1 --score=energy > nmf_cifar100_comp50_seed1_a_final.log'
# print(cmd)
# os.system(cmd)

# cmd = f'python Old_CIFAR_benchmark_final_26.47_93.36_50.py --dataset=cifar100 --model=densenet_dice --auxiliary_trans=ood --train_trans=train --num_component=50 --acquisition=ucb --kappa=2.576 --lower_percentile=65.001 --upper_percentile=100 --ash_percentile=64.999 --metric=fpr --percent=15 --seed=1 --score=energy --method=nmf > cifar100_comp50_seed1_nmf_final.log'
# print(cmd)
# os.system(cmd)

# cmd = f'python Old_CIFAR_benchmark_final_26.47_93.36_50.py --dataset=cifar100 --model=densenet_dice --auxiliary_trans=ood --train_trans=train --num_component=50 --acquisition=ucb --kappa=2.576 --lower_percentile=65.001 --upper_percentile=100 --ash_percentile=64.999 --metric=fpr --percent=15 --seed=1 --score=energy --method=pca > cifar100_comp50_seed1_pca_final.log'
# print(cmd)
# os.system(cmd)

# cmd = f'python Old_CIFAR_benchmark_final_26.47_93.36_50.py --dataset=cifar100 --model=densenet_dice --auxiliary_trans=ood --train_trans=train --num_component=50 --acquisition=ucb --kappa=2.576 --lower_percentile=65.001 --upper_percentile=100 --ash_percentile=64.999 --metric=fpr --percent=15 --seed=1 --score=energy --method=ica > cifar100_comp50_seed1_ica_final.log'
# print(cmd)
# os.system(cmd)

for i in range(50):
    cmd = f'python insrect_patterns.py --model=densenet_dice --dataset=cifar100 --iter=0 --index={i}'
    print(cmd)
    os.system(cmd)

    cmd = f'python insrect_patterns.py --model=densenet_dice --dataset=cifar100 --iter=500  --method=pca --index={i}'
    print(cmd)
    os.system(cmd)

    cmd = f'python insrect_patterns.py --model=densenet_dice --dataset=cifar100 --iter=500  --method=nmf --index={i}'
    print(cmd)
    os.system(cmd)

    cmd = f'python insrect_patterns.py --model=densenet_dice --dataset=cifar100 --iter=500  --method=ica --index={i}'
    print(cmd)
    os.system(cmd)

