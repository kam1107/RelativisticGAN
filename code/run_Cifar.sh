#!/bin/sh

export CUDA_VISIBLE_DEVICES=3;
SEED=1;

NAME="CondGAN-CIFAR10-SEED-"$SEED;

OMP_NUM_THREADS=1

python3 GAN_losses_iter.py --lr_D .0002 --lr_G .0002 --Diters 1 --arch 1 --beta1 .50 --beta2 .999 \
	--gen_every 10000 --print_every 1000 --G_h_size 32 --D_h_size 32 --gen_extra_images 5000 \
	--CIFAR10 True \
	--CIFAR10_input_folder '/home/danny/work/ICCV_2019/Datasets/CIFAR10' \
	--loss_D 1 \
	--image_size 32 \
	--n_iter 520000 \
	--seed $SEED \
	--Giters 1 \
	--spectral True \
	--batch_size    32  \
	--n_gpu         1  \
	--grad_penalty     False   \
	--penalty          10.0   \
	--output_folder "./OUTPUT/"$NAME \
	--extra_folder  "./OUTPUT/"$NAME"_Extra" 
