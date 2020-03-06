set -ex

if [ -z "$1" ]; then
    gens=300000; #default in program 500000
else
    gens=$1;
fi

python3 train.py  \
    --name moegan_fmnist \
    --dataset_mode torchvision --download_root ./datasets/FashionMNIST --dataset_name FashionMNIST \
    --batch_size 32 --eval_size 256 --dataroot None \
    --model moegan --gpu_ids 0\
    --preprocess "" --input_nc 1 --output_nc 1 --crop_size 28 \
    --z_dim 100 --z_type Uniform \
    --d_loss_mode vanilla --g_loss_mode vanilla lsgan --which_D S \
    --lambda_f 0.05 --candi_num 3 \
    --netD DCGAN_mnist --netG DCGAN_mnist --ngf 32 --ndf 32 --g_norm none \
    --d_norm batch --init_type normal \
    --init_gain 0.02 --no_dropout --no_flip --D_iters 1\
    --score_name TOY_METRIC --evaluation_size 500 --fid_batch_size 50 \
    --print_freq 1000 --display_freq 1000 --score_freq 1000 \
    --display_id -1 --save_giters_freq 1000 \
    --total_num_giters ${gens}

     #--D_iters 3

# --z_dim 100 --z_type Gaussian \
