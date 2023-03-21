gpu=0;
data_dir="../wilds"; # setup your data directory for the datasets
exp_dir=""; # the experiment directory will be the same as data_dir by default

for seed in {0..2};
do
CUDA_VISIBLE_DEVICES=${gpu} WANDB_MODE=offline python3 ./src/main.py --need_pretrain --use_old --data-dir ${data_dir} --dataset civil --algorithm pair -pc 3 -al -ac 1e-4 --seed ${seed} 
done;
