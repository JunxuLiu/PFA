#!/bin/bash

vid=2
noniid_level=10
local_steps=100
batch_size=4
dataset_size=1200
log_path='log_'${dataset_size}
save_dir='res_'${dataset_size}

#distributions=('mixgauss1' 'mixgauss2' 'mixgauss4' 'mixgauss5' 'gauss1' 'pareto1' 'uniform1')
#distributions=('mixgauss4' 'mixgauss5' 'gauss2' 'pareto1' 'uniform1')
#distributions=('min1' 'mixgauss1' 'mixgauss4' 'mixgauss5' 'gauss1' 'pareto1' 'uniform1')
#distributions=('mixgauss3_2' 'mixgauss3_3' 'mixgauss3_4' 'mixgauss3_5')
#datasets=('mnist' 'fmnist')

# lr
#python main.py --max_steps 10000 --dataset mnist --model lr --lr 0.2 --N 10 --num_microbatches 4 --client_batch_size 4 --sample_mode R --sample_ratio 0.8 --local_steps 100 --dpsgd True --eps mixgauss1 --projection True --proj_dims 1 --lanczos_iter 256 --version 2

:<<!
########## Pfizer ###########
# 0.05 & 0.01 are the best choices for lr&iid

for dataset in ${datasets[@]}; do {
for element in ${distributions[@]}; do {
for learning_rate in 0.001 0.005 0.01 0.05 0.1; do {
i=30
    
# lr-iid
log_dir=${log_path}'/log_'$vid'/'${dataset}'/lr/iid/'${element}
if [ ! -d $log_dir ]
then
mkdir $log_dir
fi
echo $log_dir'-'$i

nohup python main.py --max_steps 10000 --dataset $dataset --model lr --lr $learning_rate --N $i --client_dataset_size ${dataset_size} --num_microbatches ${batch_size} --client_batch_size ${batch_size} --sample_mode R --sample_ratio 0.8 --local_steps $local_steps  --dpsgd True --eps $element --projection True --proj_dims 1 --lanczos_iter 256 --version $vid --save_dir ${save_dir} >${log_dir}/${i}_bs${batch_size}_nm${batch_size}_10000_${local_steps}_R8_pro1_256_constlr${learning_rate} 2>&1 &

# lr-noniid-bs4
log_dir=${log_path}'/log_'$vid'/'${dataset}'/lr/noniid'${noniid_level}'/'${element}
if [ ! -d $log_dir ]
then
mkdir $log_dir
fi
echo $log_dir'-'$i

nohup python main.py --max_steps 10000 --dataset $dataset --model lr --lr $learning_rate --N $i --client_dataset_size ${dataset_size} --noniid True --noniid_level ${noniid_level} --num_microbatches ${batch_size} --client_batch_size ${batch_size} --sample_mode R --sample_ratio 0.8 --local_steps ${local_steps}  --dpsgd True --eps $element --projection True --proj_dims 1 --lanczos_iter 256 --version $vid --save_dir ${save_dir} >${log_dir}/${i}_bs${batch_size}_nm${batch_size}_10000_${local_steps}_R8_pro1_256_constlr${learning_rate} 2>&1 &

} done 
wait
} done 
wait
} done 
wait
!


########## Pfizer ###########
# 0.05 & 0.01 are the best choices for lr&iid

for proj_dims in 2 3 5 10 20 30 50 100; do {
for learning_rate in 0.0005 0.001 0.005; do {
i=30
element=mixgauss3
dataset=mnist
    
# lr-iid
log_dir=${log_path}'/log_'$vid'/'${dataset}'/lr/iid/'${element}
if [ ! -d $log_dir ]
then
mkdir $log_dir
fi
echo $log_dir'-'$i

nohup python main.py --max_steps 10000 --dataset $dataset --model lr --lr $learning_rate --N $i --client_dataset_size ${dataset_size} --num_microbatches ${batch_size} --client_batch_size ${batch_size} --sample_mode R --sample_ratio 0.8 --local_steps $local_steps  --dpsgd True --eps $element --projection True --proj_dims ${proj_dims} --lanczos_iter 256 --version $vid --save_dir ${save_dir} >${log_dir}/${i}_bs${batch_size}_nm${batch_size}_10000_${local_steps}_R8_pro${proj_dims}_256_constlr${learning_rate} 2>&1 &

# lr-noniid-bs4
log_dir=${log_path}'/log_'$vid'/'${dataset}'/lr/noniid'${noniid_level}'/'${element}
if [ ! -d $log_dir ]
then
mkdir $log_dir
fi
echo $log_dir'-'$i

nohup python main.py --max_steps 10000 --dataset $dataset --model lr --lr $learning_rate --N $i --client_dataset_size ${dataset_size} --noniid True --noniid_level ${noniid_level} --num_microbatches ${batch_size} --client_batch_size ${batch_size} --sample_mode R --sample_ratio 0.8 --local_steps ${local_steps}  --dpsgd True --eps $element --projection True --proj_dims ${proj_dims} --lanczos_iter 256 --version $vid --save_dir ${save_dir} >${log_dir}/${i}_bs${batch_size}_nm${batch_size}_10000_${local_steps}_R8_pro${proj_dims}_256_constlr${learning_rate} 2>&1 &

} done 
wait
} done 
wait