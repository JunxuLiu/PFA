#!/bin/bash

# python main.py --max_steps 10000 --dataset mnist --model cnn --lr 0.075 --N 20 --client_dataset_size 1200 --noniid True --noniid_level 10 --num_microbatches 4 --client_batch_size 4 --sample_mode R --sample_ratio 0.8 --local_steps 100 --dpsgd True --eps mixgauss2 --proj_wavg True --proj_dims 1 --lanczos_iter 256 --version 2 --save_dir res_test

# python main.py --max_steps 10000 --dataset mnist --model cnn --lr 0.01 --N 20 --client_dataset_size 1200 --num_microbatches 4 --client_batch_size 4 --sample_mode R --sample_ratio 0.8 --local_steps 100 --dpsgd True --eps mixgauss3 --projection True --proj_dims 2 --lanczos_iter 256 --version 2 --save_dir res_test

# python main.py --max_steps 10000 --dataset mnist --model cnn --lr 0.005 --N 50 --client_dataset_size 1200 --num_microbatches 4 --client_batch_size 4 --sample_mode R --sample_ratio 0.8 --local_steps 100 --dpsgd True --eps mixgauss2 --proj_wavg True --proj_dims 2 --lanczos_iter 256 --version 2 --save_dir res_test

# nohup python main.py --max_steps 10000 --dataset mnist --model cnn --lr 0.01 --N 20 --client_dataset_size 1200 --noniid True --noniid_level 10 --num_microbatches 4 --client_batch_size 4 --sample_mode R --sample_ratio 0.8 --local_steps 100 --dpsgd True --eps mixgauss2 --proj_wavg True --proj_dims 2 --lanczos_iter 256 --version 2 --save_dir res_test_k >log_test_k 2>&1 &

vid=2
noniid_level=10
local_steps=100
batch_size=4
dataset_size=1200
log_path='../PFA_res/log_wproj_'${dataset_size}
save_dir='../PFA_res/res_wproj_'${dataset_size}

#distributions=('mixgauss1' 'mixgauss2' 'mixgauss4' 'mixgauss5' 'gauss1' 'pareto1' 'uniform1')
#distributions=('mixgauss4' 'mixgauss5' 'gauss2' 'pareto1' 'uniform1')
#distributions=('min1' 'mixgauss1' 'gauss1' 'pareto1' 'uniform1')
distributions=('mixgauss2')
#distributions=('min1' 'min2' 'max1' 'max2')

dataset='mnist'
:<<!
########## Pfizer ###########
for element in ${distributions[@]}; do {
for i in 20 30 40 50; do {
<<<<<<< HEAD
for learning_rate in 0.01 0.005 0.001; do {
=======
<<<<<<< HEAD
for learning_rate in 0.025 0.05; do {
=======
for learning_rate in 0.005 0.0075; do {
>>>>>>> 9d3ea00288c01531b75f15dd8658a8305270bf4d
>>>>>>> bda3e5881a38170c65add448fec3a4140f42f949

# cnn-iid
log_dir=${log_path}'/log_'$vid'/'${dataset}'/cnn/iid/'${element}
if [ ! -d $log_dir ]
then
echo 'path not exists'
mkdir $log_dir
fi
echo $log_dir'-'$i

nohup python main.py --max_steps 10000 --dataset $dataset --model cnn --lr $learning_rate --N $i --client_dataset_size ${dataset_size} --num_microbatches ${batch_size} --client_batch_size ${batch_size} --sample_mode R --sample_ratio 0.8 --local_steps $local_steps --dpsgd True --eps $element --proj_wavg True --proj_dims 2 --lanczos_iter 256 --version $vid --save_dir ${save_dir} >${log_dir}/${i}_bs${batch_size}_nm${batch_size}_10000_${local_steps}_R8_wpro2_256_constlr${learning_rate} 2>&1 &

} done 
wait
} done 
wait
} done
wait
<<<<<<< HEAD
!
for element in ${distributions[@]}; do {
for i in 20 30 40 50; do {
for learning_rate in 0.01 0.005 0.001; do {
=======

<<<<<<< HEAD
for element in ${distributions[@]}; do {
for i in 20 30 40 50; do {
for learning_rate in 0.025 0.05; do {
=======
:<<!
for element in ${distributions[@]}; do {
for i in 20 30 40 50; do {
for learning_rate in 0.025 0.05 0.01; do {
>>>>>>> 9d3ea00288c01531b75f15dd8658a8305270bf4d
>>>>>>> bda3e5881a38170c65add448fec3a4140f42f949

# cnn-noniid-bs4
log_dir=${log_path}'/log_'$vid'/'${dataset}'/cnn/noniid'${noniid_level}'/'${element}
if [ ! -d $log_dir ]
then
mkdir $log_dir
fi
echo $log_dir'-'$i

nohup python main.py --max_steps 10000 --dataset $dataset --model cnn --lr $learning_rate --N $i --client_dataset_size ${dataset_size} --noniid True --noniid_level ${noniid_level} --num_microbatches ${batch_size} --client_batch_size ${batch_size} --sample_mode R --sample_ratio 0.8 --local_steps ${local_steps}  --dpsgd True --eps $element --proj_wavg True --proj_dims 2 --lanczos_iter 256 --version $vid --save_dir ${save_dir} >${log_dir}/${i}_bs${batch_size}_nm${batch_size}_10000_${local_steps}_R8_wpro2_256_constlr${learning_rate} 2>&1 &

} done 
wait
} done 
wait
} done 
wait
