#!/bin/bash

vid=2
noniid_level=10
batch_size=4
dataset_size=1200
log_path='log_'${dataset_size}
save_dir='res_'${dataset_size}

dataset='fmnist'

########## lr wavg/fedavg ###########
:<<!
for ((i=10; i<=50; i=i+10)); do {
for learning_rate in 0.1 0.05 0.01 0.005 0.001; do {

# lr-noniid-nodp
log_dir=${log_path}'/log_'$vid'/'${dataset}'/lr/noniid'${noniid_level}'/nodp'
echo $log_dir'-lr-noniid-nodp-'$i

nohup python main.py --max_steps 10000 --dataset $dataset --model lr --lr $learning_rate --N $i --client_dataset_size ${dataset_size} --noniid True --noniid_level ${noniid_level} --num_microbatches ${batch_size} --client_batch_size ${batch_size} --sample_mode R --sample_ratio 0.8 --local_steps 100 --version $vid --save_dir ${save_dir} >${log_dir}/${i}_bs${batch_size}_nm${batch_size}_10000_100_R8_nodp_constlr${learning_rate} 2>&1 &

# lr-iid-nodp
log_dir=${log_path}'/log_'$vid'/'${dataset}'/lr/iid/nodp'
echo $log_dir'-lr-iid-nodp-'$i

nohup python main.py --max_steps 10000 --dataset $dataset --model lr --lr $learning_rate --N $i --client_dataset_size ${dataset_size} --num_microbatches ${batch_size} --client_batch_size ${batch_size} --sample_mode R --sample_ratio 0.8 --local_steps 100 --version $vid --save_dir ${save_dir} >${log_dir}/${i}_bs${batch_size}_nm${batch_size}_10000_100_R8_nodp_constlr${learning_rate} 2>&1 &

} done 
wait
} done 
wait
!
########## cnn wavg/fedavg ###########

for ((i=10; i<=10; i=i+10)); do {
for learning_rate in 0.001 0.05 0.1; do {

# cnn-noniid-nodp
log_dir=${log_path}'/log_'$vid'/'${dataset}'/cnn/noniid'${noniid_level}'/nodp'
echo $log_dir'-cnn-noniid-nodp-'$i
nohup python main.py --max_steps 10000 --dataset $dataset --model cnn --lr $learning_rate --N $i --client_dataset_size ${dataset_size} --noniid True --noniid_level ${noniid_level} --num_microbatches ${batch_size} --client_batch_size ${batch_size} --sample_mode R --sample_ratio 0.8 --local_steps 100 --version $vid --save_dir ${save_dir} >${log_dir}/${i}_bs${batch_size}_nm${batch_size}_10000_100_R8_nodp_constlr${learning_rate} 2>&1 &

# cnn-iid-nodp
log_dir=${log_path}'/log_'$vid'/'${dataset}'/cnn/iid/nodp'
echo $log_dir'-cnn-iid-nodp-'$i
nohup python main.py --max_steps 10000 --dataset $dataset --model cnn --lr $learning_rate --N $i --client_dataset_size ${dataset_size} --num_microbatches ${batch_size} --client_batch_size ${batch_size} --sample_mode R --sample_ratio 0.8 --local_steps 100 --version $vid --save_dir ${save_dir} >${log_dir}/${i}_bs${batch_size}_nm${batch_size}_10000_100_R8_nodp_constlr${learning_rate} 2>&1 &

} done 
wait
} done 
wait
