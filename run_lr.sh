#!/bin/bash

vid=2
noniid_level=10
batch_size=4
dataset_size=1200
log_path='log_'${dataset_size}
save_dir='res_'${dataset_size}

#distributions=('mixgauss1' 'mixgauss2' 'mixgauss4' 'mixgauss5' 'gauss1' 'pareto1' 'uniform1')
#distributions=('mixgauss4' 'mixgauss5' 'gauss2' 'pareto1' 'uniform1')
distributions=('mixgauss2' 'mixgauss3')
#distributions=('min1' 'min2' 'max1' 'max2')

dataset='fmnist'

########## lr wavg/fedavg ###########

for ((i=10; i<=50; i=i+10)); do {
#for learning_rate in 0.001 0.005 0.01; do {
for element in ${distributions[@]}; do {
for learning_rate in 0.05 0.1; do {

# lr-noniid-bs4
log_dir=${log_path}'/log_'$vid'/'${dataset}'/lr/noniid'${noniid_level}'/'${element}
echo $log_dir'-'$i

nohup python main.py --max_steps 10000 --dataset $dataset --model lr --lr $learning_rate --N $i --noniid True --noniid_level ${noniid_level} --client_dataset_size ${dataset_size} --num_microbatches ${batch_size} --client_batch_size ${batch_size} --sample_mode R --sample_ratio 0.8 --local_steps 100  --dpsgd True --eps $element --weiavg True --version $vid --save_dir ${save_dir} >${log_dir}/${i}_bs${batch_size}_nm${batch_size}_10000_100_R8_wavg_constlr${learning_rate} 2>&1 &

nohup python main.py --max_steps 10000 --dataset $dataset --model lr --lr $learning_rate --N $i --noniid True --noniid_level ${noniid_level} --client_dataset_size ${dataset_size} --num_microbatches ${batch_size} --client_batch_size ${batch_size} --sample_mode R --sample_ratio 0.8 --local_steps 100 --dpsgd True --eps $element --fedavg True --version $vid --save_dir ${save_dir} >${log_dir}/${i}_bs${batch_size}_nm${batch_size}_10000_100_R8_fedavg_constlr${learning_rate} 2>&1 &

# lr-iid
log_dir=${log_path}'/log_'$vid'/'${dataset}'/lr/iid/'${element}
echo $log_dir'-'$i

nohup python main.py --max_steps 10000 --dataset $dataset --model lr --lr $learning_rate --N $i --client_dataset_size ${dataset_size} --num_microbatches ${batch_size} --client_batch_size ${batch_size} --sample_mode R --sample_ratio 0.8 --local_steps 100  --dpsgd True --eps $element --weiavg True --version $vid --save_dir ${save_dir} >${log_dir}/${i}_bs${batch_size}_nm${batch_size}_10000_100_R8_wavg_constlr${learning_rate} 2>&1 &

nohup python main.py --max_steps 10000 --dataset $dataset --model lr --lr $learning_rate --N $i --client_dataset_size ${dataset_size} --num_microbatches ${batch_size} --client_batch_size ${batch_size} --sample_mode R --sample_ratio 0.8 --local_steps 100  --dpsgd True --eps $element --fedavg True --version $vid --save_dir ${save_dir} >${log_dir}/${i}_bs${batch_size}_nm${batch_size}_10000_100_R8_fedavg_constlr${learning_rate} 2>&1 &

} done 
wait
} done 
wait
} done 
wait
