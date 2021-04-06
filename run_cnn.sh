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

########## cnn wavg/fedavg ###########

for ((i=10; i<=50; i=i+10)); do {
for learning_rate in 0.001 0.005 0.01; do {
for element in ${distributions[@]}; do {


# cnn-noniid-bs4
log_dir=${log_path}'/log_'$vid'/'${dataset}'/cnn/noniid'${noniid_level}'/'${element}
echo $log_dir'-'$i

nohup python main.py --max_steps 10000 --dataset $dataset --model cnn --lr $learning_rate --N $i --noniid True --noniid_level ${noniid_level} --client_dataset_size ${dataset_size} --num_microbatches ${batch_size} --client_batch_size ${batch_size} --sample_mode R --sample_ratio 0.8 --local_steps 100  --dpsgd True --eps $element --weiavg True --version $vid --save_dir ${save_dir} >${log_dir}/${i}_bs${batch_size}_nm${batch_size}_10000_100_R8_wavg_constlr${learning_rate} 2>&1 &

nohup python main.py --max_steps 10000 --dataset $dataset --model cnn --lr $learning_rate --N $i --noniid True --noniid_level ${noniid_level} --client_dataset_size ${dataset_size} --num_microbatches ${batch_size} --client_batch_size ${batch_size} --sample_mode R --sample_ratio 0.8 --local_steps 100 --dpsgd True --eps $element --fedavg True --version $vid --save_dir ${save_dir} >${log_dir}/${i}_bs${batch_size}_nm${batch_size}_10000_100_R8_fedavg_constlr${learning_rate} 2>&1 &

# cnn-iid
log_dir=${log_path}'/log_'$vid'/'${dataset}'/cnn/iid/'${element}
echo $log_dir'-'$i

nohup python main.py --max_steps 10000 --dataset $dataset --model cnn --lr $learning_rate --N $i --client_dataset_size ${dataset_size} --num_microbatches ${batch_size} --client_batch_size ${batch_size} --sample_mode R --sample_ratio 0.8 --local_steps 100  --dpsgd True --eps $element --weiavg True --version $vid --save_dir ${save_dir} >${log_dir}/${i}_bs${batch_size}_nm${batch_size}_10000_100_R8_wavg_constlr${learning_rate} 2>&1 &

nohup python main.py --max_steps 10000 --dataset $dataset --model cnn --lr $learning_rate --N $i --client_dataset_size ${dataset_size} --num_microbatches ${batch_size} --client_batch_size ${batch_size} --sample_mode R --sample_ratio 0.8 --local_steps 100  --dpsgd True --eps $element --fedavg True --version $vid --save_dir ${save_dir} >${log_dir}/${i}_bs${batch_size}_nm${batch_size}_10000_100_R8_fedavg_constlr${learning_rate} 2>&1 &

} done 
wait
} done 
wait
} done 
wait

:<<!
python main.py --max_steps 10000 --dataset mnist --model cnn --lr 0.001 --N 10 --num_microbatches 4 --client_batch_size 4 --sample_mode R --sample_ratio 0.8 --local_steps 100  --dpsgd True --eps mixgauss1 --fedavg True --version 2 --save_dir res --client_dataset_size 600

python main.py --max_steps 10000 --dataset mnist --model cnn --lr 0.001 --N 10 --num_microbatches 4 --client_batch_size 4 --sample_mode R --sample_ratio 0.8 --local_steps 100  --dpsgd True --eps mixgauss1 --weiavg True --version 2 --save_dir res --client_dataset_size 600

python main.py --max_steps 10000 --dataset mnist --model cnn --lr 0.005 --N 10 --num_microbatches 4 --client_batch_size 4 --sample_mode R --sample_ratio 0.8 --local_steps 100  --dpsgd True --eps mixgauss1 --projection True --version 2 --save_dir res --client_dataset_size 600

python main.py --max_steps 10000 --dataset mnist --model cnn --lr 0.01 --N 10 --noniid True --noniid_level 10 --num_microbatches 4 --client_batch_size 4 --sample_mode R --sample_ratio 0.8 --local_steps 100  --dpsgd True --eps mixgauss3 --fedavg True --version 2 --save_dir res --client_dataset_size 600

python main.py --max_steps 10000 --dataset mnist --model cnn --lr 0.005 --N 10 --noniid True --noniid_level 10 --num_microbatches 4 --client_batch_size 4 --sample_mode R --sample_ratio 0.8 --local_steps 100  --dpsgd True --eps mixgauss3 --weiavg True --version 2 --save_dir res --client_dataset_size 600

python main.py --max_steps 10000 --dataset mnist --model cnn --lr 0.05 --N 10 --noniid True --noniid_level 10 --num_microbatches 4 --client_batch_size 4 --sample_mode R --sample_ratio 0.8 --local_steps 100  --dpsgd True --eps mixgauss3 --projection True --proj_dims 1 --version 2 --save_dir res --client_dataset_size 600


==========

python main.py --max_steps 10000 --dataset mnist --model cnn --lr 0.01 --N 20 --num_microbatches 4 --client_batch_size 4 --sample_mode R --sample_ratio 0.8 --local_steps 100  --dpsgd True --eps mixgauss1 --fedavg True --version 2 --save_dir res_1200 --client_dataset_size 1200

python main.py --max_steps 10000 --dataset mnist --model cnn --lr 0.005 --N 10 --num_microbatches 4 --client_batch_size 4 --sample_mode R --sample_ratio 0.8 --local_steps 100  --dpsgd True --eps mixgauss1 --weiavg True --version 2 --save_dir res_600 --client_dataset_size 600

python main.py --max_steps 10000 --dataset mnist --model cnn --lr 0.1 --N 10 --num_microbatches 4 --client_batch_size 4 --sample_mode R --sample_ratio 0.8 --local_steps 100  --dpsgd True --eps mixgauss1 --projection True --proj_dims 1 --version 2 --save_dir res_600 --client_dataset_size 600
!