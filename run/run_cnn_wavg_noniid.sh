#!/bin/bash
vid=2
noniid_level=10
local_steps=100
batch_size=4
dataset_size=1200
log_path='../PFA_res/log_wproj_'${dataset_size}
save_dir='../PFA_res/res_wproj_'${dataset_size}

element='mixgauss3'
dataset='mnist'

########## cnn wavg/fedavg ###########

for noniid_level in 8 6 4 2; do {
for i in 20 30 40 50; do {
for learning_rate in 0.001 0.005 0.01 0.025 0.05; do {

# cnn-noniid
log_dir=${log_path}'/log_'$vid'/'${dataset}'/cnn/noniid'${noniid_level}
if [ ! -d $log_dir ]
then
mkdir $log_dir
fi

log_dir=${log_path}'/log_'$vid'/'${dataset}'/cnn/noniid'${noniid_level}'/'${element}
if [ ! -d $log_dir ]
then
mkdir $log_dir
fi
echo $log_dir'-'$i

nohup python main.py --max_steps 10000 --dataset $dataset --model cnn --lr $learning_rate --N $i --client_dataset_size ${dataset_size} --noniid True --noniid_level ${noniid_level} --num_microbatches ${batch_size} --client_batch_size ${batch_size} --sample_mode R --sample_ratio 0.8 --local_steps ${local_steps}  --dpsgd True --eps $element --weiavg True --version $vid --save_dir ${save_dir} >${log_dir}/${i}_bs${batch_size}_nm${batch_size}_10000_${local_steps}_R8_wavg_constlr${learning_rate} 2>&1 &

} done 
wait
} done 
wait
} done 
wait