#!/bin/bash

vid=2
noniid_level=10
batch_size=4

#distributions=('mixgauss1' 'mixgauss2' 'mixgauss4' 'mixgauss5' 'gauss1' 'pareto1' 'uniform1')
#distributions=('mixgauss4' 'mixgauss5' 'gauss1' 'pareto1' 'uniform1')
distributions=('mixgauss1' 'mixgauss2')
#distributions=('min1' 'min2' 'max1' 'max2')

#datasets=('mnist' 'fmnist')
datasets=('fmnist')

for dataset in ${datasets[@]}; do {
for ((i=10; i<=50; i=i+10)); do {
    
    # lr-iid
    log_dir='log/log_'$vid'/'${dataset}'/lr/iid/nodp'
    echo ${log_dir}'-'$i

    nohup python main.py --max_steps 10000 --dataset $dataset --model lr --N $i --num_microbatches ${batch_size} --client_batch_size ${batch_size} --sample_mode R --sample_ratio 0.8 --local_steps 100 --version $vid >${log_dir}/${i}_bs${batch_size}_nm${batch_size}_10000_100_R8_nodp_constlr 2>&1 &


    # lr-noniid10
    log_dir='log/log_'$vid'/'${dataset}'/lr/noniid'${noniid_level}'/nodp'
    echo ${log_dir}'-'$i

    nohup python main.py --max_steps 10000 --dataset $dataset --model lr --N $i --noniid True --noniid_level ${noniid_level}  --num_microbatches ${batch_size} --client_batch_size ${batch_size} --sample_mode R --sample_ratio 0.8 --local_steps 100 --version $vid >${log_dir}/${i}_bs${batch_size}_nm${batch_size}_10000_100_R8_nodp_constlr 2>&1 &

    # cnn-iid
    log_dir='log/log_'$vid'/'${dataset}'/cnn/iid/nodp'
    echo ${log_dir}'-'$i

    nohup python main.py --max_steps 10000 --dataset $dataset --model cnn --N $i --num_microbatches ${batch_size} --client_batch_size ${batch_size} --sample_mode R --sample_ratio 0.8 --local_steps 100 --version $vid >${log_dir}/${i}_bs${batch_size}_nm${batch_size}_10000_100_R8_nodp_constlr 2>&1 &

    # cnn-noniid10
    log_dir='log/log_'$vid'/'${dataset}'/cnn/noniid'${noniid_level}'/nodp'
    echo ${log_dir}'-'$i

    nohup python main.py --max_steps 10000 --dataset $dataset --model cnn --N $i --noniid True --noniid_level ${noniid_level} --num_microbatches ${batch_size} --client_batch_size ${batch_size} --sample_mode R --sample_ratio 0.8 --local_steps 100 --version $vid >${log_dir}/${i}_bs${batch_size}_nm${batch_size}_10000_100_R8_nodp_constlr 2>&1 &
    } done
    wait
} done
wait

# lr-noniid-bs4
log_dir='log/log_'$vid'/'${dataset}'/lr/noniid'${noniid_level}'/'${element}
echo $log_dir'-'$i
nohup python main.py --max_steps 10000 --dataset $dataset --model lr --N $i --noniid True --noniid_level ${noniid_level} --num_microbatches ${batch_size} --client_batch_size ${batch_size} --sample_mode R --sample_ratio 0.8 --local_steps 100  --dpsgd True --eps $element --projection True --proj_dims 1 --lanczos_iter 256 --version $vid >${log_dir}/${i}_bs${batch_size}_nm${batch_size}_10000_100_R8_pro1_256_constlr 2>&1 &

nohup python main.py --max_steps 10000 --dataset $dataset --model lr --N $i --noniid True --noniid_level ${noniid_level} --num_microbatches ${batch_size} --client_batch_size ${batch_size} --sample_mode R --sample_ratio 0.8 --local_steps 100  --dpsgd True --eps $element --weiavg True --version $vid >${log_dir}/${i}_bs${batch_size}_nm${batch_size}_10000_100_R8_wavg_constlr 2>&1 &

nohup python main.py --max_steps 10000 --dataset $dataset --model lr --N $i --noniid True --noniid_level ${noniid_level} --num_microbatches ${batch_size} --client_batch_size ${batch_size} --sample_mode R --sample_ratio 0.8 --local_steps 100  --dpsgd True --eps $element --fedavg True --version $vid >${log_dir}/${i}_bs${batch_size}_nm${batch_size}_10000_100_R8_fedavg_constlr 2>&1 &

} done 
wait
} done 
wait
} done 
wait


for element in ${distributions[@]}; do {
for dataset in ${datasets[@]}; do {
for ((i=10; i<=50; i=i+10)); do {

# cnn-noniid
log_dir='log/log_'$vid'/'${dataset}'/cnn/noniid'${noniid_level}'/'${element}
echo $log_dir'-'$i
nohup python main.py --max_steps 10000 --dataset $dataset --model cnn --N $i --noniid True --noniid_level ${noniid_level} --num_microbatches ${batch_size} --client_batch_size ${batch_size} --sample_mode R --sample_ratio 0.8 --local_steps 100  --dpsgd True --eps $element --projection True --proj_dims 1 --lanczos_iter 256 --version $vid >${log_dir}/${i}_bs${batch_size}_nm${batch_size}_10000_100_R8_pro1_256_constlr 2>&1 &

nohup python main.py --max_steps 10000 --dataset $dataset --model cnn --N $i --noniid True --noniid_level ${noniid_level} --num_microbatches ${batch_size} --client_batch_size ${batch_size} --sample_mode R --sample_ratio 0.8 --local_steps 100  --dpsgd True --eps $element --weiavg True --version $vid >${log_dir}/${i}_bs${batch_size}_nm${batch_size}_10000_100_R8_wavg_constlr 2>&1 &

nohup python main.py --max_steps 10000 --dataset $dataset --model cnn --N $i --noniid True --noniid_level ${noniid_level} --num_microbatches ${batch_size} --client_batch_size ${batch_size} --sample_mode R --sample_ratio 0.8 --local_steps 100  --dpsgd True --eps $element --fedavg True --version $vid >${log_dir}/${i}_bs${batch_size}_nm${batch_size}_10000_100_R8_fedavg_constlr 2>&1 &

} done 
wait
} done 
wait
} done 
wait

for element in ${distributions[@]}; do {
for dataset in ${datasets[@]}; do {
for ((i=10; i<=50; i=i+10)); do {

# lr-iid
log_dir='log/log_'$vid'/'${dataset}'/lr/iid/'${element}
echo $log_dir$i
nohup python main.py --max_steps 10000 --dataset $dataset --model lr --N $i --num_microbatches ${batch_size} --client_batch_size ${batch_size} --sample_mode R --sample_ratio 0.8 --local_steps 100  --dpsgd True --eps $element --projection True --proj_dims 1 --lanczos_iter 256 --version $vid >${log_dir}/${i}_bs${batch_size}_nm${batch_size}_10000_100_R8_pro1_256_constlr 2>&1 &

nohup python main.py --max_steps 10000 --dataset $dataset --model lr --N $i --num_microbatches ${batch_size} --client_batch_size ${batch_size} --sample_mode R --sample_ratio 0.8 --local_steps 100  --dpsgd True --eps $element --weiavg True --version $vid >${log_dir}/${i}_bs${batch_size}_nm${batch_size}_10000_100_R8_wavg_constlr 2>&1 &

nohup python main.py --max_steps 10000 --dataset $dataset --model lr --N $i --num_microbatches ${batch_size} --client_batch_size ${batch_size} --sample_mode R --sample_ratio 0.8 --local_steps 100  --dpsgd True --eps $element --fedavg True --version $vid >${log_dir}/${i}_bs${batch_size}_nm${batch_size}_10000_100_R8_fedavg_constlr 2>&1 &

} done 
wait
} done 
wait
} done 
wait

for element in ${distributions[@]}; do {
for dataset in ${datasets[@]}; do {
for ((i=10; i<=50; i=i+10)); do {

# cnn-iid
log_dir='log/log_'$vid'/'${dataset}'/cnn/iid/'${element}
echo $log_dir$i
nohup python main.py --max_steps 10000 --dataset $dataset --model cnn --N $i --num_microbatches ${batch_size} --client_batch_size ${batch_size} --sample_mode R --sample_ratio 0.8 --local_steps 100  --dpsgd True --eps $element --projection True --proj_dims 1 --lanczos_iter 256 --version $vid >${log_dir}/${i}_bs${batch_size}_nm${batch_size}_10000_100_R8_pro1_256_constlr 2>&1 &

nohup python main.py --max_steps 10000 --dataset $dataset --model cnn --N $i --num_microbatches ${batch_size} --client_batch_size ${batch_size} --sample_mode R --sample_ratio 0.8 --local_steps 100  --dpsgd True --eps $element --weiavg True --version $vid >${log_dir}/${i}_bs${batch_size}_nm${batch_size}_10000_100_R8_wavg_constlr 2>&1 &

nohup python main.py --max_steps 10000 --dataset $dataset --model cnn --N $i --num_microbatches ${batch_size} --client_batch_size ${batch_size} --sample_mode R --sample_ratio 0.8 --local_steps 100  --dpsgd True --eps $element --fedavg True --version $vid >${log_dir}/${i}_bs${batch_size}_nm${batch_size}_10000_100_R8_fedavg_constlr 2>&1 &

} done 
wait
} done 
wait
} done 
wait

