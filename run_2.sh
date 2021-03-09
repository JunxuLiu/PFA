#!/bin/bash

vid=2
noniid_level=10
batch_size=4
num_microbatches=4

#distributions=('mixgauss1' 'mixgauss2' 'mixgauss4' 'mixgauss5' 'gauss1' 'pareto1' 'uniform1')
distributions=('mixgauss4' 'mixgauss5' 'gauss1' 'pareto1' 'uniform1')

#datasets=('mnist' 'fmnist')
datasets=('fmnist')


for dataset in ${datasets[@]}; do
for element in ${distributions[@]}; do
for ((i=10; i<=50; i=i+10)); do

#echo $vid,${log_dir},${dataset},${element},${i}     
:<<!
# lr-iid
log_dir='log_'$vid'/'${dataset}'/lr/iid/'${element}
echo $log_dir$i
nohup python main.py --global_steps 10000 --dataset $dataset --model lr --N $i --num_microbatches ${num_microbatches} --client_batch_size ${batch_size} --sample_mode R --sample_ratio 0.8 --local_steps 100 --dpsgd True --eps $element --projection True --proj_dims 1 --lanczos_iter 256 --version $vid >${log_dir}/${i}_bs${batch_size}_nm${num_microbatches}_10000_100_R8_pro1_256_constlr 2>&1 &

nohup python main.py --global_steps 10000 --dataset $dataset --model lr --N $i --num_microbatches ${num_microbatches} --client_batch_size ${batch_size} --sample_mode R --sample_ratio 0.8 --local_steps 100 --dpsgd True --eps $element --weiavg True --version $vid >${log_dir}/${i}_bs${batch_size}_nm${num_microbatches}_10000_100_R8_wavg_constlr 2>&1 &

nohup python main.py --global_steps 10000 --dataset $dataset --model lr --N $i --num_microbatches ${num_microbatches} --client_batch_size ${batch_size} --sample_mode R --sample_ratio 0.8 --local_steps 100 --dpsgd True --eps $element --fedavg True --version $vid >${log_dir}/${i}_bs${batch_size}_nm${num_microbatches}_10000_100_R8_fedavg_constlr 2>&1 &
!
:<<!
# lr-noniid-bs4
log_dir='log_'$vid'/'${dataset}'/lr/noniid'${noniid_level}'/'${element}
echo $log_dir$i
nohup python main.py --global_steps 10000 --dataset $dataset --model lr --N $i --noniid True --noniid_level ${noniid_level} --num_microbatches ${num_microbatches} --client_batch_size ${batch_size} --sample_mode R --sample_ratio 0.8 --local_steps 100 --dpsgd True --eps $element --projection True --proj_dims 1 --lanczos_iter 256 --version $vid >${log_dir}/${i}_bs${batch_size}_nm${num_microbatches}_10000_100_R8_pro1_256_constlr 2>&1 &

nohup python main.py --global_steps 10000 --dataset $dataset --model lr --N $i --noniid True --noniid_level ${noniid_level} --num_microbatches ${num_microbatches} --client_batch_size ${batch_size} --sample_mode R --sample_ratio 0.8 --local_steps 100 --dpsgd True --eps $element --weiavg True --version $vid >${log_dir}/${i}_bs${batch_size}_nm${num_microbatches}_10000_100_R8_wavg_constlr 2>&1 &

nohup python main.py --global_steps 10000 --dataset $dataset --model lr --N $i --noniid True --noniid_level ${noniid_level} --num_microbatches ${num_microbatches} --client_batch_size ${batch_size} --sample_mode R --sample_ratio 0.8 --local_steps 100 --dpsgd True --eps $element --fedavg True --version $vid >${log_dir}/${i}_bs${batch_size}_nm${num_microbatches}_10000_100_R8_fedavg_constlr 2>&1 &
!
:<<!
# cnn-iid
log_dir='log_'$vid'/'${dataset}'/cnn/iid/'${element}
echo $log_dir$i
nohup python main.py --global_steps 10000 --dataset $dataset --model cnn --N $i --num_microbatches ${num_microbatches} --client_batch_size ${batch_size} --sample_mode R --sample_ratio 0.8 --local_steps 100 --dpsgd True --eps $element --projection True --proj_dims 1 --lanczos_iter 256 --version $vid >${log_dir}/${i}_bs${batch_size}_nm${num_microbatches}_10000_100_R8_pro1_256_constlr 2>&1 &

nohup python main.py --global_steps 10000 --dataset $dataset --model cnn --N $i --num_microbatches ${num_microbatches} --client_batch_size ${batch_size} --sample_mode R --sample_ratio 0.8 --local_steps 100 --dpsgd True --eps $element --weiavg True --version $vid >${log_dir}/${i}_bs${batch_size}_nm${num_microbatches}_10000_100_R8_wavg_constlr 2>&1 &

nohup python main.py --global_steps 10000 --dataset $dataset --model cnn --N $i --num_microbatches ${num_microbatches} --client_batch_size ${batch_size} --sample_mode R --sample_ratio 0.8 --local_steps 100 --dpsgd True --eps $element --fedavg True --version $vid >${log_dir}/${i}_bs${batch_size}_nm${num_microbatches}_10000_100_R8_fedavg_constlr 2>&1 &
!

# cnn-noniid8
log_dir='log_'$vid'/'${dataset}'/lr/noniid'${noniid_level}'/'${element}
echo $log_dir$i
nohup python main.py --global_steps 10000 --dataset $dataset --model cnn --N $i --noniid True --noniid_level 8 --num_microbatches ${num_microbatches} --client_batch_size ${batch_size} --sample_mode R --sample_ratio 0.8 --local_steps 100 --dpsgd True --eps $element --projection True --proj_dims 1 --lanczos_iter 256 --version $vid >${log_dir}/${i}_bs${batch_size}_nm${num_microbatches}_10000_100_R8_pro1_256_constlr 2>&1 &

nohup python main.py --global_steps 10000 --dataset $dataset --model cnn --N $i --noniid True --noniid_level 8 --num_microbatches ${num_microbatches} --client_batch_size ${batch_size} --sample_mode R --sample_ratio 0.8 --local_steps 100 --dpsgd True --eps $element --weiavg True --version $vid >${log_dir}/${i}_bs${batch_size}_nm${num_microbatches}_10000_100_R8_wavg_constlr 2>&1 &

nohup python main.py --global_steps 10000 --dataset $dataset --model cnn --N $i --noniid True --noniid_level 8 --num_microbatches ${num_microbatches} --client_batch_size ${batch_size} --sample_mode R --sample_ratio 0.8 --local_steps 100 --dpsgd True --eps $element --fedavg True --version $vid >${log_dir}/${i}_bs${batch_size}_nm${num_microbatches}_10000_100_R8_fedavg_constlr 2>&1 &

done
done
done

:<<!
# iid setting: N=10
# [0.5+0.01, 10+0.1]: 1.0
# projection dims 1/2/5/10/20/50
for i in 1 2 3 5;
do
    nohup python main.py --global_steps 10000 --dataset mnist --N 10 --num_microbatches ${num_microbatches} --client_batch_size ${batch_size} --sample_mode R --sample_ratio 0.8 --local_steps 100 --model cnn --dpsgd True --eps mixgauss1 --projection True --proj_dims $i --lanczos_iter 256 >log_2/mnist/cnn/noniid10/10_bs${batch_size}_nm${num_microbatches}_10000_100_R8_mixgauss1_pro${i}_256_constlr 2>&1 &

    nohup python main.py --global_steps 10000 --dataset mnist --N 10 --num_microbatches ${num_microbatches} --client_batch_size ${batch_size} --sample_mode R --sample_ratio 0.8 --local_steps 100 --model cnn --dpsgd True --eps mixgauss1 --projection True --proj_dims $i --lanczos_iter 256 >log_2/mnist/cnn/noniid10/10_bs${batch_size}_nm${num_microbatches}_10000_100_R8_mixgauss1_pro${i}_256_constlr 2>&1 &

    nohup python main.py --global_steps 10000 --dataset mnist --N 10 --num_microbatches ${num_microbatches} --client_batch_size ${batch_size} --sample_mode R --sample_ratio 0.8 --local_steps 100 --model cnn --dpsgd True --eps mixgauss1 --projection True --proj_dims $i --lanczos_iter 256 >log_2/mnist/cnn/noniid10/10_bs${batch_size}_nm${num_microbatches}_10000_100_R8_mixgauss1_pro${i}_256_constlr 2>&1 &

done


# iid setting: N=30
# [0.5+0.01, 10+0.1]: 1.0
# lanczos iterations 64/128/256
# projection dims 1
for i in 32 64 128 256 512;
do
    nohup python main.py --global_steps 10000 --N 30 --num_microbatches ${num_microbatches} --client_batch_size ${batch_size} --sample_mode R --sample_ratio 0.8 --local_steps 100 --model lr --dpsgd True --eps mixgauss1 --projection True --proj_dims 1 --lanczos_iter $i >log_2/log_lr_iid_30_bs${batch_size}_nm${num_microbatches}_10000_100_R8_mixgauss1_pro1_${i}_constlr_0126_v6 2>&1 &
done
!
