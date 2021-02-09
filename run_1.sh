#!/bin/bash

# iid setting: N=10/20/30/40/50
# [0.5+0.01, 10+0.1]: 1.0
:'
for ((i=10; i<=50; i=i+10));
do
    #nohup python main_v6.py --max_steps 10000 --N $i --num_microbatches 128 --client_batch_size 128 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model lr --dpsgd True --eps epsilons1 --projection True --proj_dims 1 --lanczos_iter 256 >log_1/log_lr_iid_${i}_bs128_nm128_10000_100_R8_dp1_pro1_256_constlr_0124_v6 2>&1 &

    #nohup python main_v6.py --max_steps 10000 --N $i --num_microbatches 128 --client_batch_size 128 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model lr --dpsgd True --eps epsilons1 --wei_avg True >log_1/log_lr_iid_${i}_bs128_nm128_10000_100_R8_dp1_wavg_constlr_0125_v6 2>&1 &

    #nohup python main_v6.py --max_steps 10000 --N $i --num_microbatches 128 --client_batch_size 128 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model lr --dpsgd True --eps epsilons1 --fedavg True >log_1/log_lr_iid_${i}_bs128_nm128_10000_100_R8_dp1_fedavg_constlr_0125_v6 2>&1 &

    nohup python main_v6.py --max_steps 10000 --N $i --noniid True --num_microbatches 128 --client_batch_size 128 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model lr --dpsgd True --eps epsilons1 --projection True --proj_dims 1 --lanczos_iter 256 >log_1/log_lr_noniid_${i}_bs128_nm128_10000_100_R8_dp1_pro1_256_constlr_0125_v6 2>&1 &

    nohup python main_v6.py --max_steps 10000 --N $i --noniid True --num_microbatches 128 --client_batch_size 128 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model lr --dpsgd True --eps epsilons1 --wei_avg True >log_1/log_lr_noniid_${i}_bs128_nm128_10000_100_R8_dp1_wavg_constlr_0125_v6 2>&1 &

    nohup python main_v6.py --max_steps 10000 --N $i --noniid True --num_microbatches 128 --client_batch_size 128 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model lr --dpsgd True --eps epsilons1 --fedavg True >log_1/log_lr_noniid_${i}_bs128_nm128_10000_100_R8_dp1_fedavg_constlr_0125_v6 2>&1 &
done


# iid setting: N=10/20/30/40/50
# [1.0+0.1, 10.0+0.1] 2.0
# projection dims 1
for ((i=10; i<=50; i=i+10));
do
    #nohup python main_v6.py --max_steps 10000 --N $i --num_microbatches 128 --client_batch_size 128 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model lr --dpsgd True --eps epsilons2 --projection True --proj_dims 1 --lanczos_iter 256 >log_1/log_lr_iid_${i}_bs128_nm128_10000_100_R8_dp2_pro1_256_constlr_0124_v6 2>&1 &

    #nohup python main_v6.py --max_steps 10000 --N $i --num_microbatches 128 --client_batch_size 128 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model lr --dpsgd True --eps epsilons2 --wei_avg True >log_1/log_lr_iid_${i}_bs128_nm128_10000_100_R8_dp2_wavg_constlr_0125_v6 2>&1 &

    #nohup python main_v6.py --max_steps 10000 --N $i --num_microbatches 128 --client_batch_size 128 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model lr --dpsgd True --eps epsilons2 --fedavg True >log_1/log_lr_iid_${i}_bs128_nm128_10000_100_R8_dp2_fedavg_constlr_0125_v6 2>&1 &

    nohup python main_v6.py --max_steps 10000 --N $i --noniid True --num_microbatches 128 --client_batch_size 128 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model lr --dpsgd True --eps epsilons2 --projection True --proj_dims 1 --lanczos_iter 256 >log_1/log_lr_noniid_${i}_bs128_nm128_10000_100_R8_dp2_pro1_256_constlr_0125_v6 2>&1 &

    nohup python main_v6.py --max_steps 10000 --N $i --noniid True --num_microbatches 128 --client_batch_size 128 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model lr --dpsgd True --eps epsilons2 --wei_avg True >log_1/log_lr_noniid_${i}_bs128_nm128_10000_100_R8_dp2_wavg_constlr_0125_v6 2>&1 &

    nohup python main_v6.py --max_steps 10000 --N $i --noniid True --num_microbatches 128 --client_batch_size 128 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model lr --dpsgd True --eps epsilons2 --fedavg True >log_1/log_lr_noniid_${i}_bs128_nm128_10000_100_R8_dp2_fedavg_constlr_0125_v6 2>&1 &
done
'
:'
# iid setting: N=30
# [0.5+0.01, 10+0.1]: 1.0
# projection dims 1/2/5/10/20/50
for i in 2 5 10 20 50;
do
    nohup python main_v6.py --max_steps 10000 --N 30 --num_microbatches 128 --client_batch_size 128 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model lr --dpsgd True --eps epsilons1 --projection True --proj_dims $i --lanczos_iter 256 >log_1/log_lr_iid_30_bs128_nm128_10000_100_R8_dp1_pro${i}_256_constlr_0124_v6 2>&1 &
done


# iid setting: N=10/20/30/40/50
# [0.5+0.01, 10+0.1]: 1.0
# lanczos iterations 64/128/256
# projection dims 1
for i in 32 64 128 256 512;
do
    nohup python main_v6.py --max_steps 10000 --N 30 --num_microbatches 128 --client_batch_size 128 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model lr --dpsgd True --eps epsilons1 --projection True --proj_dims 1 --lanczos_iter $i >log_1/log_lr_iid_30_bs128_nm128_10000_100_R8_dp1_pro1_${i}_constlr_0124_v6 2>&1 &
done
'
:'
# fmnist
# epsilons1
for ((i=10; i<=50; i=i+10));
do
    nohup python main_v6.py --max_steps 10000 --N $i --dataset fmnist --num_microbatches 128 --client_batch_size 128 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model lr --dpsgd True --eps epsilons1 --projection True --proj_dims 1 --lanczos_iter 256 --version 1 >log_1/fmnist/lr/iid/dp1/${i}_bs128_nm128_10000_100_R8_pro1_256_constlr 2>&1 &

    nohup python main_v6.py --max_steps 10000 --N $i --dataset fmnist --num_microbatches 128 --client_batch_size 128 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model lr --dpsgd True --eps epsilons1 --wei_avg True --version 1 >log_1/fmnist/lr/iid/dp1/${i}_bs128_nm128_10000_100_R8_wavg_constlr 2>&1 &

    nohup python main_v6.py --max_steps 10000 --N $i --dataset fmnist --num_microbatches 128 --client_batch_size 128 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model lr --dpsgd True --eps epsilons1 --fedavg True --version 1 >log_1/fmnist/lr/iid/dp1/${i}_bs128_nm128_10000_100_R8_fedavg_constlr 2>&1 &

    #nohup python main_v6.py --max_steps 10000 --N $i --dataset fmnist --num_microbatches 128 --client_batch_size 128 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model lr --version 1 >log_1/fmnist/lr/iid/nodp/${i}_bs128_nm128_10000_100_R8_constlr 2>&1 &
done

# fmnist
# epsilons2
for ((i=10; i<=50; i=i+10));
do
    nohup python main_v6.py --max_steps 10000 --N $i --dataset fmnist --num_microbatches 128 --client_batch_size 128 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model lr --dpsgd True --eps epsilons2 --projection True --proj_dims 1 --lanczos_iter 256 --version 1 >log_1/fmnist/lr/iid/dp2/${i}_bs128_nm128_10000_100_R8_pro1_256_constlr 2>&1 &

    nohup python main_v6.py --max_steps 10000 --N $i --dataset fmnist --num_microbatches 128 --client_batch_size 128 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model lr --dpsgd True --eps epsilons2 --wei_avg True --version 1 >log_1/fmnist/lr/iid/dp2/${i}_bs128_nm128_10000_100_R8_wavg_constlr 2>&1 &

    nohup python main_v6.py --max_steps 10000 --N $i --dataset fmnist --num_microbatches 128 --client_batch_size 128 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model lr --dpsgd True --eps epsilons2 --fedavg True --version 1 >log_1/fmnist/lr/iid/dp2/${i}_bs128_nm128_10000_100_R8_fedavg_constlr 2>&1 &

done
'


#-------------------------

# iid setting: N=10/20/30/40/50
# [0.5+0.01, 10+0.1]: 1.0
:'
for ((i=10; i<=50; i=i+10));
do
    nohup python main_v6.py --max_steps 10000 --N $i --num_microbatches 128 --client_batch_size 128 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model cnn --dpsgd True --eps epsilons1 --projection True --proj_dims 1 --lanczos_iter 256 --version 1 >log_1/mnist/cnn/iid/dp1/${i}_bs128_nm128_10000_100_R8_pro1_256_constlr 2>&1 &

    nohup python main_v6.py --max_steps 10000 --N $i --num_microbatches 128 --client_batch_size 128 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model cnn --dpsgd True --eps epsilons1 --wei_avg True --version 1 >log_1/mnist/cnn/iid/dp1/${i}_bs128_nm128_10000_100_R8_wavg_constlr 2>&1 &

    nohup python main_v6.py --max_steps 10000 --N $i --num_microbatches 128 --client_batch_size 128 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model cnn --dpsgd True --eps epsilons1 --fedavg True --version 1 >log_1/mnist/cnn/iid/dp1/${i}_bs128_nm128_10000_100_R8_fedavg_constlr 2>&1 &

done
'
:'
for ((i=10; i<=50; i=i+10));
do
    nohup python main_v6.py --max_steps 5000 --N $i --num_microbatches 16 --client_batch_size 16 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model cnn --dpsgd True --eps epsilons1 --projection True --proj_dims 1 --lanczos_iter 128 --version 1 >log_1/mnist/cnn/iid/dp1/${i}_bs16_nm16_5000_100_R8_pro1_128_constlr 2>&1 &

    nohup python main_v6.py --max_steps 5000 --N $i --num_microbatches 16 --client_batch_size 16 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model cnn --dpsgd True --eps epsilons1 --wei_avg True --version 1 >log_1/mnist/cnn/iid/dp1/${i}_bs16_nm16_5000_100_R8_wavg_constlr 2>&1 &

    nohup python main_v6.py --max_steps 5000 --N $i --num_microbatches 16 --client_batch_size 16 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model cnn --dpsgd True --eps epsilons1 --fedavg True --version 1 >log_1/mnist/cnn/iid/dp1/${i}_bs16_nm16_5000_100_R8_fedavg_constlr 2>&1 &

    nohup python main_v6.py --max_steps 5000 --N $i --num_microbatches 16 --client_batch_size 16 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model cnn --version 1 >log_1/mnist/cnn/iid/nodp/${i}_bs16_nm16_5000_100_R8_constlr 2>&1 &
done
'

for i in 32 64 128;
do
    nohup python main_v6.py --max_steps 5000 --N 10 --num_microbatches $i --client_batch_size $i --sample_mode R --sample_ratio 0.8 --local_steps 100 --model cnn --dpsgd True --eps epsilons1 --projection True --proj_dims 1 --lanczos_iter 128 --version 1 >log_1/mnist/cnn/iid/dp1/10_bs${i}_nm${i}_5000_100_R8_pro1_128_constlr 2>&1 &
done

for i in 32 64 128;
do
    nohup python main_v6.py --max_steps 5000 --N 10 --num_microbatches 16 --client_batch_size $i --sample_mode R --sample_ratio 0.8 --local_steps 100 --model cnn --dpsgd True --eps epsilons1 --projection True --proj_dims 1 --lanczos_iter 128 --version 1 >log_1/mnist/cnn/iid/dp1/10_bs${i}_nm16_5000_100_R8_pro1_128_constlr 2>&1 &
done




