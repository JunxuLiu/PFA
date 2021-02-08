#!/bin/bash

# iid setting: N=10/20/30/40/50
# [0.5+0.01, 10+0.1]: 1.0

# iid setting: N=10/20/30/40/50
# [0.5+0.01, 10.0+0.1] 1.0
# projection dims 1
:'
for ((i=10; i<=50; i=i+10));
do
    nohup python main_v6.py --max_steps 10000 --N $i --num_microbatches 128 --client_batch_size 128 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model lr --dpsgd True --eps epsilons1 --projection True --proj_dims 1 --lanczos_iter 256 >log_2/log_lr_iid_${i}_bs128_nm128_10000_100_R8_dp1_pro1_256_constlr_0126_v6 2>&1 &

    nohup python main_v6.py --max_steps 10000 --N $i --num_microbatches 128 --client_batch_size 128 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model lr --dpsgd True --eps epsilons1 --wei_avg True >log_2/log_lr_iid_${i}_bs128_nm128_10000_100_R8_dp1_wavg_constlr_0126_v6 2>&1 &

    nohup python main_v6.py --max_steps 10000 --N $i --num_microbatches 128 --client_batch_size 128 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model lr --dpsgd True --eps epsilons1 --fedavg True >log_2/log_lr_iid_${i}_bs128_nm128_10000_100_R8_dp1_fedavg_constlr_0126_v6 2>&1 &

done


# iid setting: N=10/20/30/40/50
# [1.0+0.1, 10.0+0.1] 2.0
# projection dims 1
for ((i=10; i<=50; i=i+10));
do
    nohup python main_v6.py --max_steps 10000 --N $i --num_microbatches 128 --client_batch_size 128 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model lr --dpsgd True --eps epsilons2 --projection True --proj_dims 1 --lanczos_iter 256 >log_2/log_lr_iid_${i}_bs128_nm128_10000_100_R8_dp2_pro1_256_constlr_0126_v6 2>&1 &

    nohup python main_v6.py --max_steps 10000 --N $i --num_microbatches 128 --client_batch_size 128 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model lr --dpsgd True --eps epsilons2 --wei_avg True >log_2/log_lr_iid_${i}_bs128_nm128_10000_100_R8_dp2_wavg_constlr_0126_v6 2>&1 &

    nohup python main_v6.py --max_steps 10000 --N $i --num_microbatches 128 --client_batch_size 128 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model lr --dpsgd True --eps epsilons2 --fedavg True >log_2/log_lr_iid_${i}_bs128_nm128_10000_100_R8_dp2_fedavg_constlr_0126_v6 2>&1 &

done
'
:'
# iid setting: N=30
# [0.5+0.01, 10+0.1]: 1.0
# projection dims 1/2/5/10/20/50
for i in 2 5 10 20 50;
do
    nohup python main_v6.py --max_steps 10000 --N 30 --num_microbatches 128 --client_batch_size 128 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model lr --dpsgd True --eps epsilons1 --projection True --proj_dims $i --lanczos_iter 256 >log_2/log_lr_iid_30_bs128_nm128_10000_100_R8_dp1_pro${i}_256_constlr_0126_v6 2>&1 &
done


# iid setting: N=30
# [0.5+0.01, 10+0.1]: 1.0
# lanczos iterations 64/128/256
# projection dims 1
for i in 32 64 128 256 512;
do
    nohup python main_v6.py --max_steps 10000 --N 30 --num_microbatches 128 --client_batch_size 128 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model lr --dpsgd True --eps epsilons1 --projection True --proj_dims 1 --lanczos_iter $i >log_2/log_lr_iid_30_bs128_nm128_10000_100_R8_dp1_pro1_${i}_constlr_0126_v6 2>&1 &
done
'
:'
# fmnist
for ((i=10; i<=50; i=i+10));
do
    #nohup python main_v6.py --max_steps 10000 --N $i --dataset fmnist --num_microbatches 128 --client_batch_size 128 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model lr --dpsgd True --eps epsilons1 --projection True --proj_dims 1 --lanczos_iter 256 --save_dir res_2 >log_2/log_lr_fmnist_iid_${i}_bs128_nm128_10000_100_R8_dp1_pro1_256_constlr_0128_v6 2>&1 &

    #nohup python main_v6.py --max_steps 10000 --N $i --dataset fmnist --num_microbatches 128 --client_batch_size 128 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model lr --dpsgd True --eps epsilons1 --wei_avg True --save_dir res_2 >log_2/log_lr_fmnist_iid_${i}_bs128_nm128_10000_100_R8_dp1_wavg_constlr_0128_v6 2>&1 &

    #nohup python main_v6.py --max_steps 10000 --N $i --dataset fmnist --num_microbatches 128 --client_batch_size 128 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model lr --dpsgd True --eps epsilons1 --fedavg True --save_dir res_2 >log_2/log_lr_fmnist_iid_${i}_bs128_nm128_10000_100_R8_dp1_fedavg_constlr_0128_v6 2>&1 &

    #nohup python main_v6.py --max_steps 10000 --N $i --dataset fmnist --num_microbatches 128 --client_batch_size 128 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model lr --version 2 >log_2/fmnist/lr/iid/nodp/${i}_bs128_nm128_10000_100_R8_nodp_constlr 2>&1 &
done

# iid setting: N=10/20/30/40/50
# [1.0+0.1, 10.0+0.1] 2.0
# projection dims 1
for ((i=10; i<=50; i=i+10));
do
    #nohup python main_v6.py --max_steps 10000 --N $i --dataset fmnist --num_microbatches 128 --client_batch_size 128 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model lr --dpsgd True --eps epsilons2 --projection True --proj_dims 1 --lanczos_iter 256 --save_dir res_2 >log_2/log_lr_fmnist_iid_${i}_bs128_nm128_10000_100_R8_dp2_pro1_256_constlr_0128_v6 2>&1 &

    #nohup python main_v6.py --max_steps 10000 --N $i --dataset fmnist --num_microbatches 128 --client_batch_size 128 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model lr --dpsgd True --eps epsilons2 --wei_avg True --save_dir res_2 >log_2/log_lr_fmnist_iid_${i}_bs128_nm128_10000_100_R8_dp2_wavg_constlr_0128_v6 2>&1 &

    #nohup python main_v6.py --max_steps 10000 --N $i --dataset fmnist --num_microbatches 128 --client_batch_size 128 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model lr --dpsgd True --eps epsilons2 --fedavg True --save_dir res_2 >log_2/log_lr_fmnist_iid_${i}_bs128_nm128_10000_100_R8_dp2_fedavg_constlr_0128_v6 2>&1 &

done
'
nohup python main_v6.py --max_steps 10000 --N 10 --dataset fmnist --num_microbatches 128 --client_batch_size 128 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model lr --dpsgd True --eps epsilons1 --projection True --proj_dims 1 --lanczos_iter 256 --version 2 >log_2/fmnist/lr/iid/dp1/10_bs128_nm128_10000_100_R8_dp1_constlr 2>&1 &

