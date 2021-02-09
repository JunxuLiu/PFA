#!/bin/bash
:'
# iid setting: N=10/20/30/40/50
# [0.5+0.01, 10+0.1]: 1.0

# iid setting: N=10/20/30/40/50
# [0.5+0.01, 10.0+0.1] 1.0
# projection dims 1

for ((i=10; i<=50; i=i+10));
do
    nohup python main_v6.py --max_steps 10000 --N $i --num_microbatches 128 --client_batch_size 128 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model lr --dpsgd True --eps epsilons1 --projection True --proj_dims 1 --lanczos_iter 256 --save_dir res_3 >log_3/log_lr_iid_${i}_bs128_nm128_10000_100_R8_dp1_pro1_256_constlr_0131_v6 2>&1 &

    nohup python main_v6.py --max_steps 10000 --N $i --num_microbatches 128 --client_batch_size 128 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model lr --dpsgd True --eps epsilons1 --wei_avg True --save_dir res_3 >log_3/log_lr_iid_${i}_bs128_nm128_10000_100_R8_dp1_wavg_constlr_0131_v6 2>&1 &

    nohup python main_v6.py --max_steps 10000 --N $i --num_microbatches 128 --client_batch_size 128 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model lr --dpsgd True --eps epsilons1 --fedavg True --save_dir res_3 >log_3/log_lr_iid_${i}_bs128_nm128_10000_100_R8_dp1_fedavg_constlr_0131_v6 2>&1 &

    nohup python main_v6.py --max_steps 10000 --N $i --noniid True --num_microbatches 128 --client_batch_size 128 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model lr --dpsgd True --eps epsilons1 --projection True --proj_dims 1 --lanczos_iter 256 --save_dir res_3 >log_3/log_lr_noniid_${i}_bs128_nm128_10000_100_R8_dp1_pro1_256_constlr_0131_v6 2>&1 &

    nohup python main_v6.py --max_steps 10000 --N $i --noniid True --num_microbatches 128 --client_batch_size 128 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model lr --dpsgd True --eps epsilons1 --wei_avg True --save_dir res_3 >log_3/log_lr_noniid_${i}_bs128_nm128_10000_100_R8_dp1_wavg_constlr_0131_v6 2>&1 &

    nohup python main_v6.py --max_steps 10000 --N $i --noniid True --num_microbatches 128 --client_batch_size 128 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model lr --dpsgd True --eps epsilons1 --fedavg True --save_dir res_3 >log_3/log_lr_noniid_${i}_bs128_nm128_10000_100_R8_dp1_fedavg_constlr_0131_v6 2>&1 &

done


# iid setting: N=10/20/30/40/50
# [1.0+0.1, 10.0+0.1] 2.0
# projection dims 1
for ((i=10; i<=50; i=i+10));
do
    nohup python main_v6.py --max_steps 10000 --N $i --num_microbatches 128 --client_batch_size 128 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model lr --dpsgd True --eps epsilons2 --projection True --proj_dims 1 --lanczos_iter 256 --save_dir res_3 >log_3/log_lr_iid_${i}_bs128_nm128_10000_100_R8_dp2_pro1_256_constlr_0131_v6 2>&1 &

    nohup python main_v6.py --max_steps 10000 --N $i --num_microbatches 128 --client_batch_size 128 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model lr --dpsgd True --eps epsilons2 --wei_avg True --save_dir res_3 >log_3/log_lr_iid_${i}_bs128_nm128_10000_100_R8_dp2_wavg_constlr_0131_v6 2>&1 &

    nohup python main_v6.py --max_steps 10000 --N $i --num_microbatches 128 --client_batch_size 128 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model lr --dpsgd True --eps epsilons2 --fedavg True --save_dir res_3 >log_3/log_lr_iid_${i}_bs128_nm128_10000_100_R8_dp2_fedavg_constlr_0131_v6 2>&1 &

    nohup python main_v6.py --max_steps 10000 --N $i --noniid True --num_microbatches 128 --client_batch_size 128 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model lr --dpsgd True --eps epsilons2 --projection True --proj_dims 1 --lanczos_iter 256 --save_dir res_3 >log_3/log_lr_noniid_${i}_bs128_nm128_10000_100_R8_dp2_pro1_256_constlr_0131_v6 2>&1 &

    nohup python main_v6.py --max_steps 10000 --N $i --noniid True --num_microbatches 128 --client_batch_size 128 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model lr --dpsgd True --eps epsilons2 --wei_avg True --save_dir res_3 >log_3/log_lr_noniid_${i}_bs128_nm128_10000_100_R8_dp2_wavg_constlr_0131_v6 2>&1 &

    nohup python main_v6.py --max_steps 10000 --N $i --noniid True --num_microbatches 128 --client_batch_size 128 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model lr --dpsgd True --eps epsilons2 --fedavg True --save_dir res_3 >log_3/log_lr_noniid_${i}_bs128_nm128_10000_100_R8_dp2_fedavg_constlr_0131_v6 2>&1 &

done
'

# iid setting: N=10/20/30/40/50
# epsilons: 3/4/uniform
# projection dims 1
:'
for ((i=10; i<=50; i=i+10));
do
    #nohup python main_v6.py --max_steps 10000 --N $i --num_microbatches 128 --client_batch_size 128 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model lr --dpsgd True --eps epsilons5 --projection True --proj_dims 1 --lanczos_iter 256 --save_dir res_3 >log_3/mnist/lr/iid/log_${i}_bs128_nm128_10000_100_R8_dp5_pro1_256 2>&1 &

    #nohup python main_v6.py --max_steps 10000 --N $i --num_microbatches 128 --client_batch_size 128 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model lr --dpsgd True --eps epsilons5 --wei_avg True --save_dir res_3 >log_3/mnist/lr/iid/log_${i}_bs128_nm128_10000_100_R8_dp5_wavg 2>&1 &

    #nohup python main_v6.py --max_steps 10000 --N $i --num_microbatches 128 --client_batch_size 128 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model lr --dpsgd True --eps epsilons5 --fedavg True --save_dir res_3 >log_3/mnist/lr/iid/log_${i}_bs128_nm128_10000_100_R8_dp5_fedavg 2>&1 &

    #nohup python main_v6.py --max_steps 10000 --N $i --num_microbatches 128 --client_batch_size 128 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model lr --dpsgd True --eps epsilons4 --projection True --proj_dims 1 --lanczos_iter 256 --save_dir res_3 >log_3/mnist/lr/iid/log_${i}_bs128_nm128_10000_100_R8_dp4_pro1_256_constlr_0131_v6 2>&1 &

    #nohup python main_v6.py --max_steps 10000 --N $i --num_microbatches 128 --client_batch_size 128 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model lr --dpsgd True --eps epsilons4 --wavg True --save_dir res_3 >log_3/mnist/lr/iid/log_${i}_bs128_nm128_10000_100_R8_dp4_wavg 2>&1 &

    #nohup python main_v6.py --max_steps 10000 --N $i --num_microbatches 128 --client_batch_size 128 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model lr --dpsgd True --eps epsilons4 --fedavg True --save_dir res_3 >log_3/mnist/lr/iid/log_${i}_bs128_nm128_10000_100_R8_dp4_fedavg 2>&1 &


    #nohup python main_v6.py --max_steps 10000 --N $i --num_microbatches 128 --client_batch_size 128 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model lr --dpsgd True --eps epsilonsu --projection True --proj_dims 1 --lanczos_iter 256 --save_dir res_3 >log_3/mnist/lr/iid/log_${i}_bs128_nm128_10000_100_R8_dpuni_pro1_256_constlr_0131_v6 2>&1 &
    
    #nohup python main_v6.py --max_steps 10000 --N $i --num_microbatches 128 --client_batch_size 128 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model lr --dpsgd True --eps epsilonsu --wei_avg True --save_dir res_3 >log_3/mnist/lr/iid/log_${i}_bs128_nm128_10000_100_R8_dpuni_wavg 2>&1 &

    #nohup python main_v6.py --max_steps 10000 --N $i --num_microbatches 128 --client_batch_size 128 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model lr --dpsgd True --eps epsilonsu --fedavg True --save_dir res_3 >log_3/mnist/lr/iid/log_${i}_bs128_nm128_10000_100_R8_dpuni_fedavg 2>&1 &

done
'
:'
nohup python main_v6.py --max_steps 10000 --N 10 --num_microbatches 128 --client_batch_size 128 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model lr --dpsgd True --eps epsilonsu --projection True --proj_dims 1 --lanczos_iter 256 --save_dir res_3 >log_3/mnist/lr/iid/log_10_bs128_nm128_10000_100_R8_dpuni_pro1_256 2>&1 &

nohup python main_v6.py --max_steps 10000 --N 10 --num_microbatches 128 --client_batch_size 128 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model lr --dpsgd True --eps epsilonsu --fedavg True --save_dir res_3 >log_3/mnist/lr/iid/log_10_bs128_nm128_10000_100_R8_dpuni_fedavg 2>&1 &
'
:'
nohup python main_v6.py --max_steps 10000 --N 20 --num_microbatches 128 --client_batch_size 128 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model lr --dpsgd True --eps epsilons1 --projection True --proj_dims 1 --lanczos_iter 256 --save_dir res_3 >log_3/mnist/lr/iid/dp1/log_20_bs128_nm128_10000_100_R8_dp1_pro1_256 2>&1 &

nohup python main_v6.py --max_steps 10000 --N 10 --num_microbatches 128 --client_batch_size 128 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model lr --dpsgd True --eps epsilons1 --fedavg True --version 3 >log_3/mnist/lr/iid/dp1/log_10_bs128_nm128_10000_100_R8_fedavg 2>&1 &
'
:'
# iid setting: N=10
# [0.5+0.01, 10+0.1]: 1.0
# projection dims 1/2/5/10/20/50
for i in 2 5 10 20 50;
do
    nohup python main_v6.py --max_steps 10000 --N 10 --num_microbatches 128 --client_batch_size 128 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model lr --dpsgd True --eps epsilons1 --projection True --proj_dims $i --lanczos_iter 256 --save_dir res_3 >log_3/mnist/lr/iid/log_30_bs128_nm128_10000_100_R8_dp1_pro${i}_256_constlr_0202_v6 2>&1 &
done
'
:'
for ((i=10; i<=50; i=i+10));
do

nohup python main_v6.py --max_steps 10000 --N $i --num_microbatches 128 --client_batch_size 128 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model lr --version 3 >log_3/mnist/lr/iid/log_10_bs128_nm128_10000_100_R8_nodp 2>&1 &

done
'
:'
# fmnist
for ((i=10; i<=50; i=i+10));
do
    nohup python main_v6.py --max_steps 10000 --N $i --dataset fmnist --num_microbatches 128 --client_batch_size 128 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model lr --dpsgd True --eps epsilons1 --projection True --proj_dims 1 --lanczos_iter 256 --version 3 >log_3/fmnist/lr/iid/${i}_bs128_nm128_10000_100_R8_dp1_pro1_256_constlr 2>&1 &

    nohup python main_v6.py --max_steps 10000 --N $i --dataset fmnist --num_microbatches 128 --client_batch_size 128 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model lr --dpsgd True --eps epsilons1 --wei_avg True --version 3 >log_3/fmnist/lr/iid/${i}_bs128_nm128_10000_100_R8_dp1_wavg_constlr 2>&1 &

    nohup python main_v6.py --max_steps 10000 --N $i --dataset fmnist --num_microbatches 128 --client_batch_size 128 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model lr --dpsgd True --eps epsilons1 --fedavg True --version 3 >log_3/fmnist/lr/iid/${i}_bs128_nm128_10000_100_R8_dp1_fedavg_constlr 2>&1 &

    nohup python main_v6.py --max_steps 10000 --N $i --dataset fmnist --num_microbatches 128 --client_batch_size 128 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model lr --version 3 >log_3/fmnist/lr/iid/${i}_bs128_nm128_10000_100_R8_nodp_constlr 2>&1 &
done

# fmnist
for ((i=10; i<=50; i=i+10));
do
    nohup python main_v6.py --max_steps 10000 --N $i --dataset fmnist --num_microbatches 128 --client_batch_size 128 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model lr --dpsgd True --eps epsilons2 --projection True --proj_dims 1 --lanczos_iter 256 --version 3 >log_3/fmnist/lr/iid/${i}_bs128_nm128_10000_100_R8_dp2_pro1_256_constlr 2>&1 &

    nohup python main_v6.py --max_steps 10000 --N $i --dataset fmnist --num_microbatches 128 --client_batch_size 128 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model lr --dpsgd True --eps epsilons2 --wei_avg True --version 3 >log_3/fmnist/lr/iid/${i}_bs128_nm128_10000_100_R8_dp2_wavg_constlr 2>&1 &

    nohup python main_v6.py --max_steps 10000 --N $i --dataset fmnist --num_microbatches 128 --client_batch_size 128 --sample_mode R --sample_ratio 0.8 --local_steps 100 --model lr --dpsgd True --eps epsilons2 --fedavg True --version 3 >log_3/fmnist/lr/iid/${i}_bs128_nm128_10000_100_R8_dp2_fedavg_constlr 2>&1 &

done
'

