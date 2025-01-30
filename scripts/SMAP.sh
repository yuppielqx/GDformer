export CUDA_VISIBLE_DEVICES=1

python main_OT.py --k 2 --num_proto 12 --len_map 6    --mode train --anomaly_ratio 0.7 --dataset SMAP --data_path dataset/SMAP --input_c 25    --output_c 25

python main_OT.py --k 2 --num_proto 12 --len_map 6    --mode test --anomaly_ratio 0.7 --dataset SMAP --data_path dataset/SMAP --input_c 25    --output_c 25
