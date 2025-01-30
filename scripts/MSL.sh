export CUDA_VISIBLE_DEVICES=1

python main_OT.py --k 3 --num_proto 12 --len_map 16   --mode train --dataset MSL --data_path dataset/MSL --input_c 55    --output_c 55

python main_OT.py --k 3 --num_proto 12 --len_map 16   --mode test --anomaly_ratio 0.8 --dataset MSL --data_path dataset/MSL --input_c 55 --output_c 55