CUDA_VISIBLE_DEVICES=0 python -u -W ignore train.py --path ./data_preprocessed/zinc250k_clean_sorted \
    --train --num_workers 4 \
    --batch_size 32 --lr 0.001 --epochs 3 \
    --shuffle --deq_coeff 0.9 --save --name epoch3_1gpu \
    --num_flow_layer 12 --nhid 128 --nout 128 --gcn_layer 3 \
    --is_bn --divide_loss --st_type exp --seed 2019 \
    --all_save_prefix ./
