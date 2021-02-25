CUDA_VISIBLE_DEVICES=0 python -u -W ignore train.py --path ./data_preprocessed/zinc250k_clean_sorted \
    --gen --gen_out_path ./mols/test_100mol.txt \
    --batch_size 32 --lr 0.001 --epochs 100 \
    --shuffle --deq_coeff 0.9 --save --name l12_h128_o128_exp_sbatch \
    --num_flow_layer 12 --nhid 128 --nout 128 --gcn_layer 3 \
    --is_bn --divide_loss --st_type exp \
    --init_checkpoint ./good_ckpt/checkpoint277 \
    --gen_num 100 --min_atoms 10 --save --seed 66666666 --temperature 0.7
