output_dir='lora/alpaca-7b-fact'
mkdir -p ${output_dir}
CUDA_VISIBLE_DEVICES="0,1,2" torchrun --nproc_per_node=3 --master_port=1331 src/finetune.py \
    --base_model 'models/alpaca-7b' \
    --train_path 'data/fact/train.json' \
    --output_dir=${output_dir}  \
    --batch_size 240 \
    --micro_train_batch_size 10 \
    --micro_eval_batch_size 10 \
    --preprocessing_num_workers 4 \
    --num_epochs 10 \
    --learning_rate 1e-4 \
    --cutoff_len 800 \
    --val_set_size 2000 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --group_by_length \
    | tee ${output_dir}/train.log \
    2> ${output_dir}/train.err

