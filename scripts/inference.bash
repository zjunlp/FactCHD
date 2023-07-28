CUDA_VISIBLE_DEVICES="0" python src/inference.py \
    --lora_weights 'lora/alpaca-7b-fact' \
    --base_model 'models/alpaca-7b' \
    --input_file 'data/test.json' \
    --output_file 'results/alpaca_7b_fact_test.json' 