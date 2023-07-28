CUDA_VISIBLE_DEVICES="0" python src/inference.py \
    --lora_weights 'lora/vicuna-7b-fact' \
    --base_model 'models/vicuna-7b' \
    --input_file 'data/test.json' \
    --output_file 'results/vicuna_7b_fact_test.json' \
    --prompt_template 'vicuna'