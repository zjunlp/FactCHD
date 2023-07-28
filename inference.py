import os
import sys
import json
import fire
import torch
from tqdm import tqdm
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    set_peft_model_state_dict,
)
from peft.utils import WEIGHTS_NAME
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer
from utils.prompter import Prompter
from utils import MODEL_DICT

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["WANDB_DISABLED"] = "true"


def get_model_name(model_name):
    model_name = model_name.lower()
    for key, values in MODEL_DICT.items():
        for v in values:
            if v in model_name:
                return key
    return ""


def main(
    load_8bit: bool = False,
    base_model: str = None,
    lora_weights: str = None,
    prompt_template: str = "alpaca",  # The prompt template to use, will default to alpaca.
    input_file: str = None,
    output_file: str = None,
    mode: str = "w",
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(prompt_template)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        if lora_weights is not None:
            if load_8bit:
                config = LoraConfig.from_pretrained(lora_weights)
                model = get_peft_model(model, config) 
                adapters_weights = torch.load(os.path.join(lora_weights, WEIGHTS_NAME), map_location=model.device)
                set_peft_model_state_dict(model, adapters_weights)
            else:
                model = PeftModel.from_pretrained(
                    model,
                    lora_weights,
                )

    model_name = get_model_name(base_model)
    print("model_name", model_name)
    if  model_name == 'llama':
        model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = tokenizer.bos_token_id = 1
        model.config.eos_token_id = tokenizer.eos_token_id = 2
        tokenizer.padding_side = "left"
    else:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
    print(f"BOS:{tokenizer.bos_token_id},{tokenizer.bos_token}\tEOS:{tokenizer.eos_token_id},{tokenizer.eos_token}\tPAD:{tokenizer.pad_token_id},{tokenizer.pad_token}")


    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
        instruction,
        input=None,
        temperature=0.2,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=256,
        model_name='llama',
        print_cnt=5,
        **kwargs,
    ):
        prompt = prompter.generate_prompt(instruction, input)
        if print_cnt > 0:
            print(prompt+"\n\n")
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        if input_ids.size(-1) > 1024:
            print(input_ids.size(-1), "> 1024")
            return ""
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            repetition_penalty=1.3,
            **kwargs,
        )

        with torch.no_grad():
            if model_name == 'falcon':
                generation_output = model.generate(
                    input_ids=input_ids,
                    generation_config=generation_config,
                    max_new_tokens=max_new_tokens,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=tokenizer.pad_token_id,
                    **kwargs,
                )
            else:
                generation_output = model.generate(
                    input_ids=input_ids,
                    generation_config=generation_config,
                    max_new_tokens=max_new_tokens,
                    return_dict_in_generate=True,
                    output_scores=True,
                    **kwargs,
                )

        generation_output = generation_output.sequences[0]
        output = tokenizer.decode(generation_output, skip_special_tokens=True)
        output = prompter.get_response(output)
        return output

    data_path = os.path.join(input_file)
    writer = open(os.path.join(output_file,), mode)
    already = set()
    if mode == "a":
        with open(os.path.join(output_file), "r") as reader:
            for line in reader:
                data = json.loads(line)
                already.add(data["id"])

    cnt = 5
    with open(data_path) as f:
        lines = f.readlines()
        for line in tqdm(lines):
            data = json.loads(line)
            if data["id"] in already:
                print(data["id"], "already exists!")
                continue
            response = evaluate(instruction=data["instruction"], input=data.get("input", None), model_name=model_name, print_cnt=cnt)
            data["output"] = response
            print(response)
            writer.write(json.dumps(data, ensure_ascii=False)+"\n")
            cnt -= 1


if __name__ == "__main__":
    fire.Fire(main)
