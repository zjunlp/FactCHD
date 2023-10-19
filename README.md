# Unveiling the Siren’s Song: Towards Reliable Fact-Conflicting Hallucination Detection

[![Pytorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?e&logo=PyTorch&logoColor=white)](https://pytorch.org/)![](https://img.shields.io/badge/version-1.0.1-blue) [![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/zjunlp/MolGen/blob/main/LICENSE)

🔥  Code for the paper "[Unveiling the Siren’s Song: Towards Reliable Fact-Conflicting Hallucination Detection](https://arxiv.org/abs/2310.12086)".

# 🚀 Overview
Large Language Models (LLMs), such as ChatGPT/GPT-4, have garnered widespread attention owing to their myriad of practical applications, yet their adoption has been constrained by issues of fact-conflicting hallucinations across web platforms. The assessment of factuality in text, produced by LLMs, remains inadequately explored, extending not only to the judgment of vanilla facts but also encompassing the evaluation of factual errors emerging in complex inferential tasks like multi-hop, and etc. In response, we introduce FACTCHD, a fact-conflicting hallucination detection benchmark meticulously designed for LLMs. Functioning as a pivotal tool in evaluating factuality within `"Query-Respons"`  contexts, our benchmark assimilates a large-scale dataset, encapsulating a broad spectrum of factuality patterns, such as vanilla, multi-hops, comparison, and set-operation patterns. A distinctive feature of our benchmark
is its incorporation of fact-based chains of evidence, thereby facilitating comprehensive and conducive factual reasoning throughout the assessment process. We evaluate multiple LLMs, demonstrating the effectiveness of the benchmark and current methods fall short of faithfully detecting factual errors. Furthermore, we present TRUTH-TRIANGULATOR that synthesizes reflective considerations by tool-enhanced ChatGPT and LoRA-tuning based on Llama2, aiming to yield more credible detection through the amalgamation of predictive results and evidence.


## Illustration of Fact-Conflicting Hallucination Detection 
Inspired by the saying ``to know it and to know the reason why of it`` by ``Zhuzi``, `FACTCHD` incorporates fact-based chains of evidence to provide  explanations for its binary predictions.  Based on `FACTCHD`, we aim to explore the application  of fact-conflicting hallucination detection task.

<div align=center><img src="figs/intro.png" width="70%" height="70%" /></div>


## FACTCHD Benchmark Construction

Our benchmark `FACTCHD` comprises a comprehensive dataset, consisting of ``51,383`` ``factual``/``non-factual`` samples for training and  additional ``6,960`` samples for LLM analysis and evaluation. It covers a wide range of domains, including health, medicine, climate, science, and more. `FACTCHD` endeavors to explore the factuality aspect of LLMs by examining four distinct patterns that encompass individual facts and interactions between multiple facts. 
Our automated data construction strategy  centers around harnessing a wealth of extensive knowledge (KG), including the data collection process,  generation of `"Query-Respons"` contexts, fact-based chains of evidence and  human filtering and statistical analysis.

<div align=center><img src="figs/construct.png" width="95%" height="95%" /></div>


# 📕 Requirements
To run the codes, you need to install the requirements:
```
conda create -n fact python=3.9
conda activate fact
pip install -r requirements.txt

mkdir results
mkdir lora
mkdir data
```

# 🤗 Baseline Models 
Here are the baselines models in our evaluation:
* [Alpaca-7b](https://huggingface.co/circulus/alpaca-7b)
* [Vicuna-7b](https://huggingface.co/eachadea/vicuna-7b-1.1)
* [Alpaca-13b](https://huggingface.co/chavinlo/alpaca-13b)
* [Llama2-7b](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)


# 📚 Resource Illustration

## Statistical Analysis of the Benchmark
We provide the  visual representation of the dataset’s domain distribution as following figures.
<div align=center><img src="figs/distribution.png" width="70%" height="70%" /></div>

## Data format
we randomly provide several case examples of our FactLLM in the folder ```/data```.

```json 
id: unique identifier ID.
instruction: instruction for the model indicating a fallacy finder task.
input: includes #Question# and #Answer#.
output: includes labels(NON-FACTUAL or FACTUAL) and justification.
```
The whole dataset is available in [FactLLM](https://drive.google.com/drive/folders/1x6azKA6YqRcUT60utkE630iINHauT1Yg?usp=drive_link) on Google Drive.

##  Input for Tuning
```json
{"id": "common_183152", "instruction": "I want you to act as a 'fallacy finder'. You will be on the lookout for invalid arguments so you can call out any logical errors or inconsistencies that may be present in the #Question# and the #Answer#. Your job is to provide evidence-based feedback and point out any fallacies, faulty reasoning, false assumptions, or incorrect conclusions which may be present in the #Question# and the #Answer#. Begin your response with either FACTUAL or NON-FACTUAL, followed by your justification.", "input": "#Question#: Could you please inform me whether Tata Motors is included in the BSE SENSEX index?\n#Answer#: Tata Motors is excluded from the BSE SENSEX index.", "output": "NON-FACTUAL.The answer that Tata Motors is excluded from the BSE SENSEX index is incorrect. According to the evidence provided, Tata Motors is actually a constituent of the BSE SENSEX index. Therefore, this is a false incorrect conclusion."}
```

# 🙌 How to run


+ ## Data Generation
1. ### Knowledge Graph
   Taking the high-quality dense subset [Wikidata15k](https://cloud.tsinghua.edu.cn/f/ea83c57d262b4a09ab92/?dl=1) from Wikidata as an example, it includes 794 concepts, 16,960 entities, 363 relations, and 846 properties. The generated subgraphs of the knowledge graph contain four types: `MULTI_HOP_REASONING`、`SET_OPERATION`、`QUANTITATIVE_COMPARISON`、`ALL`. 
   Other knowledge graph datasets can refer to this code. This code involves datasets like `Wikidata` and `PrimeKG`. 
   ```bash
   cd data_generate
   python kg_data_generate.py
   ```
2. ### Text
   Taking the FEVER dataset as an example, download the preprocessed dataset [FEVER](https://thunlp.oss-cn-qingdao.aliyuncs.com/KernelGAT/FEVER/KernelGAT.zip). Other datasets in different domains can refer to the following process. This code involves datasets like `FEVER`, `Climate-Fever`, `Health-Fever`, `COVID-FACT`, and `SCIFACT`.
    ```bash
   cd data_generate
   python text_data_generate.py
    ```
+ ## Finetune
```bash
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
```
There are some differents among the finetune of alpaca、vicuna, please refer to `scripts/run_*` for more details. 

+ ## Inference
Please make sure the trained lora weights are placed in the path of `lora_weights`.
```bash
CUDA_VISIBLE_DEVICES="0" python src/inference.py \
    --lora_weights 'lora/alpaca-7b-fact' \
    --base_model 'models/alpaca-7b' \
    --input_file 'data/test.json' \
    --output_file 'results/alpaca_7b_fact_test.json' 
```

