import datetime
import json
import os
import random
import re
import time
from collections import defaultdict

import wikipediaapi
from loguru import logger
import pandas as pd
from wikipedia import wikipedia

from data_generate.openai_service import get_response, get_response_retry
from utils.const import *
from utils import prompter, create_dir

# wiki_wiki = wikipediaapi.Wikipedia('en')
# wikipedia.set_lang("en")

now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

logger.add("../log/text_info-" + now + ".log", format="{message}", filter=lambda record: record["level"].name == "INFO")
logger.add("../log/text_error-" + now + ".log", format="{message}",
           filter=lambda record: record["level"].name == "ERROR")


def clean_text(sentence):
    sentence = re.sub(" \-LSB\-.*?\-RSB\-", "", sentence)
    sentence = re.sub("\-LRB\- \-RRB\- ", "", sentence)
    sentence = re.sub(" -LRB-", " ( ", sentence)
    sentence = re.sub("-RRB-", " )", sentence)

    sentence = re.sub(" LSB.*?RSB", "", sentence)
    sentence = re.sub("LRB RRB ", "", sentence)
    sentence = re.sub("LRB", " ( ", sentence)
    sentence = re.sub("RRB", " )", sentence)
    sentence = re.sub("--", "-", sentence)
    sentence = re.sub("``", '"', sentence)
    sentence = re.sub("''", '"', sentence)
    sentence = re.sub('  ', ' ', sentence)
    return sentence


def clean_title(title):
    title = re.sub("_", " ", title)
    title = re.sub(" -LRB-", " ( ", title)
    title = re.sub("-RRB-", " )", title)
    title = re.sub("-COLON-", ":", title)
    title = re.sub('  ', ' ', title)
    return title


def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError


def to_fine_tune_data(data):
    evidence = []
    golden_evidence = []
    for e in data['evidence']:
        evidence.append([e[0], e[1], e[2]])
        if e[3] >= 1:
            golden_evidence.append([e[0], e[1], e[2]])

    input = {
        "claim": data['claim'],
        "evidence": evidence,
    }
    output = {
        "label": data['label'],
        "golden_evidence": golden_evidence
    }
    return {
        "instruction": "You are now tasked with verifying facts with a high degree of intelligence and "
                       "accuracy. Please review the evidence, which consists of sets of ordered sentences "
                       "from Wikipedia pages in the form of [Wikipedia Title, sentence ID, "
                       "sentence] tuples. Select one or more evidence from the evidence as golden "
                       "evidence, that is as little as possible but at least one, and label the claim as "
                       "SUPPORTS or REFUTES based on it. Please note again that the golden evidence "
                       "should be as small as possible but at least one, and it can be inferred whether "
                       "the claim is supported or refuted. If there is not enough evidence to support or "
                       "refute the claim, label it as NOT ENOUGH INFO, and the golden evidence should be "
                       "empty. The input format for this task strictly follows the JSON format, "
                       "including the fields claim and evidence. The output format also strictly follows "
                       "the JSON format, including the fields label and golden_evidence. Be careful not "
                       "to output anything else!",
        "input": json.dumps(input, ensure_ascii=False),
        "output": json.dumps(output, ensure_ascii=False)
    }, golden_evidence


def transform_label(label):
    if label == SUPPORTS:
        return "True"
    elif label == REFUTES:
        return "False"
    else:
        return NOT_ENOUGH_INFO


def to_fine_tune_data_v2(data, only_label=False):
    golden_evidence_list = []
    evidence_list = []
    # n = random.randint(1, 4)
    n = 0

    if data['label'] == 'NOT ENOUGH INFO':
        n = max(n, 1)

    for e in data['evidence']:
        if e[3] >= 1:
            golden_evidence_list.append(clean_title(e[0]) + ":" + clean_text(e[2]))
            evidence_list.append(clean_title(e[0]) + ":" + clean_text(e[2]))
        else:
            if n > 0:
                n -= 1
                evidence_list.append(clean_title(e[0]) + ":" + clean_text(e[2]))

    evidence = ""
    for e in evidence_list:
        evidence += "[" + e + "],"
    evidence = evidence[:-1]

    if only_label:
        output = transform_label(data['label'])
    else:
        golden_evidence = ""
        for i in range(len(golden_evidence_list)):
            if i == len(golden_evidence_list) - 1:
                golden_evidence += "[" + golden_evidence_list[i] + "]"
            elif i == len(golden_evidence_list) - 2:
                golden_evidence += "[" + golden_evidence_list[i] + "] and "
            else:
                golden_evidence += "[" + golden_evidence_list[i] + "], "
        if data['label'] == 'NOT ENOUGH INFO':
            output = NOT_ENOUGH_INFO + "\n" + " There isn’t sufficient evidence to either support of refute it"
        else:
            output = data['label'] + "\n" + "I infer {claim: " + data['claim'] + "} is " + data['label'] + \
                     " based on the following evidence: " + golden_evidence + "."

    return {
        "instruction": "I want you to act as a fallacy finder. You will be on the lookout for invalid arguments so "
                       "you can call out any logical errors or inconsistencies that may be present in claim. Your job "
                       "is to provide evidence-based feedback and determine whether there are any fallacies, "
                       "faulty reasoning, faulty assumptions, or incorrect conclusions that may be present in the "
                       "claim. Your results should be returned in the following format only: <True/False/Not Enough "
                       "Info>.",
        "input": "\n Claim:" + data['claim'] + "\n Evidence:" + evidence,
        "output": output
    }, golden_evidence_list


def get_golden_evidence_list(data, data_type):
    golden_evidence_list = []
    golden_title_set = set()
    if data_type == "climate_fever":
        for e in data['evidences']:
            if e['evidence_label'] == data['label']:
                golden_evidence_list.append([e['article'], e['evidence']])
                golden_title_set.add(e['article'])
        return golden_evidence_list, golden_title_set

    n = 0
    if data['label'] == NOT_ENOUGH_INFO:
        n = max(n, 1)

    for e in data['evidence']:
        if isinstance(e, str):
            golden_evidence_list.append(e)
            continue

        if e[3] >= 1:
            golden_evidence_list.append([clean_title(e[0]), clean_text(e[2])])
            golden_title_set.add(clean_title(e[0]))
        else:
            if n > 0:
                n -= 1
                golden_evidence_list.append([clean_title(e[0]), clean_text(e[2])])
                golden_title_set.add(clean_title(e[0]))
    return golden_evidence_list, golden_title_set


def chatgpt_judge_claim(data, data_type, output_file):
    # random.seed(time.time())
    # response =
    # if random.random() < 0.7:
    #     if data['label'] == REFUTES:
    #         data['response'] = response
    #         data['data_type'] = data_type
    #         with open(output_file, "a", encoding='utf-8') as f:
    #             f.write(json.dumps(data, ensure_ascii=False) + "\n")
    #         return data, True
    #     else:
    #         data['response'] = response
    #         return data, False
    # else:
    #     data['response'] = response
    #     data['data_type'] = data_type
    #     with open(output_file, "a", encoding='utf-8') as f:
    #         f.write(json.dumps(data, ensure_ascii=False) + "\n")
    #     return data, True

    prompt = prompter.CLAIM_JUDGE_DATA.format(claim=data['claim'])
    response = get_response_retry(prompt, temperature=0.75, max_tokens=50)

    if response is None:
        logger.error("chatgpt_judge_claim response is None")
        logger.error(data)
        return None, False
    flag = False
    if "#Output#:" in response:
        response = response.split("#Output#:")[1]
    response = response.upper().strip()
    if response.startswith(data['label'].upper()):
        data['response'] = response
        return data, flag
    else:
        data['response'] = response
        data['data_type'] = data_type
        with open(output_file, "a", encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
        flag = True
        return data, flag


def generate_data_use_chatgpt(golden_title_set, data):
    label = data['label'].upper()
    # for title in golden_title_set:
    #     summary = ""
    #     try:
    #         summary = wikipedia.summary(title)
    #     except Exception as exc:
    #         logger.info(exc)
    #     if summary == "":
    #         summary = wiki_wiki.page(title).summary.strip()
    #     if summary == "":
    #         logger.info("summary为空" + title)
    #         return None, None
    #     evidence += "[" + summary + "],"
    summary = {}
    for e in data['evidence']:
        if clean_title(e[0]) in golden_title_set:
            summary.setdefault(clean_title(e[0]), []).append((clean_text(e[2]), e[1]))

    summary_evidence = ""
    for title in golden_title_set:
        summary[title].sort(key=lambda x: x[1])
        summary_str = ""
        for i in range(len(summary[title])):
            summary_str += summary[title][i][0]
        summary_evidence += "[" + summary_str + "],"

    summary_evidence = summary_evidence[:-1]

    prompt = prompter.get_summary_generate_data_for_label(label).format(label=label,
                                                                        claim=data['claim'],
                                                                        evidence=summary_evidence)
    response = ""
    try:
        response = get_response(prompt, temperature=0, max_tokens=2048)
    except TimeoutError as e:
        logger.info(e)
        time.sleep(60)
        try:
            response = get_response(prompt, temperature=0, max_tokens=2048)
        except:
            logger.info(e)
            return None
    except Exception as e:
        logger.info(e)
        logger.info(len(prompt))
        return None

    predict_label = response.split("\n")[0].upper()

    data["predict_label"] = predict_label
    data["reason"] = response
    data["golden_title_set"] = golden_title_set
    data["summary_evidence"] = summary_evidence

    if label not in predict_label:
        logger.error(json.dumps(data, ensure_ascii=False, default=set_default))
        return None
    else:
        return data
        # return {
        #     "data": data,
        #     "instruction": "I want you to act as a fallacy finder. You will be on the lookout for invalid arguments "
        #                    "so you can call out any logical errors or inconsistencies that may be present in claim. "
        #                    "Your job is to provide evidence-based feedback and point out any fallacies, "
        #                    "faulty reasoning, false assumptions, or incorrect conclusions that may be present in the "
        #                    "claim.",
        #     "input": "Claim:" + data['claim'] + "\n Evidence:" + evidence,
        #     "output": response,
        # }


def generate_question_use_chatgpt(data):
    prompt = prompter.QUESTION_GENERATE_DATA.format(answer=data['claim'])
    response = get_response_retry(prompt, temperature=0.7, max_tokens=1024)
    if response is None:
        logger.error("generate_question_use_chatgpt response is None")
        logger.error(data)
        return None
    data["question"] = response
    return data


def get_statistics(evidence_nums, evidence_len, title_len, title_nums, golden_evidence_nums, golden_evidence_len, data):
    evidence_nums.append(len(data['evidence']))
    evidence_sum = 0
    title_sum = 0
    golden_evidence_len_sum = 0
    golden_evidence_num_sum = 0
    s = set()
    for e in data['evidence']:
        evidence_sum += len(e[2])
        title_sum += len(e[0])
        s.add(e[0])
        if e[3] >= 1:
            golden_evidence_num_sum += 1
            golden_evidence_len_sum += len(e[2])
    golden_evidence_nums.append(golden_evidence_num_sum)
    golden_evidence_len.append(golden_evidence_len_sum)
    evidence_len.append(evidence_sum)
    title_len.append(title_sum)
    title_nums.append(len(s))
    # # 格式化保存list文件
    # with open(save_path, 'w', encoding='utf-8') as f:
    #     f.write(json.dumps(datas, indent=4, ensure_ascii=False))

    # print("evidence_nums", "mean", "std", "min", "max")
    # print("evidence_nums", np.mean(evidence_nums), np.std(evidence_nums), min(evidence_nums), max(evidence_nums))
    # print("evidence_len", np.mean(evidence_len), np.std(evidence_len), min(evidence_len), max(evidence_len))
    # print("title_nums", np.mean(title_nums), np.std(title_nums), min(title_nums), max(title_nums))
    # print("title_len", np.mean(title_len), np.std(title_len), min(title_len), max(title_len))
    # print("golden_evidence_nums", np.mean(golden_evidence_nums), np.std(golden_evidence_nums),
    #       min(golden_evidence_nums),
    #       max(golden_evidence_nums))
    # print("golden_evidence_len", np.mean(golden_evidence_len), np.std(golden_evidence_len),
    #       min(golden_evidence_len),
    #       max(golden_evidence_len))
    # # 统计evidence_nums每一个数出现了多少次，然后画出直方图，纵坐标粒度为10
    # evidence_nums = np.array(evidence_nums)
    # bins = np.arange(0, max(evidence_nums) + 1, 1)
    # plt.hist(evidence_nums, bins=bins)
    # plt.show()
    #
    # # 统计title_nums每一个数出现了多少次，然后画出直方图，纵坐标粒度为10
    # title_nums = np.array(title_nums)
    # bins = np.arange(0, max(title_nums) + 1, 1)
    # plt.hist(title_nums, bins=bins)
    # plt.show()
    #
    # # 统计golden_evidence_nums每一个数出现了多少次，然后画出直方图，纵坐标粒度为1
    # golden_evidence_nums = np.array(golden_evidence_nums)
    # bins = np.arange(0, max(golden_evidence_nums) + 1, 1)
    # plt.hist(golden_evidence_nums, bins=bins)
    # plt.show()


def response_to_label(response):
    response = response.upper()
    if response.startswith(SUPPORTS):
        return SUPPORTS
    if response.startswith(REFUTES):
        return REFUTES
    if response.startswith(I_DONT_KNOW):
        return I_DONT_KNOW
    return EXCEPTION


labels = {SUPPORTS, REFUTES, NOT_ENOUGH_INFO, I_DONT_KNOW}

# 处理类型的枚举
FILTER = "filter"
STATISTICS = "statistics"
GENERATE_QA = "generate_qa"
GENERATE_ALL = "generate_all"


def generate_all_use_chatgpt(data, golden_evidence_list):
    evidence = ""
    for e in golden_evidence_list:
        if isinstance(e, str):
            evidence += "[" + e + "], "
        else:
            evidence += "[" + e[0] + ": " + e[1] + "], "
    evidence = evidence[:-2]
    if data['label'] == SUPPORTS:
        label = "FACTUAL"
    elif data['label'] == REFUTES:
        label = "NON-FACTUAL"
    else:
        return
    prompt = prompter.TEXT_GENERATE_ALL_DATA.format(label=label,
                                                    question=data['question'],
                                                    answer=data['claim'],
                                                    evidence=evidence)
    response = get_response_retry(prompt, temperature=0, max_tokens=4000)
    if response is None:
        logger.error("generate_question_use_chatgpt response is None")
        logger.error(data)
        return None
    predict_label = response.split("\n")[0].upper()
    if data['label'] == "NON-FACTUAL" and "NON-FACTUAL" not in predict_label:
        logger.error("data: " + str(data))
        logger.error("response: " + response)
        return
    elif data['label'] == "FACTUAL" and "NON-FACTUAL" in predict_label:
        logger.error("data: " + str(data))
        logger.error("response: " + response)
        return
    data["label"] = label
    data["reason"] = response
    data["golden_evidence_list"] = golden_evidence_list

    return data


def fever_process_data(
        file_path: str = "../data/dataset/fever/all_dev.json",
        save_path: str = "../data/dataset/fever/fever_chatgpt_filter_test_all.json",
        output_path: str = "../data/output/text/fever_chatgpt_filter_test_all_filter.json",
        process_type: str = STATISTICS,
        data_type: str = "fever",
        limit: int = 20000,
):
    # 导入json文件
    id_set = set()
    if not os.path.exists(save_path):
        with open(save_path, "w", encoding='utf-8') as f:
            f.write("")
    tot = 0
    with open(save_path, "r", encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                if process_type == FILTER:
                    if data['response'].startswith(data['label'].upper()) is False:
                        tot += 1
                if 'id' in data:
                    id_set.add(data['id'])
    if process_type != FILTER:
        tot = len(id_set)
    tot = 0
    if tot >= limit:
        logger.info(str(process_type) + "数据量已有{}条，不再继续处理".format(tot))
        return
    logger.info(str(process_type) + "数据量已有{}条，继续处理".format(tot))

    i = 0
    label2nums = defaultdict(int)
    wf = open(save_path, "a", encoding='utf-8')
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            i += 1
            if i % 100 == 0:
                print(str(process_type) + ", 数据下标:", i, ",产生数据: ", tot, ",label_num:", str(label2nums))
            line = line.strip()
            if line:
                data = json.loads(line)
                if 'claim_label' in data:
                    data['label'] = data['claim_label']
                if 'claim_id' in data:
                    data['id'] = data['claim_id']

                if data['label'] == NOT_ENOUGH_INFO or data['label'] == "DISPUTED":
                    continue

                if data['id'] in id_set:
                    continue

                golden_evidence_list, golden_title_set = get_golden_evidence_list(data, data_type)
                if len(golden_evidence_list) > 5:
                    continue

                if process_type == FILTER:
                    data, is_ok = chatgpt_judge_claim(data, data_type, output_path)
                elif process_type == GENERATE_QA:
                    data = generate_question_use_chatgpt(data)
                elif process_type == GENERATE_ALL:
                    data = generate_all_use_chatgpt(data, golden_evidence_list)

                if data is None:
                    continue
                tot += 1
                if process_type == FILTER and is_ok is False:
                    tot -= 1

                data['data_type'] = data_type
                id_set.add(data['id'])
                wf.write(json.dumps(data, ensure_ascii=False) + '\n')
                if tot >= limit:
                    break

                label = data['label']
                flag = None
                if len(golden_evidence_list) > 1:
                    if label == SUPPORTS:
                        flag = "SUPPORTS_MANY_EVIDENCE"
                    elif label == REFUTES:
                        flag = "REFUTES_MANY_EVIDENCE"
                elif label == SUPPORTS:
                    flag = SUPPORTS
                elif label == REFUTES:
                    flag = REFUTES
                elif label == NOT_ENOUGH_INFO:
                    flag = NOT_ENOUGH_INFO
                label2nums[label] += 1

    wf.close()
    print("done! 数据下标:", i, ",产生数据: ", tot, ",label_num:", str(label2nums))


def generate_data(
        file_path: str = "../data/output/text/chatgpt_judge_claim_output.json",
        save_path: str = "../data/output/text/chatgpt_output_qa.json",
):
    id_set = set()
    if not os.path.exists(save_path):
        with open(save_path, "w", encoding='utf-8') as f:
            f.write("")

    with open(save_path, "r", encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                if 'id' in data:
                    id_set.add(data['id'])

    label2nums = {
        SUPPORTS: 0,
        "SUPPORTS_MANY_EVIDENCE": 0,
        REFUTES: 0,
        "REFUTES_MANY_EVIDENCE": 0,
        NOT_ENOUGH_INFO: 0,
    }
    i = 0
    wf = open(save_path, "a", encoding='utf-8')
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            i += 1
            line = line.strip()
            if line:
                data = json.loads(line)

                if 'response' in data:
                    if response_to_label(data['response']) not in labels:
                        continue

                golden_evidence_list, golden_title_set = get_golden_evidence_list(data)
                label = data['label']
                flag = None
                if len(golden_evidence_list) > 1:
                    if label == SUPPORTS:
                        flag = "SUPPORTS_MANY_EVIDENCE"
                    elif label == REFUTES:
                        flag = "REFUTES_MANY_EVIDENCE"
                elif label == SUPPORTS:
                    flag = SUPPORTS
                elif label == REFUTES:
                    flag = REFUTES
                elif label == NOT_ENOUGH_INFO:
                    flag = NOT_ENOUGH_INFO

                if data['id'] not in id_set:
                    # response = generate_data_use_chatgpt(golden_title_set, data)
                    response = generate_question_use_chatgpt(data)
                    if response is None:
                        continue
                    id_set.add(data['id'])
                    wf.write(json.dumps(response, ensure_ascii=False, default=set_default) + '\n')
                else:
                    label2nums[flag] += 1

                if i % 100 == 0:
                    print("训练数据:", i, ",产生数据:", len(id_set), label2nums)

    print(label2nums)


def do_scifact():
    corpus_candidates = {}
    with open("../data/dataset/scifact/data/corpus.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                corpus_candidates[str(data['doc_id'])] = data

    with open("../data/dataset/scifact/scifact.jsonl", 'w', encoding='utf-8') as wf:
        for file in ["../data/dataset/scifact/data/claims_train.jsonl",
                     "../data/dataset/scifact/data/claims_dev.jsonl"]:
            with open(file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        evidences = []
                        for doc_id, es in data['evidence'].items():
                            flag = False
                            for e in es:
                                doc = corpus_candidates[doc_id]
                                for idx in e['sentences']:
                                    evidences.append([doc['title'], idx, doc['abstract'][idx], 1])
                                if e['label'] == 'CONTRADICT':
                                    data['label'] = REFUTES
                                else:
                                    data['label'] = SUPPORTS
                                data['evidence'] = evidences
                                wf.write(json.dumps(data, ensure_ascii=False) + '\n')
                                flag = True
                                break
                            if flag:
                                break


def do_pubhealth():
    # 读取TSV文件
    for file in [
        "../data/dataset/pubhealth/test.tsv",
        "../data/dataset/pubhealth/dev.tsv",
        "../data/dataset/pubhealth/train.tsv",
    ]:
        save_file = "../data/dataset/pubhealth/" + file.split('/')[-1].split('.')[0] + ".jsonl"
        with open(save_file, 'w', encoding='utf-8') as wf:
            tsv = pd.read_csv(file, sep='\t')
            # 遍历
            for index, row in tsv.iterrows():
                label = None
                if row['label'] == 'true':
                    label = SUPPORTS
                elif row['label'] == 'false':
                    label = REFUTES
                else:
                    continue
                data = {
                    "id": row['claim_id'],
                    "claim": row['claim'],
                    "subjects": row['subjects'],
                    "main_text": row['main_text'],
                    "date_published": row['date_published'],
                    "evidence": [row["explanation"]],
                    "sources": row['sources'],
                    "label": label,
                }
                wf.write(json.dumps(data, ensure_ascii=False) + '\n')


def do_healthver():
    for file in [
        "../data/dataset/healthver/healthver_test.csv",
        "../data/dataset/healthver/healthver_dev.csv",
        "../data/dataset/healthver/healthver_train.csv",
    ]:
        save_file = "../data/dataset/healthver/" + file.split('/')[-1].split('.')[0] + ".jsonl"
        with open(save_file, 'w', encoding='utf-8') as wf:
            tsv = pd.read_csv(file, sep=',')
            # 遍历
            for index, row in tsv.iterrows():
                label = None
                if row['label'] == 'Supports':
                    label = SUPPORTS
                elif row['label'] == 'Refutes':
                    label = REFUTES
                else:
                    continue
                data = {
                    "id": row['id'],
                    "claim": row['claim'],
                    "evidence": [row["evidence"]],
                    "label": label,
                    "question": row['question']
                }
                wf.write(json.dumps(data, ensure_ascii=False) + '\n')


def merge_data(suffix="test"):
    label_map = defaultdict(int)
    domain_map = defaultdict(int)
    deep_map = {}
    files = []
    if suffix == "train":
        files = [
            "../data/output/text/fever_chatgpt_generate_all_train.jsonl",
            "../data/output/text/climate_fever_chatgpt_generate_all_train1.jsonl",
            "../data/output/text/healthver_chatgpt_generate_all_train.jsonl",
            "../data/output/text/healthver_chatgpt_generate_all_dev.jsonl",
            "../data/output/text/pubhealth_chatgpt_generate_all_train.jsonl",
            "../data/output/text/pubhealth_chatgpt_generate_all_dev.jsonl",
            "../data/output/text/scifact_chatgpt_generate_all.jsonl",
        ]
    elif suffix == "test":
        files = [
            "../data/output/text/fever_chatgpt_generate_all_test.jsonl",
            "../data/output/text/climate_fever_chatgpt_generate_all_test.jsonl",
            "../data/output/text/healthver_chatgpt_generate_all_test.jsonl",
            "../data/output/text/pubhealth_chatgpt_generate_all_test.jsonl",
            "../data/output/text/scifact_chatgpt_generate_all.jsonl",
        ]
    save_file = "../data/output/text/text_" + suffix + ".jsonl"
    with open(save_file, 'w', encoding='utf-8') as wf:
        for file in files:
            with open(file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        temp = json.loads(line)
                        if "data_type" not in temp or temp['data_type'] == "fever":
                            temp['domain'] = "common"
                        elif temp['data_type'] == "climate_fever":
                            temp['domain'] = "climate"
                        elif temp['data_type'] == "pubhealth":
                            temp['domain'] = "public health"
                        elif temp['data_type'] == "scifact_open" or temp['data_type'] == "scifact":
                            temp['domain'] = "scientific"
                        elif temp['data_type'] == "healthver":
                            temp['domain'] = "COVID-19"
                        else:
                            pass
                        temp['id'] = temp['domain'] + "_" + str(temp['id'])
                        data = {
                            'id': temp['id'],
                            'label': temp['label'],
                            'question': temp['question'],
                            'response': temp['claim'],
                            'evidence': temp['golden_evidence_list'],
                            'reason': temp['reason'],
                            'type': "text",
                            'domain': temp['domain']
                        }
                        wf.write(json.dumps(data, ensure_ascii=False) + '\n')

                        label_map[data['label']] += 1
                        domain_map[data['domain']] += 1
                        if data['domain'] not in deep_map:
                            deep_map[data['domain']] = defaultdict(int)
                        deep_map[data['domain']][data['label']] += 1

    # 格式化输出
    print("label_map:")

    for k, v in label_map.items():
        print(k, v)
    print()
    print("domain_map:")
    for k, v in domain_map.items():
        print(k, v)
    print()
    print("deep_map:")
    for k, v in deep_map.items():
        print(k, v)
    print()
    print("sum:")
    print(sum(label_map.values()))


def add_file_to_file():
    text_train = []
    with open("../data/output/text/text_train1.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                temp = json.loads(line)
                data = {
                    'id': temp['id'],
                    'label': temp['label'],
                    'query': temp['question'],
                    'response': temp['response'],
                    'evidence': temp['evidence'],
                    'reason': temp['reason'],
                    'domain': temp['domain'],
                    'type': "text",
                }
                if isinstance(data['evidence'], str):
                    continue
                text_train.append(data)

    text_test = []
    with open("../data/output/text/text_test2.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                temp = json.loads(line)
                data = {
                    'id': temp['id'],
                    'label': temp['label'],
                    'query': temp['question'],
                    'response': temp['response'],
                    'evidence': temp['evidence'],
                    'reason': temp['reason'],
                    'domain': temp['domain'],
                    'type': "text",
                }
                if isinstance(data['evidence'], str):
                    continue
                text_test.append(data)

    kg_test = []
    with open("../data/output/kg/kg_test3.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                kg_test.append(data)

    kg_train = []
    with open("../data/output/kg/kg_train4.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                kg_train.append(data)

    fact_check_test = []
    fact_check_test.extend(kg_test)
    fact_check_test.extend(text_test)
    random.shuffle(fact_check_test)

    fact_check_train = []
    fact_check_train.extend(kg_train)
    fact_check_train.extend(text_train)
    random.shuffle(fact_check_train)

    with open("../data/output/fact_check_test.jsonl", "w", encoding="utf-8") as wf:
        for data in fact_check_test:
            wf.write(json.dumps(data, ensure_ascii=False) + '\n')

    with open("../data/output/fact_check_train.jsonl", "w", encoding="utf-8") as wf:
        for data in fact_check_train:
            wf.write(json.dumps(data, ensure_ascii=False) + '\n')


def do_sci_file():
    datas = []
    start = 3004
    i = 1
    with open("../data/output/kg/medicine_multi_hop_reasoning_all_train.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                if i <= start:
                    i += 1
                    continue
                temp = json.loads(line)
                temp['domain'] = "common"
                temp['id'] = temp['domain'] + "_" + str(temp['id'])
                data = {
                    'id': temp['id'],
                    'label': temp['label'],
                    'query': temp['question'],
                    'response': temp['chatgpt_response'],
                    'evidence': temp['used_knowledge'],
                    'reason': temp['reason'],
                    'type': "kg",
                    'domain': 'medicine',
                    'subgraph_type': temp['type'],
                }
                datas.append(data)

    fact_check_train = []
    with open("../data/output/fact_check_train3.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                fact_check_train.append(data)

    fact_check_train.extend(datas)
    random.shuffle(fact_check_train)

    with open("../data/output/fact_check_train4.jsonl", "w", encoding="utf-8") as wf:
        for data in fact_check_train:
            wf.write(json.dumps(data, ensure_ascii=False) + '\n')

    pass


if "__main__" == __name__:
    # Fever Dataset Construction Process
    data_dir = "data/dataset/fever"
    create_dir(data_dir)
    data_path = os.path.join(data_dir, "all_train.json")

    filter_save_dir = "data/dataset/fever"
    create_dir(filter_save_dir)
    filter_save_path = os.path.join(filter_save_dir, "fever_chatgpt_filter_all_train.jsonl")

    filter_output_dir = "output/text/filter"
    create_dir(filter_output_dir)
    filter_output_path = os.path.join(filter_output_dir, "fever_chatgpt_filter_all_train.jsonl")

    # 1. Filter out data from ChatGPT that did not pass evidence verification
    fever_process_data(data_path, filter_save_path, filter_output_path, FILTER, limit=1)

    # 2. Generate questions using ChatGPT
    generate_qa_data_dir = "output/text/qa"
    create_dir(generate_qa_data_dir)
    generate_qa_data_path = os.path.join(generate_qa_data_dir, "fever_chatgpt_generate_qa_train.jsonl")
    fever_process_data(filter_output_path, generate_qa_data_path, "", GENERATE_QA, limit=1)

    # 3. Generate evidence reasoning chains using ChatGPT
    all_save_dir = "output/text/all"
    create_dir(all_save_dir)
    all_save_path = os.path.join(all_save_dir, "fever_chatgpt_generate_all_train.jsonl")
    fever_process_data(generate_qa_data_path, all_save_path, "", GENERATE_ALL, limit=1)

