from metrics_plus import Metrics
import json
import os
import prettytable
from tqdm import tqdm
print(os.getcwd())

def read_loads(path):
    with open(path, "r", encoding="utf-8") as reader:
        for line in reader:
            # t_line = line.replace('\n','')
            try:
                data = json.loads(line)
            except:
                print("line:",line)
            yield data

def clean(s):
    s = s.strip()
    if s.startswith("Output:"):
        s = s[len("Output:"):]
    if s.startswith("Response:"):
        s = s[len("Response:"):]
    if s.startswith("FACTS:"):
        if "NON-FACTS" in s:
            s = "NON-FACTUAL." + s[len("FACTS:"):]
        else:
            s = "FACTUAL." + s[len("FACTS:"):]
    return s


def compute_score(subgraph_type, cate_format, path1, path2, path3):
    mapper1 = {}
    for data in read_loads(path1):
        mapper1[data['id']] = data[cate_format]

    metrics_dict = {}
    for key in subgraph_type:
        metrics_dict[key] = Metrics(title=key)

    mapper2 = {}
    mapper3 = {}
    for data in read_loads(path2):
        mapper2[data['id']] = data

    for data in read_loads(path3):
        mapper3[data['id']] = data

    for key, value in tqdm(mapper1.items()):
        try:
            data1 = mapper2[key]
            data2 = mapper3[key]
        except:
            print("pass")
        metrics_dict[value].evaluate_response(clean(data1['output']), clean(data2['output']))
        metrics_dict['all'].evaluate_response(clean(data1['output']), clean(data2['output']))
    return metrics_dict



def get_score(metrics_dict, subgraph_type, path4):
    cols = ['type', ]
    metrics_dict_all = {}
    metrics_dict_all1, metrics_dict_all2 = metrics_dict['all'].report()
    metrics_dict_all['all'] = metrics_dict_all2
    for key, _ in metrics_dict_all1.items():
        cols.append(key)
    table = prettytable.PrettyTable(field_names=cols)

    for key in subgraph_type:
        r_dict1, r_dict2 = metrics_dict[key].report()
        metrics_dict_all[key] = r_dict2
        r_dict1['type'] = key
        values = []
        for key in cols:
            values.append(r_dict1[key])
        table.add_row(values) 

    metrics_dict['all'].write_score_dict(path4, metrics_dict_all) 
    return table   

    
#The input test file:
#{"id": "public health_9444", "instruction": "I want you to act as a 'fallacy finder'. You will be on the lookout for invalid arguments so you can call out any logical errors or inconsistencies that may be present in the #Question# and the #Answer#. Your job is to provide evidence-based feedback and point out any fallacies, faulty reasoning, false assumptions, or incorrect conclusions which may be present in the #Question# and the #Answer#. Begin your response with either FACTUAL or NON-FACTUAL, followed by your justification.", 
#"input": "#Question#: Could you provide more information about the results of the first test of anti-epilepsy surgeries in children?\n#Answer#: First test of anti-epilepsy surgeries in children shows dramatic benefit", 
#"output": "FACTUAL.The statement that the first test of anti-epilepsy surgeries in children shows dramatic benefit is supported by the evidence provided. The article describes a randomized trial of neurosurgical procedures designed to eliminate seizures in children under 18 years of age whose epilepsy is resistant to drug therapy. The article also notes significant side effects attributable to the surgeries and the struggle to receive insurance coverage. However, the evidence supports the conclusion that the surgeries have shown to improve quality of life and maintain cognitive health, particularly in young children for whom brain plasticity is stronger than in adults."}


if __name__ == "__main__":
    READ_ONLY = False
    path1 = "./data/all_test.json" # the original test file
    base_path2 = "../lora/results/llama2_7b_4shotdemo_realtrain.json"
    path3 = "./data/test.json" # the input test file
    base_path4 = "./scores_log"
    
    if os.path.isdir(base_path2):
        fs = sorted(os.listdir(base_path2))
    else:
        spt = base_path2.split("/")
        fs = [spt[-1], ]
        base_path2 = base_path2.replace(spt[-1], "")
    for f in fs:
        if os.path.isdir(os.path.join(base_path2, f)):
            continue
        path2 = os.path.join(base_path2, f)
        path4 = os.path.join(base_path4, f)

        subgraph_type = ['Conventional', 'Reasoning', 'Comparing', 'Operation', 'all']
        #subgraph_type = ['COVID-19', 'climate', 'common_wikidata', 'public health', 'medicine', 'common_wikipedia', 'scientific', 'all']
        if not READ_ONLY:
            metrics_dict = compute_score(subgraph_type, 'category', path1, path2, path3)
        else:
            metrics_dict = Metrics.read_score_dict(path4)
        table = get_score(metrics_dict, subgraph_type, path4)

        print(path2.split('/')[-1])
        print(table)



