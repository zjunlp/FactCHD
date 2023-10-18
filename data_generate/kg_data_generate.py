import itertools
import json
import os
import random
import time
import uuid
from collections import defaultdict, OrderedDict

from kopl.kopl import KoPLEngine
from loguru import logger
from tqdm import tqdm

import openai_service
from utils import prompter, create_dir
from utils.subgraph_enum import SubgraphType, get_all_type


def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError


# Pass in a list and randomly select n different elements
def random_select_n_from_list(l, n):
    if len(l) <= n:
        return l
    return random.sample(l, n)


def generate_quantitative_comparison_subgraph(engine, num, num_qualifier):
    file_name = 'graph_' + str(SubgraphType.QUANTITATIVE_COMPARISON) + '.jsonl'
    with open(os.path.join(subgraph_dir, file_name), 'w', encoding='utf-8') as fin:
        for attr_name, entity_id_index in engine.kb.attribute_inv_index.items():
            flag = False
            cid2eid = defaultdict(list)
            eid2triple = {}
            qualifier2eid_triple = {}
            for ent_id, index_list in entity_id_index.items():
                entity = engine.kb.entities[ent_id]
                attr_info = engine.kb.entities[ent_id]['attributes'][index_list[0]]
                if attr_info['value'].type == 'string':
                    flag = True
                    break

                for index in index_list:
                    attr_info = engine.kb.entities[ent_id]['attributes'][index]
                    triple = [entity['name'], attr_name, str(attr_info['value'])]
                    if 'qualifiers' not in attr_info or attr_info['qualifiers'] == {} or attr_info[
                        'qualifiers'] is None:
                        eid2triple[ent_id] = triple
                    else:
                        triple_qualifiers = []
                        # 对attr_info['qualifiers']排序
                        sorted_keys = sorted(attr_info['qualifiers'].keys())
                        qualifier_key = ""
                        for k in sorted_keys:
                            # 只选第一个
                            triple_qualifiers.append([triple, k, str(attr_info['qualifiers'][k][0])])
                            qualifier_key += str(k) + ": " + str(attr_info['qualifiers'][k][0]) + ", "
                        if qualifier_key not in qualifier2eid_triple:
                            qualifier2eid_triple[qualifier_key] = {}
                        qualifier2eid_triple[qualifier_key][ent_id] = triple_qualifiers

                for isa in entity['isA']:
                    cid2eid[isa].append(ent_id)

            if flag:
                continue

            # 普通
            for isa, ent_id_set in cid2eid.items():
                triples = []
                for ent_id in ent_id_set:
                    if ent_id not in eid2triple:
                        continue
                    triples.append([
                        [engine.kb.entities[ent_id]['name'], "instance of", engine.kb.entities[isa]['name']],
                        eid2triple[ent_id],
                    ])

                for i in range(2, min(5, len(triples)) + 1):
                    if len(triples) < i:
                        break
                    # 随机选一个
                    subgraph = []
                    triple_list = random_select_n_from_list(triples, i)
                    for it in triple_list:
                        for t in it:
                            subgraph.append(t)
                    data = {
                        "triples": subgraph,
                        "subgraph_type": str(SubgraphType.QUANTITATIVE_COMPARISON),
                        "size": len(subgraph),
                        "id": str(uuid.uuid1()),
                        "attr_name": attr_name,
                        "mark": i,
                        "is_qualifiers": False,
                    }
                    fin.write(json.dumps(data, ensure_ascii=False) + "\n")
                    num[i] = num.get(i, 0) + 1

                    # # 从triples中选出i个不同的元素所有方案
                    # l, r = 0, 100
                    # for triple_list in itertools.combinations(triples, i):
                    #     subgraph = []
                    #     for it in triple_list:
                    #         for t in it:
                    #             subgraph.append(t)
                    #     data = {
                    #         "triples": subgraph,
                    #         "subgraph_type": str(subgraph_type),
                    #         "subgraph_id": int(time.time()),
                    #         "attr_name": attr_name,
                    #         "mark": i,
                    #         "is_qualifiers": False,
                    #     }
                    #     fin.write(json.dumps(data, ensure_ascii=False) + "\n")
                    #     num[i] = num.get(i, 0) + 1
                    #     l += 1
                    #     if l >= r:
                    #         break

            # 2：qualifiers
            for qualifier_key, ent_id_qualifier_triple_list in qualifier2eid_triple.items():
                triples = []
                for isa, ent_id_set in cid2eid.items():
                    for ent_id in ent_id_set:
                        if ent_id not in ent_id_qualifier_triple_list:
                            continue
                        temp = [
                            [engine.kb.entities[ent_id]['name'], "instance of", engine.kb.entities[isa]['name']]
                        ]
                        for triple in ent_id_qualifier_triple_list[ent_id]:
                            temp.append(triple)
                        triples.append(temp)

                    for i in range(2, min(5, len(triples)) + 1):
                        if len(triples) < i:
                            break
                        # 随机选择一个
                        triple_list = random_select_n_from_list(triples, i)
                        subgraph = []
                        for it in triple_list:
                            for t in it:
                                subgraph.append(t)
                        data = {
                            "triples": subgraph,
                            "subgraph_type": str(SubgraphType.QUANTITATIVE_COMPARISON),
                            "size": len(subgraph),
                            "id": str(uuid.uuid1()),
                            "attr_name": attr_name,
                            "mark": i,
                            "is_qualifiers": True,
                        }
                        fin.write(json.dumps(data, ensure_ascii=False) + "\n")
                        num_qualifier[i] = num_qualifier.get(i, 0) + 1
                        # 从triples中选出i个不同的元素所有方案
                        # for triple_list in itertools.combinations(triples, i):
                        #     subgraph = []
                        #     for it in triple_list:
                        #         for t in it:
                        #             subgraph.append(t)
                        #     data = {
                        #         "triples": subgraph,
                        #         "subgraph_type": str(subgraph_type),
                        #         "subgraph_id": int(time.time()),
                        #         "attr_name": attr_name,
                        #         "mark": i,
                        #         "is_qualifiers": True,
                        #     }
                        #     fin.write(json.dumps(data, ensure_ascii=False) + "\n")
                        #     num_qualifier[i] = num_qualifier.get(i, 0) + 1


def generate_set_operation_subgraph(engine, num):
    # 寻找父目录
    fa_dir = os.path.dirname(subgraph_dir)
    kb = json.load(open(os.path.join(fa_dir, 'kb.json')))
    file_name = 'graph_' + str(SubgraphType.SET_OPERATION) + '.jsonl'
    with open(os.path.join(subgraph_dir, file_name), 'w', encoding='utf-8') as wf:
        # 1: other relation
        tail_relation = defaultdict(list)
        for (head_id, tail_id), idx_list in engine.kb.forward_relation_index.items():
            for idx in idx_list:
                relation = engine.kb.entities[head_id]['relations'][idx]
                tail_relation[(relation['relation'], tail_id)].append(head_id)

        keys = list(tail_relation.keys())
        set_num_limit = {2: 10000, 3: 10, 4: 10, 5: 10}
        for i in range(2, 3):
            cnt = 0
            flag = True
            # reverse keys
            random.shuffle(keys)
            for select_keys in itertools.combinations(keys, i):
                cnt += 1
                if cnt >= 1e8:
                    break
                # find intersection
                intersection = set(tail_relation[select_keys[0]])
                for key in select_keys[1:]:
                    intersection = intersection & set(tail_relation[key])
                if 5 >= len(intersection) >= 2:
                    flag = False
                    triples = []
                    for ent_id in intersection:
                        for key in select_keys:
                            triples.append(
                                [engine.kb.entities[ent_id]['name'], key[0],
                                 engine.kb.entities[key[1]]['name']]
                            )
                    wf.write(json.dumps({
                        "triples": triples,
                        "subgraph_type": str(SubgraphType.SET_OPERATION),
                        "size": len(triples),
                        "id": str(uuid.uuid1()),
                        "attr_name": [key[0] for key in select_keys],
                        "answer_size": len(intersection),
                        "set_size": i,
                    }, ensure_ascii=False) + "\n")
                    num[i] = num.get(i, 0) + 1
                    if num[i] >= set_num_limit[i]:
                        break
            if flag:
                break

        # 2: isA
        isa_cid = defaultdict(list)
        for eid, ent in kb['entities'].items():
            for isa in ent['instanceOf']:
                isa_cid[isa].append(eid)

        for i in range(2, 6):
            flag = True
            keys = list(isa_cid.keys())
            random.shuffle(keys)
            cnt = 0
            for select_keys in itertools.combinations(keys, i):
                cnt += 1
                if cnt >= 1e6:
                    break
                intersection = set(isa_cid[select_keys[0]])
                for key in select_keys[1:]:
                    intersection = intersection & set(isa_cid[key])
                if 2 <= len(intersection) <= 5:
                    flag = False
                    triples = []
                    for ent_id in intersection:
                        for key in select_keys:
                            triples.append(
                                [engine.kb.entities[ent_id]['name'], "instance of",
                                 engine.kb.entities[key]['name']]
                            )
                    data = {
                        "triples": triples,
                        "subgraph_type": str(SubgraphType.SET_OPERATION),
                        "size": len(triples),
                        "id": str(uuid.uuid1()),
                        "attr_name": ["instance of"],
                        "answer_size": len(intersection),
                        "set_size": i,
                    }
                    wf.write(json.dumps(data, ensure_ascii=False) + "\n")
                    num[i] = num.get(i, 0) + 1
                    if num[i] >= set_num_limit[i]:
                        break
            if flag:
                break


def bfs_kqa_pro(eid, subgraph, engine, max_deep, max_width, max_size, entity_set):
    # 实现一个宽度最多为max_width、深度最多为max_deep的bfs算法
    if len(subgraph) >= max_size:
        return
    que = [(eid, 0, "")]
    while len(que) > 0:
        eid, deep, fr = que.pop(0)
        if deep >= max_deep:
            continue
        ent = engine.kb.entities[eid]
        forwards = []
        add = 0
        for idx, rel_info in enumerate(ent['relations']):
            if rel_info['direction'] == 'forward':
                if fr != "":
                    if rel_info['relation'] == fr and add < 1:
                        triple = [ent['name'], rel_info['relation'], engine.kb.entities[rel_info['object']]['name']]
                        if triple in subgraph:
                            continue
                        subgraph.append(triple)
                        que.append((rel_info['object'], deep + 1, rel_info['relation']))
                        entity_set.add(rel_info['object'])
                        add += 1
                        if len(subgraph) >= max_size:
                            return
                        continue
                forwards.append(idx)
        random.shuffle(forwards)
        for idx in forwards[:min(len(forwards), max_width - add)]:
            rel_info = ent['relations'][idx]
            triple = [ent['name'], rel_info['relation'], engine.kb.entities[rel_info['object']]['name']]
            if triple in subgraph:
                continue
            if rel_info['object'] in entity_set:
                continue
            que.append((rel_info['object'], deep + 1, rel_info['relation']))
            entity_set.add(rel_info['object'])
            subgraph.append(triple)
            if len(subgraph) >= max_size:
                return


def get_most_attributes(subgraph, engine, bfs_entity_set, max_size):
    attr_inverse = defaultdict(set)
    attr_qualifiers_inverse = defaultdict(set)
    for eid in bfs_entity_set:
        ent = engine.kb.entities[eid]
        for idx, attr_info in enumerate(ent['attributes']):
            if attr_info['value'].type == 'string':
                break
            if 'qualifiers' not in attr_info or attr_info['qualifiers'] == {} or attr_info['qualifiers'] is None:
                attr_inverse[attr_info['key']].add((eid, idx))
            else:
                sorted_keys = sorted(attr_info['qualifiers'].keys())
                qualifier_key = ""
                for k in sorted_keys:
                    qualifier_key += str(k) + ": " + str(attr_info['qualifiers'][k][0]) + ", "
                attr_qualifiers_inverse[attr_info['key'] + " " + qualifier_key].add((eid, idx))
    attr_inverse = sorted(attr_inverse.items(), key=lambda x: len(x[1]), reverse=True)
    attr_qualifiers_inverse = sorted(attr_qualifiers_inverse.items(), key=lambda x: len(x[1]), reverse=True)
    for attr, attr_set in attr_inverse:
        if len(attr_set) <= 1:
            continue
        if max_size == 1:
            continue
        for eid, idx in attr_set:
            ent = engine.kb.entities[eid]
            subgraph.append([
                ent['name'],
                attr,
                str(ent['attributes'][idx]['value'])
            ])
            if len(subgraph) >= max_size:
                return
    for attr, attr_set in attr_qualifiers_inverse:
        if len(attr_set) <= 1:
            continue
        if max_size == 1:
            continue
        for eid, idx in attr_set:
            ent = engine.kb.entities[eid]
            attr = ent['attributes'][idx]
            qk = ""
            qv = ""
            for qk, qv in attr['qualifiers'].items():
                if "time" in qk:
                    break
            subgraph.append([
                [ent['name'], attr['key'], str(attr['value'])],
                qk,
                str(qv[0])
            ])
            if len(subgraph) >= max_size:
                return
    pass


def generate_all_subgraph(engine, num, num_qualifier):
    # 实现一个bfs算法
    head_set = set()
    tot = 0
    with open(os.path.join(subgraph_dir, 'graph_all.jsonl'), 'w', encoding='utf-8') as fin:
        while True:
            entity_set = set()
            for eid, ent in engine.kb.entities.items():
                if eid in engine.kb.concepts:
                    continue
                if eid in entity_set or eid in head_set:
                    continue
                entity_set.add(eid)
                head_set.add(eid)
                # 实现一个宽度为3的bfs算法
                subgraph = []
                # 3 + 3 * 3 + 3 * 3 * 3 = 39
                bfs_entity_set = {eid}
                bfs_kqa_pro(eid, subgraph, engine, 3, 3, 39, bfs_entity_set)
                if len(subgraph) <= 5:
                    continue
                entity_set = entity_set | bfs_entity_set
                get_most_attributes(subgraph, engine, bfs_entity_set, 50)
                fin.write(json.dumps({
                    "triples": list(subgraph),
                    "subgraph_type": str(SubgraphType.ALL),
                    "size": len(subgraph),
                    "id": str(uuid.uuid1()),
                    "head": engine.kb.entities[eid]['name']
                }, ensure_ascii=False) + '\n')
                num[len(subgraph)] = num.get(len(subgraph), 0) + 1
                tot += 1
                if tot >= 10000:
                    break
            break


def dfs_kqa_pro(eid, engine, subgraph, max_deep, dfs_entity_set, fr="", y=0):
    if len(subgraph) >= max_deep:
        return
    ent = engine.kb.entities[eid]
    # 筛选正向关系
    forwards = []
    add = 0
    for idx, rel_info in enumerate(ent['relations']):
        if rel_info['direction'] == 'forward':
            if rel_info['object'] in dfs_entity_set:
                continue
            if fr != "" and y == 1:
                if rel_info['relation'] == fr:
                    dfs_entity_set.add(rel_info['object'])
                    subgraph.append([ent['name'], rel_info['relation'], engine.kb.entities[rel_info['object']]['name']])
                    dfs_kqa_pro(rel_info['object'], engine, subgraph, max_deep, dfs_entity_set, rel_info['relation'], y)
                    return
            forwards.append(idx)

    # 随机选择一个forwards
    if len(forwards) == 0:
        return
    idx = random.randint(0, len(forwards) - 1)
    dfs_entity_set.add(ent['relations'][forwards[idx]]['object'])
    rel_info = ent['relations'][forwards[idx]]
    subgraph.append([ent['name'], rel_info['relation'], engine.kb.entities[rel_info['object']]['name']])
    dfs_kqa_pro(rel_info['object'], engine, subgraph, max_deep, dfs_entity_set, rel_info['relation'], y)


def get_attributes(subgraph, engine):
    eid, _ = engine.Find(subgraph[-1][2])
    ent = engine.kb.entities[eid[0]]
    for idx, attr_info in enumerate(ent['attributes']):
        if attr_info['value'].type == 'string':
            continue
        if len(attr_info['qualifiers']) > 0:
            for k, v in attr_info['qualifiers'].items():
                if "time" not in k:
                    continue
                subgraph.append([[ent['name'], attr_info['key'], str(attr_info['value'])], k, str(v[0])])
                return

    for idx, attr_info in enumerate(ent['attributes']):
        if attr_info['value'].type == 'string':
            continue
        if len(attr_info['qualifiers']) > 0:
            continue
        subgraph.append([ent['name'], attr_info['key'], str(attr_info['value'])])
        return

    for idx, attr_info in enumerate(ent['attributes']):
        if len(attr_info['qualifiers']) > 0:
            continue
        subgraph.append([ent['name'], attr_info['key'], str(attr_info['value'])])
        return

    for idx, attr_info in enumerate(ent['attributes']):
        for k, v in attr_info['qualifiers'].items():
            subgraph.append([[ent['name'], attr_info['key'], str(attr_info['value'])], k, str(v[0])])
            return


def generate_multi_hop_reasoning_subgraph(engine):
    # 实现一个dfs算法
    head_set = set()
    tot = 0
    file_name = 'graph_' + str(SubgraphType.MULTI_HOP_REASONING) + '.jsonl'
    with open(os.path.join(subgraph_dir, file_name), 'w', encoding='utf-8') as fin:
        while True:
            entity_set = set()
            for eid, ent in engine.kb.entities.items():
                if eid in engine.kb.concepts:
                    continue
                if eid in entity_set or eid in head_set:
                    continue
                entity_set.add(eid)
                head_set.add(eid)
                subgraph = []
                dfs_entity_set = {eid}
                # 随机生成数字 1 - 6
                x = random.randint(2, 5)
                y = random.randint(0, 1)
                dfs_kqa_pro(eid, engine, subgraph, x, dfs_entity_set, "", y)
                if len(subgraph) <= 1:
                    continue

                entity_set = entity_set | dfs_entity_set
                get_attributes(subgraph, engine)
                fin.write(json.dumps({
                    "triples": list(subgraph),
                    "subgraph_type": str(SubgraphType.MULTI_HOP_REASONING),
                    "size": len(subgraph),
                    "id": str(uuid.uuid1()),
                    "head": engine.kb.entities[eid]['name']
                }, ensure_ascii=False) + '\n')
                tot += 1
                if tot >= 20000:
                    break
            break
    pass


def generate_subgraph_kqa_pro(subgraph_types):
    engine = KoPLEngine(json.load(open(os.path.join(data_dir, 'kb.json'))))
    for subgraph_type in subgraph_types:
        num = OrderedDict()
        num_qualifier = OrderedDict()
        logger.info('subgraph_type: {} start!'.format(subgraph_type))
        if subgraph_type == SubgraphType.MULTI_HOP_REASONING:
            generate_multi_hop_reasoning_subgraph(engine)
        elif subgraph_type == SubgraphType.QUANTITATIVE_COMPARISON:
            generate_quantitative_comparison_subgraph(engine, num, num_qualifier)
        elif subgraph_type == SubgraphType.SET_OPERATION:
            generate_set_operation_subgraph(engine, num)
        elif subgraph_type == SubgraphType.ALL:
            generate_all_subgraph(engine, num, num_qualifier)
        logger.info('subgraph_type: {} end!'.format(subgraph_type))


def generate_qa_use_chatgpt(data, subgraph_type, suffix, domain):
    s = ""
    for t in data['triples']:
        s += '[\"' + str(t[0]) + '\", \"' + t[1] + "\", \"" + t[2] + "\"], "
    s = s.strip()
    s = s[:-1]
    if subgraph_type == SubgraphType.MULTI_HOP_REASONING:
        if domain == "medicine":
            prompt = prompter.KG_QA_MEDICINE_MULTI_HOP_REASONING_DATA.format(triples=s)
        else:
            prompt = prompter.KG_QA_MULTI_HOP_REASONING_DATA.format(triples=s)
    elif subgraph_type == SubgraphType.QUANTITATIVE_COMPARISON:
        # 随机选择在0和1，选择一个数字
        if random.randint(0, 1) == 0:
            prompt = prompter.KG_QA_QUANTITATIVE_COMPARISON_SHORT_DATA.format(triples=s)
        else:
            prompt = prompter.KG_QA_QUANTITATIVE_COMPARISON_LONG_DATA.format(triples=s)
    elif subgraph_type == SubgraphType.SET_OPERATION:
        prompt = prompter.KG_QA_SET_OPERATION_DATA.format(triples=s)
    else:
        prompt = prompter.KG_QA_ALL_DATA.format(triples=s)
    response = openai_service.get_response_retry(prompt, temperature=0, max_tokens=4000)
    if response is None:
        logger.error("response is None")
        logger.info("prompt:" + prompt)
        return
    responses = response.strip().split("\n")

    ans = []
    res = {}
    file_name = os.path.join(qa_dir, domain + "_chatgpt_" + str(subgraph_type) + "_qa_" + suffix + ".jsonl")
    if not os.path.exists(file_name):
        with open(file_name, "w", encoding='utf-8') as w:
            w.write("")

    with open(file_name, "a", encoding='utf-8') as w:
        for r in responses:
            if r.startswith("<Question>:") or "<Question>:" in r:
                res['question'] = r.split("<Question>:")[1].strip()
            elif "Question:" in r:
                res['question'] = r.split("Question:")[1].strip()
            elif r.startswith("<Correct answer>:") or "<Correct answer>:" in r:
                res['correct_answer'] = r.split("<Correct answer>:")[1].strip()
            elif "Correct answer:" in r:
                res['correct_answer'] = r.split("Correct answer:")[1].strip()
            elif r.startswith("<Hallucinated answer>:") or "<Hallucinated answer>:" in r:
                res['hallucinated_answer'] = r.split("<Hallucinated answer>:")[1].strip()
            elif "Hallucinated answer:" in r:
                res['hallucinated_answer'] = r.split("Hallucinated answer:")[1].strip()
            elif r.startswith("<Only used knowledge>:") or "<Only used knowledge>:" in r:
                res['used_knowledge'] = r.split("<Only used knowledge>:")[1].strip()
            elif "Only used knowledge:" in r:
                res['used_knowledge'] = r.split("Only used knowledge:")[1].strip()
            t = 3
            if subgraph_type == SubgraphType.ALL:
                t = 4
            if len(res) >= t:
                ans.append(res)
                # 0:未验证, 1:chatgpt核验正确, 2:chatgpt核验错误, 3:人工核验正确, 4:人工核验错误
                res['verification'] = 0
                if 'used_knowledge' not in res:
                    res['used_knowledge'] = data['triples']
                res['subgraph_id'] = data["id"]
                res['old_size'] = data['size']
                res['type'] = str(subgraph_type)
                res['response'] = response
                if "mark" in data:
                    res['mark'] = data['mark']
                if "is_qualifiers" in data:
                    res['is_qualifiers'] = data['is_qualifiers']
                if "answer_size" in data:
                    res['answer_size'] = data['answer_size']
                if "set_size" in data:
                    res['set_size'] = data['set_size']
                res['id'] = str(uuid.uuid1())
                w.write(json.dumps(res, ensure_ascii=False) + "\n")
                res = {}

    if len(ans) == 0:
        logger.error("response: " + response)
        logger.error("data:" + str(data))
        return None

    return ans


def generate_qa(subgraph_types, limit, condition_map, suffix="test", domain="common"):
    qa_backup_dir = os.path.join(data_dir, "qa_backup")
    if not os.path.exists(qa_backup_dir):
        os.mkdir(qa_backup_dir)

    for subgraph_type in subgraph_types:
        save_path = str(subgraph_type) + "_qa_backup.jsonl"
        logger.info(save_path + ":开始{}生成问题!".format(str(subgraph_type)))

        id_set = set()
        if not os.path.exists(os.path.join(qa_backup_dir, save_path)):
            with open(os.path.join(qa_backup_dir, save_path), "w", encoding='utf-8') as f:
                f.write("")

        tot = 0
        with open(os.path.join(qa_backup_dir, save_path), "r", encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    if 'id' in data:
                        id_set.add(data['id'])
                    flag = True
                    for ck, cv in condition_map.items():
                        if ck not in data or data[ck] != cv:
                            flag = False
                    if flag:
                        tot += 1

        logger.info("限制条件：" + str(condition_map))
        logger.info("已存在数据量: " + str(tot))

        if tot >= limit:
            logger.info("已存在数据量大于等于" + str(limit) + "，跳过")
            continue

        wf = open(os.path.join(qa_backup_dir, save_path), "a", encoding='utf-8')
        i = 0
        graph_file_name = "graph_" + str(subgraph_type) + "_process.jsonl"
        with open(os.path.join(subgraph_dir, graph_file_name), "r", encoding='utf-8') as f:
            for line in f:
                if tot >= limit:
                    break
                i += 1
                if i % 50 == 0:
                    logger.info("条件:" + str(condition_map) + ", 已使用{i}个子图, 生成了{tot}个问题, 共{limit}"
                                .format(i=i, tot=tot, limit=limit))
                line = line.strip()
                if line:
                    data = json.loads(line)
                    if data["id"] in id_set:
                        continue

                    flag = True

                    for ck, cv in condition_map.items():
                        if ck not in data or data[ck] != cv:
                            flag = False
                            break
                    if not flag:
                        continue

                    response = generate_qa_use_chatgpt(data, subgraph_type, suffix, domain)
                    if response is None:
                        continue
                    data["response"] = response
                    data["type"] = str(subgraph_type)
                    wf.write(json.dumps(data, ensure_ascii=False) + "\n")
                    tot += len(response)

        wf.close()
        logger.info(save_path + "生成完成: 使用{i}个子图, 生成了{tot}个问题".format(i=i, tot=tot))


def generate_all_use_chatgpt(data):
    s = ""
    if isinstance(data['used_knowledge'], list):
        for t in data['used_knowledge']:
            s += '[\"' + str(t[0]) + '\", \"' + t[1] + "\", \"" + t[2] + "\"], "
        s = s.strip()
        s = s[:-1]
    else:
        s = data['used_knowledge'].strip()
        while s.startswith("[") is False:
            s = s[1:]
        if s.startswith("[["):
            s = s[1:]
        while s.endswith("]") is False:
            s = s[0:-1]
        if s.endswith("]]"):
            s = s[0:-1]
        try:
            triple = json.loads("[" + s + "]")
        except:
            triple = s
            logger.error("解析错误，triple: " + s)
        data['used_knowledge'] = triple

    # 0,1随机生成一个数
    random.seed(time.time())
    if random.random() < 0.7:
        data['label'] = "NON-FACTUAL"
        data['chatgpt_response'] = data['hallucinated_answer']
    else:
        data['label'] = "FACTUAL"
        data['chatgpt_response'] = data['correct_answer']

    if data['type'] == str(SubgraphType.MULTI_HOP_REASONING):
        prompt = prompter.KG_GENERATE_DATA_MULTI_HOP.format(label=data['label'], question=data['question'],
                                                            answer=data['chatgpt_response'], evidence=s)
    else:
        prompt = prompter.KG_GENERATE_DATA.format(label=data['label'], question=data['question'],
                                                  answer=data['chatgpt_response'], evidence=s)

    response = openai_service.get_response_retry(prompt, temperature=0, max_tokens=4000)
    if response is None:
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

    data['reason'] = response
    return data


def generate_all(subgraph_types, limit, suffix="test", domain="common"):
    for subgraph_type in subgraph_types:
        save_path = domain + "_" + str(subgraph_type) + "_all_" + suffix + ".jsonl"

        id_set = set()
        if not os.path.exists(os.path.join(all_dir, save_path)):
            with open(os.path.join(all_dir, save_path), "w", encoding='utf-8') as f:
                f.write("")

        with open(os.path.join(all_dir, save_path), "r", encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    if 'id' in data:
                        id_set.add(data['id'])

        logger.info("已存在数据量: " + str(len(id_set)))

        if len(id_set) >= limit:
            logger.info("已存在数据量大于等于" + str(limit) + "，跳过")
            continue

        wf = open(os.path.join(all_dir, save_path), "a", encoding='utf-8')
        i = 0
        tot = len(id_set)
        qa_file_name = domain + "_chatgpt_" + str(subgraph_type) + "_qa_" + suffix + ".jsonl"
        with open(os.path.join(qa_dir, qa_file_name), "r", encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    if tot >= limit:
                        break
                    i += 1
                    if i % 50 == 0:
                        logger.info(save_path + ":已使用{i}个问题, 生成了{tot}个证据链".format(i=i, tot=tot))
                    data = json.loads(line)
                    if data["id"] in id_set:
                        continue
                    # if data["old_size"] >= 6:
                    #     continue
                    response = generate_all_use_chatgpt(data)
                    if response is None:
                        continue
                    data["domain"] = domain
                    wf.write(json.dumps(data, ensure_ascii=False) + "\n")
                    tot += 1
        wf.close()
        logger.info(save_path + ":已使用{i}个问题, 生成了{tot}个证据链".format(i=i, tot=tot))


def statistics_filter_subgraph(subgraph_types, is_filter=False):
    # filter
    for subgraph_type in subgraph_types:
        if is_filter is False:
            continue
        triple_set = set()
        datas = []
        graph_file_name = 'graph_' + str(subgraph_type) + '.jsonl'
        with open(os.path.join(subgraph_dir, graph_file_name), "r", encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    # 排序
                    triples = data['triples']
                    for i in range(len(triples)):
                        if isinstance(triples[i][0], list):
                            triples[i][0] = str(triples[i][0])
                        triples[i] = tuple(triples[i])
                    triples = tuple(sorted(triples))
                    # 排序

                    if triples in triple_set:
                        continue
                    triple_set.add(triples)
                    datas.append(data)

        # shuffle
        random.shuffle(datas)
        # save
        graph_file_name = 'graph_' + str(subgraph_type) + '_process.jsonl'
        with open(os.path.join(subgraph_dir, graph_file_name), "w", encoding='utf-8') as f:
            for data in datas:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")

    for subgraph_type in subgraph_types:
        tot = 0
        subgraph_num = defaultdict(int)
        set_size_num = defaultdict(int)
        answer_size_num = defaultdict(int)
        mark_num = defaultdict(int)
        graph_file_name = 'graph_' + str(subgraph_type) + '_process.jsonl'
        with open(os.path.join(subgraph_dir, graph_file_name), "r", encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    tot += 1
                    subgraph_num[data['size']] += 1
                    if 'set_size' in data:
                        set_size_num[data['set_size']] += 1
                    if 'answer_size' in data:
                        answer_size_num[data['answer_size']] += 1
                    if 'mark' in data:
                        mark_num[data['mark']] += 1

        print("subgraph_type: " + str(subgraph_type))
        # 按key排序输出
        subgraph_num = sorted(subgraph_num.items(), key=lambda x: x[0])
        set_size_num = sorted(set_size_num.items(), key=lambda x: x[0])
        answer_size_num = sorted(answer_size_num.items(), key=lambda x: x[0])
        mark_num = sorted(mark_num.items(), key=lambda x: x[0])
        print("subgraph_num: " + str(subgraph_num))
        print("set_size_num: " + str(set_size_num))
        print("answer_size_num: " + str(answer_size_num))
        print("mark_num: " + str(mark_num))
        print("tot: " + str(tot))
        print("----------------------------------")


def pool(suffix):
    datas = []
    for subgraph_type in [SubgraphType.MULTI_HOP_REASONING, SubgraphType.SET_OPERATION,
                          SubgraphType.QUANTITATIVE_COMPARISON, SubgraphType.ALL]:
        file_name = str(subgraph_type) + '_all_' + suffix + '.jsonl'
        with open(os.path.join(data_dir, file_name), "r", encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    temp = json.loads(line)
                    data = {
                        'id': temp['id'],
                        'label': temp['label'],
                        'question': temp['question'],
                        'response': temp['chatgpt_response'],
                        'evidence': temp['used_knowledge'],
                        'reason': temp['reason'],
                        'type': "triple",
                        'subgraph_type': temp['type']
                    }
                    if 'mark' in temp:
                        data['comparison_size'] = temp['mark']
                    if 'set_size' in temp:
                        data['set_size'] = temp['set_size']
                    if 'answer_size' in temp:
                        data['answer_size'] = temp['answer_size']
                    datas.append(data)
    # 随机打乱
    random.shuffle(datas)
    # 保存
    file_name = 'kg_' + suffix + '.jsonl'
    with open(os.path.join(data_dir, file_name), "w", encoding='utf-8') as f:
        for data in datas:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")


# Statistical data
def statistics_data(file):
    print(file)
    label_map = defaultdict(int)
    domain_map = defaultdict(int)
    subgraph_type = defaultdict(int)
    evidence_size = defaultdict(int)
    type_map = defaultdict(int)
    category = defaultdict(int)
    deep_label = {}
    deep_evidence_size = {}
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                if "subgraph_type" in data:
                    subgraph_type[data['subgraph_type']] += 1
                if isinstance(data['evidence'], str) is False:
                    evidence_size[str(len(data['evidence']))] += 1
                type_map[data['type']] += 1
                label_map[data['label']] += 1
                domain_map[data['domain']] += 1
                category[data['category']] += 1
                if 'domain:' + data['domain'] not in deep_label:
                    deep_label['domain:' + data['domain']] = defaultdict(int)
                if 'type:' + data['type'] not in deep_label:
                    deep_label['type:' + data['type']] = defaultdict(int)

                deep_label['domain:' + data['domain']][data['label']] += 1
                deep_label['type:' + data['type']][data['label']] += 1

                if 'domain:' + data['domain'] not in deep_evidence_size:
                    deep_evidence_size['domain:' + data['domain']] = defaultdict(int)
                if 'type:' + data['type'] not in deep_evidence_size:
                    deep_evidence_size['type:' + data['type']] = defaultdict(int)

                deep_evidence_size['domain:' + data['domain']][str(len(data['evidence']))] += 1
                deep_evidence_size['type:' + data['type']][str(len(data['evidence']))] += 1

    # 格式化输出
    print("label_map:")
    for k, v in label_map.items():
        print(k, v)
    print()

    print("type_map:")
    for k, v in type_map.items():
        print(k, v)
    print()

    print("category:")
    for k, v in category.items():
        print(k, v)
    print()

    print("domain_map:")
    for k, v in domain_map.items():
        print(k, v)
    print()
    # 按照key排序
    deep = sorted(deep_label.items(), key=lambda x: x[0])
    for k, v in deep:
        print(k, v)
    print()

    print("subgraph_type:")
    for k, v in subgraph_type.items():
        print(k, v)
    print()

    print("evidence_size:")
    # 排序deep_evidence_size
    deep_evidence_size = sorted(deep_evidence_size.items(), key=lambda x: x[0])
    for k, v in deep_evidence_size:
        print(k, v)
    print()

    #
    # print("evidence_size:")
    # # 排序evidence_size
    # evidence_size = sorted(evidence_size.items(), key=lambda x: int(x[0]))
    # for k, v in evidence_size:
    #     print(k, v)

    print("sum:")
    print(sum(label_map.values()))
    print("------------------")
    print()


data_dir = 'data/dataset/Wikidata15k'
subgraph_dir = data_dir + "/subgraph"
qa_dir = "output/qa"
all_dir = "output/all"


def init():
    create_dir(data_dir)
    create_dir(subgraph_dir)
    create_dir(qa_dir)
    create_dir(all_dir)


if __name__ == "__main__":
    # 0. init
    data_dir = 'data/dataset/Wikidata15k'
    subgraph_dir = data_dir + "/subgraph"
    qa_dir = "output/kg/qa"
    all_dir = "output/kg/all"
    init()

    # 1. generate subgraph and filter
    generate_subgraph_kqa_pro(get_all_type())
    statistics_filter_subgraph(get_all_type(), True)

    # 2. MULTI_HOP_REASONING
    num_limit = {"2": 1, "3": 0, "4": 0, "5": 0, "6": 0}
    for i in range(2, 7):
        generate_qa(subgraph_types=[SubgraphType.MULTI_HOP_REASONING], limit=num_limit[str(i)],
                    condition_map={"size": i})
    generate_all([SubgraphType.MULTI_HOP_REASONING], 1)

    # 3. SET_OPERATION
    generate_qa(subgraph_types=[SubgraphType.SET_OPERATION], limit=1, condition_map={}, suffix="train")
    generate_all(subgraph_types=[SubgraphType.SET_OPERATION], limit=1, suffix="train")

    # 4. QUANTITATIVE_COMPARISON
    num_limit = {"2": 1, "3": 0, "4": 0, "5": 0}
    # "mark" represents the quantity of objects to be compared
    # "is_qualifiers" indicates whether qualifiers should be used, such as comparing the heights of Yao Ming's wife and LeBron James' wife.
    for mark in range(2, 6):
        condition = {"mark": mark, "is_qualifiers": False}
        generate_qa(subgraph_types=[SubgraphType.QUANTITATIVE_COMPARISON], limit=num_limit[str(mark)],
                    condition_map=condition, suffix="train")
        condition = {"mark": mark, "is_qualifiers": True}
        generate_qa(subgraph_types=[SubgraphType.QUANTITATIVE_COMPARISON], limit=num_limit[str(mark)],
                    condition_map=condition, suffix="train")
    generate_all(subgraph_types=[SubgraphType.QUANTITATIVE_COMPARISON], limit=1, suffix="train")

    # 5. ALL
    condition = {}
    generate_qa([SubgraphType.ALL], 1, condition)
    generate_all([SubgraphType.ALL], 1)
