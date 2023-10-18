import json
import re
import time

import rouge
from collections import Counter
from nltk.translate import bleu_score as nltkbleu
#from bert_score import score
# from sentence_transformers import util

re_art = re.compile(r'\b(a|an|the)\b')
re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')
ROUGE_METRICS_MEASURES = {'r', 'f', 'p'}


def normalize_answer(s):
    s = s.lower()
    s = re_punc.sub(' ', s)
    s = re_art.sub(' ', s)
    s = ' '.join(s.split())
    return s


def split_response(s):
    s = s.strip()
    res = {}
    # s = s.lower()
    if s.startswith("non-factual") or s.startswith("NON-FACTUAL"):
        res["label"] = "non-factual"
        s = s[len("non-factual"):]
    elif s.startswith("factual") or s.startswith("FACTUAL"):
        res["label"] = "factual"
        s = s[len("factual"):]
    else:
        i = 0
        while i < len(s) and not str.isalpha(s[i]):
            i += 1
        if i < len(s):
            res["label"] = "unknown"
            s = s[i:]
    s = s.lstrip(" .,;:'\"\n")
    sentences = s.split(".")
    # 列表去除空字符串
    sentences = list(filter(None, sentences))
    res["body"] = ""
    res["head"] = ""
    res["tail"] = ""
    if len(sentences) >= 3:
        res["head"] = sentences[0].strip() + ". "
        res["tail"] = sentences[-1].strip() + ". "
        for i in range(1, len(sentences) - 1):
            res["body"] += sentences[i].strip() + ". "
    else:
        # for i in range(0, len(sentences)):
        #     res["body"] += sentences[i].strip() + ". "
        if len(sentences) >= 1:
            if sentences[0].strip().startswith("The") and sentences[0].strip().startswith("the"):
                res['head'] = sentences[0].strip() + ". "
            else:
                res['body'] = sentences[0].strip() + ". "
        if len(sentences) >= 2:
            if sentences[1].strip().startswith("Therefore") or sentences[1].strip().startswith("therefore"):
                res['tail'] = sentences[1].strip() + ". "
            else:
                res['body'] += sentences[1].strip() + ". "
    return res


# POSITIVE = 'factual'
# NEGATIVE = 'non factual'
POSITIVE = 'non factual'
NEGATIVE = 'factual'


def safe_div(a, b):
    if b == 0.:
        return 0.
    else:
        return round(a / b, 5)


class Metrics:
    def __init__(self, title='', metrics_list='num,fail,acc,p,r,f1,rouge_L,sentence_body,sentence_head,sentence_tail,sentence_all,sentence_transformers,score'):
        self.num = 0
        self.acc = 0.0
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0
        self.fact_precision = 0.0
        self.fact_recall = 0.0
        self.fact_f1 = 0.0
        self.body_fact_precision = 0.0
        self.body_fact_recall = 0.0
        self.body_fact_f1 = 0.0
        self.blue_k = {'blue_1': 0.0, 'blue_2': 0.0, 'blue_3': 0.0, 'blue_4': 0.0}
        self.rouge_1 = 0.0
        self.rouge_2 = 0.0
        self.rouge_L = 0.0
        self.rouge_W = 0.0
        self.head_rouge_L = 0.0
        self.tail_rouge_L = 0.0
        self.fail = 0
        self.bert_score = 0.0
        self.predictions_body = []
        self.labels_body = []
        self.predictions_head = []
        self.labels_head = []
        self.predictions_tail = []
        self.labels_tail = []
        self.predictions = []
        self.labels = []
        self.title = title
        self.metrics_list = self.init_metrics(metrics_list)

    def init_metrics(self, metrics_list):
        return set(metrics_list.split(","))

    def get_part(self, s):
        if s.startswith(POSITIVE):
            return s[len(POSITIVE):], POSITIVE
        elif s.startswith(NEGATIVE):
            return s[len(NEGATIVE):], NEGATIVE
        else:
            self.fail += 1
            #print(s)
            return s, ""

    def acc_compute(self, prediction: str, label: str):
        if prediction is None or label is None:
            raise TypeError
        if prediction == label:
            return 1
        return 0

    def _prec_recall_f1_score(self, pred_items, gold_items):
        common = Counter(gold_items) & Counter(pred_items)
        num_same = sum(common.values())
        if num_same == 0:
            return 0, 0, 0
        precision = 1.0 * num_same / len(pred_items)
        recall = 1.0 * num_same / len(gold_items)
        f1 = (2 * precision * recall) / (precision + recall)
        return precision, recall, f1

    def fact_f1_compute(self, prediction: str, label: str, expose_p_and_r=False):
        p_tokens = prediction.split()
        precision, recall, f1 = self._prec_recall_f1_score(p_tokens, label.split())

        if expose_p_and_r:
            return precision, recall, f1
        else:
            return f1

    def cls_f1_compute(self, prediction: str, label: str):
        if prediction == POSITIVE and label == POSITIVE:
            self.tp += 1
        elif prediction == POSITIVE and label == NEGATIVE:
            self.fp += 1
        elif prediction == NEGATIVE and label == POSITIVE:
            self.fn += 1

    def blue_compute(self, prediction: str, label: str, k: int = 4):
        weights = [1 / k for _ in range(k)]

        score = nltkbleu.sentence_bleu(
            [label.split(" ")],
            prediction.split(" "),
            smoothing_function=nltkbleu.SmoothingFunction(epsilon=1e-12).method1,
            weights=weights,
        )
        return score

    def rouge_compute(self, prediction: str, label: str, measure: str = 'r'):
        measure = measure.lower()
        assert (
                measure in ROUGE_METRICS_MEASURES
        ), "Use one of recall 'r' (default), f1 'f', or precision 'p'."

        _evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'], max_n=2)
        # try:
        score = _evaluator.get_scores(prediction, label)
        # except LookupError:
        #     raise LookupError

        scores_rouge1 = score['rouge-1'][measure]
        scores_rouge2 = score['rouge-2'][measure]
        scores_rougeL = score['rouge-l'][measure]
        scores_rougeW = score['rouge-w'][measure]

        return scores_rouge1, scores_rouge2, scores_rougeL, scores_rougeW

    #def bert_score_compute(self, predictions: list, labels: list):
        #if len(predictions) == 0 or len(labels) == 0:
        #    return 0.0
        #P, R, F1 = score(predictions, labels, lang="en", verbose=False, batch_size=512)
        #return F1.mean().item()

    # def sentence_transformers_compute(self, predictions: list, labels: list):
    #     if len(predictions) == 0 or len(labels) == 0:
    #         return 0.0
    #     # time.sleep(0.1)
    #     model = SentenceTransformer('/home/chenxiang/sentence-transformers/all-MiniLM-L6-v2')
        
    #     embeddings1 = model.encode(predictions, convert_to_tensor=True, batch_size=512)
    #     embeddings2 = model.encode(labels, convert_to_tensor=True, batch_size=512)
    #     cosine_scores = util.cos_sim(embeddings1, embeddings2)
    #     sum = 0.0
    #     for i in range(len(cosine_scores)):
    #         sum += cosine_scores[i][i].item()
    #     return sum / len(cosine_scores)


    def evaluate_response(self, prediction, label) -> None:
        if prediction is not None:
            old_prediction = prediction
            old_label = label
            prediction = normalize_answer(prediction)
            label = normalize_answer(label)
            pred_sent, pred_flag = self.get_part(prediction)
            label_sent, label_flag = self.get_part(label)
            self.num += 1
            acc = self.acc_compute(pred_flag, label_flag)
            self.acc += acc
            self.cls_f1_compute(pred_flag, label_flag)

            fact_precision, fact_recall, fact_f1 = self.fact_f1_compute(
                pred_sent, label_sent, expose_p_and_r=True
            )
            self.fact_precision += fact_precision
            self.fact_recall += fact_recall
            self.fact_f1 += fact_f1

            for k in range(1, 5):
                self.blue_k[f"blue_{k}"] += self.blue_compute(pred_sent, label_sent, k)

            r1, r2, rL, rW = self.rouge_compute(pred_sent, label_sent)
            self.rouge_1 += r1
            self.rouge_2 += r2
            self.rouge_L += rL
            self.rouge_W += rW
            prediction_res = split_response(old_prediction)
            label_res = split_response(old_label)

            body_fact_precision, body_fact_recall, body_fact_f1 = self.fact_f1_compute(
                prediction_res['body'], label_res['body'], expose_p_and_r=True
            )
            self.body_fact_precision += body_fact_precision
            self.body_fact_recall += body_fact_recall
            self.body_fact_f1 += body_fact_f1
            _, _, head_rL, _ = self.rouge_compute(prediction_res['head'], label_res['head'])
            _, _, tail_rL, _ = self.rouge_compute(prediction_res['tail'], label_res['tail'])
            self.head_rouge_L += head_rL
            self.tail_rouge_L += tail_rL
            # self.predictions_body.append(prediction_res['body'])
            # self.labels_body.append(label_res['body'])
            # self.predictions_head.append(prediction_res['head'])
            # self.labels_head.append(label_res['head'])
            # self.predictions_tail.append(prediction_res['tail'])
            # self.labels_tail.append(label_res['tail'])
            # self.predictions.append(prediction_res["head"] + prediction_res["body"] + prediction_res["tail"])
            # self.labels.append(label_res["head"] + label_res["body"] + label_res["tail"])


    @staticmethod
    def write_score_dict(path, metrics_dict_all):
        json.dump(metrics_dict_all, open(path, 'w'))
    
    def read_score_dict(cls, path):
        metrics_dict_all = json.load(open(path, 'r'))
        score_all = {}
        for key, value in metrics_dict_all.items():
            score = {}
            for key, value in value.items():
                if key in cls.metrics_list:
                    score[key] = value
        return score_all

    def report(self):
        precision = safe_div(self.tp, self.tp + self.fp)
        recall = safe_div(self.tp, self.tp + self.fn)
        f1 = 2 * safe_div(precision * recall, precision + recall)
        time.sleep(0.1)
        all_score = {
            'num': self.num,
            'fail': self.fail,
            'acc': safe_div(self.acc, self.num),
            'p': precision,
            'r': recall,
            'f1': f1,
            # 'fact_p': safe_div(self.fact_precision, self.num),
            # 'fact_r': safe_div(self.fact_recall, self.num),
            'fact_f1': safe_div(self.fact_f1, self.num),
            'body_fact_f1': safe_div(self.body_fact_f1, self.num),
            # 'blue_1': safe_div(self.blue_k['blue_1'], self.num),
            # 'blue_2': safe_div(self.blue_k['blue_2'], self.num),
            # 'blue_3': safe_div(self.blue_k['blue_3'], self.num),
            # 'blue_4': safe_div(self.blue_k['blue_4'], self.num),
            # 'rouge_1': safe_div(self.rouge_1, self.num),
            # 'rouge_2': safe_div(self.rouge_2, self.num),
            'rouge_L': safe_div(self.rouge_L, self.num),
            'head_rouge_L': safe_div(self.head_rouge_L, self.num),
            'tail_rouge_L': safe_div(self.tail_rouge_L, self.num),
            # 'rouge_W': safe_div(self.rouge_W, self.num),
            #'bert_score': self.bert_score_compute(self.predictions, self.labels),
            # 'sentence_body': round(self.sentence_transformers_compute(self.predictions_body, self.labels_body), 5),
            # 'sentence_head': round(self.sentence_transformers_compute(self.predictions_head, self.labels_head), 5),
            # 'sentence_tail': round(self.sentence_transformers_compute(self.predictions_tail, self.labels_tail), 5),
            # 'sentence_all': round(self.sentence_transformers_compute(self.predictions, self.labels), 5),

        }
        # all_score['sentence_transformers'] = round(0.2 * all_score['sentence_head'] + 0.6 * all_score["sentence_body"] + 0.2 * all_score["sentence_tail"], 5)
        all_score['score'] = round(0.15 * all_score['head_rouge_L'] 
        +0.15 * all_score['tail_rouge_L']
        + 0.7 * all_score['body_fact_f1'], 4)

        score = {}
        for key, value in all_score.items():
            if key in self.metrics_list:
                score[key] = value
        return score, all_score

    def expend(self, convention_m):
        for k, v in convention_m.__dict__.items():
            if k == 'bert_score_cache' or k == 'bert_score_cache_file':
                continue
            if isinstance(v, dict):
                for k1, v1 in v.items():
                    self.__dict__[k][k1] += v1
            else:
                self.__dict__[k] += v