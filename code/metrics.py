## UTILS :
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import nltk
from nltk.tokenize import TweetTokenizer
from data_processing import dataset
from torch.utils.data import DataLoader
from calcul_moverscore import word_mover_score

## METRICS MODULES :
import torch
import timeit
from torchmetrics.functional import bleu_score

from torchmetrics.functional import sacre_bleu_score

try :
    from nltk.translate.meteor_score import meteor_score
except :
    nltk.download('wordnet')
    nltk.download('omw-1.4')

from nltk.translate import nist_score

from torchmetrics.functional import word_error_rate

from nltk.translate.chrf_score import sentence_chrf

from torchmetrics.functional import translation_edit_rate, chrf_score, translation_edit_rate

from torchmetrics.functional import sacre_bleu_score
from torchmetrics.functional.text.rouge import rouge_score
from torchmetrics.functional.text.bert import bert_score

from nlg_eval_via_simi_measures import bary_score
from nlg_eval_via_simi_measures import depth_score
from nlg_eval_via_simi_measures import infolm

from rouge_metric import PyRouge

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

metric_call_bary_score = bary_score.BaryScoreMetric(
    model_name = 'bert-base-uncased',
    use_idfs = False
    )

metric_call_depth_score = depth_score.DepthScoreMetric(
    model_name = 'bert-base-uncased',
    considered_measure='wasserstein'
)

metric_call_info_lm = infolm.InfoLM(
    measure_to_use='fisher_rao', 
    temperature=1.5, 
    use_idf_weights=False
)

from transformers.utils import logging

def BLEU(
    reference : str,
    candidate : str
):
    
    reference = [[reference]]
    candidate = [candidate]
    
    return bleu_score(preds = candidate, target = reference, n_gram = 1).item()

def Sacre_BLEU(
    reference : str,
    candidate : str
):
    
    reference = [[reference]]
    candidate = [candidate]
    
    return sacre_bleu_score(candidate, reference, n_gram = 1, lowercase=True, smooth='exp').item()
    
def ROUGE_1(
    reference : str,
    candidate : str
):
    
    reference = [reference]
    
    return rouge_score(candidate, reference, use_stemmer = True, rouge_keys = 'rouge1')['rouge1_fmeasure'].item()

def ROUGE_L(
    reference : str,
    candidate : str
):
    
    reference = [reference]
    
    return rouge_score(candidate, reference, use_stemmer = True, rouge_keys = 'rougeL')['rougeL_fmeasure'].item()

def ROUGE_S4(
    reference : str,
    candidate : str
):
    
    reference = [[reference]]
    candidate = [candidate]
    
    rouge = PyRouge(rouge_s=True, skip_gap=4)

    return rouge.evaluate(candidate, reference)['rouge-s4']['f']

def CHRF(
    reference : str,
    candidate : str
):
    
    hyp = TweetTokenizer().tokenize(candidate)
    ref = TweetTokenizer().tokenize(reference)
    
    return sentence_chrf(ref, hyp)

def CHRF_1(
    reference : str,
    candidate : str
):
    
    reference = [reference]
    
    return chrf_score(candidate, reference, beta = 1, lowercase = True).item()

def CHRF_pp(
    reference : str,
    candidate : str
):
    
    reference = [reference]
    
    return chrf_score(candidate, reference, n_word_order = 2, lowercase = True).item()

def METEOR(
    reference : str,
    candidate : str
):
    
    hyp = TweetTokenizer().tokenize(candidate)
    ref = TweetTokenizer().tokenize(reference)
    
    return meteor_score([ref], hyp)


def WER(
    reference : str,
    candidate : str
):
    
    reference = [reference]
    
    wer = word_error_rate(preds = candidate, target = reference).item()
    
    return 1 - wer

def TER(
    reference : str,
    candidate : str
):
    
    reference = [reference]
    
    ter = translation_edit_rate(candidate, reference, asian_support = True, lowercase = True).item()
    
    return 1 - ter

def INFO_LM(
    reference : str,
    candidate : str
):
    
    reference = [reference]
    candidate = [candidate]
    
    metric_call_info_lm.prepare_idfs(reference, candidate)
    return metric_call_info_lm.evaluate_batch(reference, candidate)['fisher_rao'][0]


def BERT(
    reference : str,
    candidate : str
):
    reference = [reference]
    candidate = [candidate]
    
    return bert_score(candidate, reference, model_name_or_path = 'bert-base-uncased', idf = False)['f1']


def BARY(
    reference : str,
    candidate : str
):
    reference = [reference]
    candidate = [candidate]
    
    metric_call_bary_score.prepare_idfs(candidate, reference)
    return metric_call_bary_score.evaluate_batch(candidate, reference)['baryscore_W'][0]


def DEPTH(
    reference : str,
    candidate : str
):
    
    reference = [reference]
    candidate = [candidate]
    
    metric_call_depth_score.prepare_idfs(reference, candidate)
    return metric_call_depth_score.evaluate_batch(reference, candidate)['depth_score'][0]


def MOVER(
    reference : str,
    candidate : str
):
    
    reference = [reference]
    candidate = [candidate]
    idf_dict_ref = defaultdict(lambda: 1.)
    idf_dict_hyp = defaultdict(lambda: 1.)
    
    return word_mover_score(reference, candidate, idf_dict_ref, idf_dict_hyp, n_gram=1)[0]

class MetricsComputer(torch.nn.Module):
    
    def __init__(self, metrics):
        super().__init__()
        
        self.metrics = metrics
    
    def forward(self, hyp, ref):
        
        return {str(metric.__name__) : metric(ref, hyp) for metric in self.metrics}
        
def compute_metrics(
    set_name : str,
    metrics : list,
    sample_size : float = None,
    path : str = None
):
    print("device : ", device)
    model = MetricsComputer(metrics = metrics).to(device)
    
    if sample_size is not None:
        df = dataset(set_name = set_name, sample_size = sample_size)
    else:
        df = dataset(set_name = set_name)
    
    dataloader = DataLoader(
        df,
        batch_size = 1,
        shuffle = False
    )
    
    sltls = { "sltl" : [] }
    hyps = { "hyp" : [] }
    refs = { "ref" : [] }
    domains = { "domain" : [] }
    seg_ids = { "seg_id" : [] }
    gold_scores = {"gold_score" : [] }
    metrics_scores = { str(metric.__name__) : []
                      for metric in metrics
    }
    
    count = 0
    
    df = pd.DataFrame([])
    
    for batch in tqdm(dataloader):
        
        sltl, hyp, ref, domain, seg_id, gold_score = batch
        
        sltls["sltl"].append(sltl[0])
        hyps["hyp"].append(hyp[0])
        refs["ref"].append(ref[0])
        domains["domain"].append(domain[0])
        seg_ids["seg_id"].append(seg_id.item())
        gold_scores["gold_score"].append(gold_score.item())
        
        scores = model(hyp[0], ref[0])
        
        for metric in metrics:
            metrics_scores[str(metric.__name__)].append(scores[str(metric.__name__)])
        
        count += 1
        
        if count >= 100:
            temp = pd.DataFrame(data = sltls | hyps | refs | domains | seg_ids | gold_scores | metrics_scores)
                
            for metric in metrics:
                temp[str(metric.__name__)] -= temp[str(metric.__name__)].mean()
                temp[str(metric.__name__)] /= temp[str(metric.__name__)].std()
            
            temp["gold_score"] -= temp["gold_score"].mean()
            temp["gold_score"] /= temp["gold_score"].std()
            
            df = pd.concat([df, temp])
            
            if path is not None:
                df.to_csv(path)
            
            sltls = { "sltl" : [] }
            hyps = { "hyp" : [] }
            refs = { "ref" : [] }
            domains = { "domain" : [] }
            seg_ids = { "seg_id" : [] }
            gold_scores = {"gold_score" : [] }
            metrics_scores = { str(metric.__name__) : []
                            for metric in metrics
            }
            
            count = 0
    
    
    return df
        
    
    
    
    
        
        