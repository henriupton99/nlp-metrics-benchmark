## UTILS :
import numpy as np
import pandas as pd
from tqdm import tqdm
import nltk
from nltk.tokenize import TweetTokenizer
from data_processing import dataset
from torch.utils.data import DataLoader

## METRICS MODULES :
from torchmetrics.functional import sacre_bleu_score, chrf_score, translation_edit_rate
from torchmetrics.functional.text.rouge import rouge_score
from torchmetrics.functional.text.infolm import infolm
from torchmetrics.functional.text.bert import bert_score

from bary_score import BaryScoreMetric
from depth_score import DepthScoreMetric

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

def CHRF(
    reference : str,
    candidate : str
):
    
    reference = [reference]
    
    return chrf_score(candidate, reference, n_word_order = 0, lowercase = True).item()

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


def TER(
    reference : str,
    candidate : str
):
    
    reference = [reference]
    
    return translation_edit_rate(candidate, reference, asian_support = True, lowercase = True).item()


def INFO_LM(
    reference : str,
    candidate : str
):
    
    reference = [reference]
    
    return infolm(candidate, reference, idf = False).item() 


def BERT(
    reference : str,
    candidate : str
):
    reference = [reference]
    candidate = [candidate]
    
    return bert_score(candidate, reference, model_name_or_path = 'bert-base-uncased', verbose=True, idf = False)['f1']


def BARY(
    reference : str,
    candidate : str
):
    
    reference = [reference]
    candidate = [candidate]
    
    metric_call = BaryScoreMetric(model_name = 'bert-base-uncased', use_idfs = False)
    metric_call.prepare_idfs(reference, candidate)
    return metric_call.evaluate_batch(reference, candidate)['baryscore_W']


def DEPTH(
    reference : str,
    candidate : str
):
    
    reference = [reference]
    candidate = [candidate]
    
    metric_call = DepthScoreMetric(model_name = 'bert-base-uncased',considered_measure='wasserstein')
    metric_call.prepare_idfs(reference, candidate)
    return metric_call.evaluate_batch(reference, candidate)['depth_score']



