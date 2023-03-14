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
    
    return translation_edit_rate(candidate, reference, lowercase = True).item()

def TER_asian(
    reference : str,
    candidate : str
):
    
    reference = [reference]
    
    return translation_edit_rate(candidate, reference, lowercase = True, asian_support = True).item()


def INFO_LM(
    reference : str,
    candidate : str
):
    
    reference = [reference]
    
    return infolm(candidate, reference, idf = False).item()   