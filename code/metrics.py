## UTILS :
import numpy as np
import pandas as pd
from tqdm import tqdm
import nltk
from nltk.tokenize import TweetTokenizer
from data_processing import dataset
from torch.utils.data import DataLoader

## METRICS MODULES :
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

from torchmetrics.functional import translation_edit_rate

from torchmetrics.functional import sacre_bleu_score
from torchmetrics.functional.text.rouge import rouge_score
from torchmetrics.functional.text.bert import bert_score

from nlg_eval_via_simi_measures import bary_score
from nlg_eval_via_simi_measures import depth_score
from nlg_eval_via_simi_measures import infolm

metric_call_bary_score = bary_score.BaryScoreMetric(
    model_name = 'bert-base-uncased',
    use_idfs = False
    )

metric_call_depth_score = depth_score.DepthScoreMetric(
    model_name = 'bert-base-uncased',
    considered_measure='wasserstein'
)

metric_call_info_lm = infolm.InfoLM()



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


def WACC(
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
    
    return bert_score(candidate, reference, model_name_or_path = 'bert-base-uncased', verbose=True, idf = False)['f1']


def BARY(
    reference : str,
    candidate : str
):
    
    reference = [reference]
    candidate = [candidate]
    
    metric_call_bary_score.prepare_idfs(reference, candidate)
    return metric_call_bary_score.evaluate_batch(reference, candidate)['baryscore_W'][0]


def DEPTH(
    reference : str,
    candidate : str
):
    
    reference = [reference]
    candidate = [candidate]
    
    metric_call_depth_score.prepare_idfs(reference, candidate)
    return metric_call_depth_score.evaluate_batch(reference, candidate)['depth_score'][0]


def compute_metrics(
    set_name : str,
    metrics : list,
    batch_size : int,
    path : str = None):
    
    df = dataset(set_name = set_name)
    
    dataloader = DataLoader(
        df,
        batch_size = batch_size,
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
    
    
    for batch in tqdm(dataloader):
        
        sltl, hyp, ref, domain, seg_id, gold_score = batch
        
        for i in range(len(hyp)):
            
            sltls["sltl"].append(sltl[i])
            hyps["hyp"].append(hyp[i])
            refs["ref"].append(ref[i])
            domains["domain"].append(domain[i])
            seg_ids["seg_id"].append(seg_id[i].item())
            gold_scores["gold_score"].append(gold_score[i].item())
                    
            for metric in metrics:   
 
                metrics_scores[str(metric.__name__)].append(metric(ref[i], hyp[i]))
            
    if path is not None:
            
        df = pd.DataFrame(data = sltls | hyps | refs | domains | seg_ids | gold_scores | metrics_scores)
        
        for metric in metrics:
            df[str(metric.__name__)] -= df[str(metric.__name__)].mean()
            df[str(metric.__name__)] /= df[str(metric.__name__)].std()
        
        df["gold_score"] -= df["gold_score"].mean()
        df["gold_score"] /= df["gold_score"].std()
        df = df.set_index("gold_score")
        
        df.to_csv(path)
    
    
    return metrics_scores
            
    
    
    
    
        
        