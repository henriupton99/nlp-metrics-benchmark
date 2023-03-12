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

from nltk.translate.chrf_score import sentence_chrf

from torchmetrics.functional import word_error_rate

from torchmetrics.functional import translation_edit_rate

def BLEU(
    reference : str,
    candidate : str
):
    
    reference = [[reference]]
    candidate = [candidate]
    
    return bleu_score(preds = candidate, target = reference, n_gram = 1).item()
    

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
    
    reference = [[reference]]
    candidate = [candidate]
    
    ter = translation_edit_rate(preds = candidate, target = reference).item()
    
    return 1 - ter

def CHRF(
    reference : str,
    candidate : str
):
    
    hyp = TweetTokenizer().tokenize(candidate)
    ref = TweetTokenizer().tokenize(reference)
    
    return sentence_chrf(ref, hyp)


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
            
    
    
    
    
        
        