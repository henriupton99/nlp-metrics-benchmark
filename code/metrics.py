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

try :
    from nltk.translate.meteor_score import meteor_score
except :
    nltk.download('wordnet')
    nltk.download('omw-1.4')

from nltk.translate import nist_score

from torchmetrics.functional import word_error_rate

def BLEU(
    reference : str,
    candidate : str
):
    
    reference = [reference]
    
    return bleu_score(candidate, reference, n_gram = 1).item()
    

def METEOR(
    reference : str,
    candidate : str
):
    
    hyp = TweetTokenizer().tokenize(candidate)
    ref = TweetTokenizer().tokenize(reference)
    
    return meteor_score([ref], hyp)

def NIST(
    reference : str,
    candidate : str
):
    
    hyp = TweetTokenizer().tokenize(candidate)
    ref = TweetTokenizer().tokenize(reference)
    
    return nist_score.sentence_nist([ref], hyp, n = 1)

def WACC(
    reference : str,
    candidate : str
):
    
    reference = [reference]
    
    wer = word_error_rate(candidate, reference).item()
    
    return 1 - wer

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
    
    hyps = { "hyp" : [] }
    refs = { "ref" : [] }
    gold_scores = {"gold_score" : [] }
    metrics_scores = { str(metric.__name__) : []
                      for metric in metrics
    }
    
    
    for batch in tqdm(dataloader):
        
        hyp, ref, gold_score = batch
        
        for i in range(len(hyp)):
            
            hyps["hyp"].append(hyp[i])
            refs["ref"].append(ref[i])
            gold_scores["gold_score"].append(gold_score[i].item())
                    
            for metric in metrics:   
 
                metrics_scores[str(metric.__name__)].append(metric(ref[i], hyp[i]))
            
    if path is not None:
            
        df = pd.DataFrame(data = hyps | refs | gold_scores | metrics_scores)
        
        df = df.set_index("gold_score")
        
        df.to_csv(path)
    
    
    return metrics_scores
            
            
        
        
    
    
    
    
    
    
    
## KEEP ?
      
class Sentence_BLEU:
    
    def __init__(self, N):
        
        self.N = N
        self.w = np.ones(N)/N
    
    def compute_n_grams(self, sentence, n):
        
        sentence = sentence.lower()
        
        sentence = [x for x in sentence]
        
        L = len(sentence)
        
        assert n <= L
        
        grams = []
        
        for l in range(L-n+1):
            
            grams.append(" ".join(sentence[l : l + n]))
            
        Grams, Counts = np.unique(grams, return_counts = True)
        
        return Grams, Counts
    
    def modified_precision(self, reference, candidate, n):
        
        G_ref, C_ref = self.compute_n_grams(reference, n)
        
        G_hyp, C_hyp = self.compute_n_grams(candidate, n)
        
        dict_ref = {G_ref[k] : C_ref[k] for k in range(len(G_ref)) }
        dict_hyp = {G_hyp[k] : C_hyp[k] for k in range(len(G_hyp)) }
        
        p_n = 0
        
        for gram, count in dict_hyp.items():
            
            if gram not in G_ref :
                
                pass
            
            else:
                
                p_n += min(dict_ref[gram], count) / count
            
        return p_n
    
    def compute_score(self, reference, candidate):
        
        P_n = np.empty(self.N)
        
        for n in range(1, self.N+1):
            
            p_n = self.modified_precision(reference, candidate, n)
            
            P_n[n-1] = p_n
        
        
        return np.exp(np.sum(self.w * np.log(P_n + 1e-10)))
    
        
        

            
        
        
        
        
        
        
    
    
    
    
        
        