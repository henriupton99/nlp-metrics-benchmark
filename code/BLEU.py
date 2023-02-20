import numpy as np

class Sentence_BLEU:
    
    def __init__(self, tokenizer):
        
        self.tok = tokenizer
    
    @staticmethod
    def compute_n_grams(sentence, n):
        
        sentence = sentence.lower()
        
        sentence = [x for x in sentence]
        
        L = len(sentence)
        
        assert n <= L
        
        grams = []
        
        for l in range(L-n+1):
            
            grams.append(" ".join(sentence[l : l + n]))
            
        return grams
            
        
        
        
        
        
        
    
    
    
    
        
        