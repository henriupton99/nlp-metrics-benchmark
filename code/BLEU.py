import numpy as np

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
        
        
        return np.exp(np.sum(self.w * np.log(P_n)))
    
        
        

            
        
        
        
        
        
        
    
    
    
    
        
        