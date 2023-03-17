# IMPORTS :
import pandas as pd
from nltk import TweetTokenizer

import torch
from torch.utils.data import Dataset

## DICTIONNAIRE DES COUPLES DE TRADUCTIONS :
sl_tls = {
    "ende" : "English-German",
    "enru" : "English-Russian",
    "zhen" : "Chinese-English" 
}

tokenizer = TweetTokenizer()

class WMT22:
    
    @staticmethod
    def MQM_unlabel_score(
        df_rates : pd.DataFrame,
        source : str,
        sys : str,
        doc : str
    ):
        penality = {
            "No-error" : 0,
            "neutral" : 0,
            "minor" : -1,
            "major" : -5,
            "critical" : -10
        }
        
        try:
            query = df_rates[df_rates.source.values == source]
            
            query = query[query["system"].values == sys]
            
            query = query[query["doc"].values == doc]
            
            if len(query) != 0:
            
                query["score"] = query["severity"].map(penality)
            
                return query["score"].sum()
            
            else : 
                return 0
            
        except :
            return "None"
    
    @staticmethod
    def MQM_google_score(
        df_rates : pd.DataFrame,
        source : str,
        sys : str,
        doc : str
    ):
        
        def penality(category, severity):
            
            if severity == "major":
                if category == "No-translation":
                    return -25
                else:
                    return -5
            
            elif severity == "minor":
                if category == "Fluency/Punctuation":
                    return -0.1
                else :
                    return -1
            else:
                return 0
            
        try:
            
            query = df_rates[df_rates.source.values == source]
            
            query = query[query["system"].values == sys]
            
            query = query[query["doc"].values == doc]
            
            if len(query) != 0:
            
                query["score"] = query.apply(lambda x : penality(x.category, x.severity), axis = 1)
            
                return query["score"].sum()
            
            else : 
                return 0
            
        except :
            return "None"
    
    @classmethod
    def data_gold_scores(
        cls,
        sl_tl : str
        ):
        """collect the data for a given couple (sl, tl)

        Args:
            sl_tl (str): traduction couple (source, target)
            set_type (str): type of the set : train, test or dev

        Returns:
            df: DataFrame
        """
        
        assert sl_tl in sl_tls.keys()
        
        df = pd.read_table("./data/generalMT2022/" + sl_tl + "/mqm_generalMT2022_" + sl_tl + ".avg_seg_scores.tsv",
                        on_bad_lines = "skip")

        
        ## DROP NA VALUES :
        df = df.dropna()
        
        ## CHANGE TYPES OF VARIABLES :
        df['seg_id'] = df['seg_id'].astype(int)
        
        ## SET INDEX :
        df = df.set_index("seg_id")
        
        ## GET SCORES :
        
        if sl_tl in ["ende", "zhen"]:
            
            df["doc"] = df.apply(lambda x : x.domain + "_" + x.doc + ":" + sl_tls[sl_tl], axis = 1)
            
            df_rates = pd.read_table("./data/generalMT2022/" + sl_tl + "/mqm_generalMT2022_" + sl_tl + ".tsv",
                                 on_bad_lines = "skip")
            
            if sl_tl == "zhen":
                df_rates["system"] = df_rates["system"].apply(lambda x : x.replace(".en", ""))
            
        
            df["score"] = df.apply(lambda x : WMT22.MQM_google_score(
                df_rates = df_rates,
                source = x.source,
                sys = x.sys,
                doc = x.doc
            ), axis = 1)
        
        else :
            df_rates = pd.read_table("./data/generalMT2022/" + sl_tl + "/mqm_generalMT2022_" + sl_tl + ".tsv",
                                 on_bad_lines = "skip",
                                 index_col = False)
            
            df["score"] = df.apply(lambda x : WMT22.MQM_unlabel_score(
                df_rates = df_rates,
                source = x.source,
                sys = x.sys,
                doc = x.doc
            ), axis = 1)
        
        
        df.to_csv("./data/WMT22_w_gold_scores_" + sl_tl + ".csv")
        
        return df
    
    @classmethod
    def read_data(
        cls,
        sample_size : float = None
    ):
        
        for index, sl_tl in enumerate(sl_tls.keys()):
            
            query = pd.read_csv("./data/WMT22_w_gold_scores_" + sl_tl + ".csv")
            
            if sample_size is not None:
                query = query[:sample_size]
            
            query["sltl"] = sl_tl
            
            if index == 0:
                
                df = query.copy()
            
            else :
                
                df = pd.concat([df, query])
            
        
        return df
    
    
class dataset(Dataset):
    
    def __init__(self,
                 set_name : str,
                 sample_size : float = None):
        
        if set_name == "WMT22":
            if sample_size is not None:
                self.df = WMT22.read_data(sample_size = sample_size)
            else:
                self.df = WMT22.read_data()
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        
        sltl = self.df.iloc[idx].sltl
        hyp = self.df.iloc[idx].hyp
        ref = self.df.iloc[idx].ref
        domain = self.df.iloc[idx].domain
        seg_id = self.df.iloc[idx].seg_id
        gold_score = self.df.iloc[idx].score
        
        return sltl, hyp, ref, domain, seg_id, gold_score
    
    
    
    
        
        
    