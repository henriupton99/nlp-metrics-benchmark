# IMPORTS :
import pandas as pd
from nltk import TweetTokenizer

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
    def data_golden_scores(
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
        
        df.to_csv("./data/WMT22_w_golden_scores_" + sl_tl + ".csv")
        
        return df
    
    @classmethod
    def read_data(
        cls,
        sl_tl : str
    ):
        
        return pd.read_csv("./data/WMT22_w_golden_scores_" + sl_tl + ".csv")
    
    
    
        
        
    