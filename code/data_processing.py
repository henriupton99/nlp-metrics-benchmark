# IMPORTS :
import pandas as pd

## DICTIONNAIRE DES COUPLES DE TRADUCTIONS :
sl_tls = {
    "English-German" : "ende",
    "English-Russian" : "enru",
    "Chinese-English" : "zhen"
}

class WMT22:
    
    @staticmethod
    def MQM_unlabel_score(
        df_rates : pd.DataFrame,
        seg_id : int,
        sys : str
    ):
        penality = {
            "No-error" : 0,
            "neutral" : 0,
            "minor" : 1,
            "major" : 5,
            "critical" : 10
        }
        
        try:
            query = df_rates.loc[seg_id]
            
            query = query[query["system"] == sys]
            
            if len(query) != 0:
            
                query["score"] = query["severity"].map(penality)
            
                return 100 * (1 - query["score"].sum()) # ajouter la somme des tokens
            
            else : 
                return 100
            
        except :
            return "rien"
    
    @staticmethod
    def MQM_google_score(
        df_rates : pd.DataFrame,
        seg_id : int,
        sys : str
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
            query = df_rates.loc[seg_id]
            
            query = query[query["system"] == sys]
            
            if len(query) != 0:
            
                query["score"] = query.apply(lambda x : penality(x.category, x.severity), axis = 1)
            
                return query["score"].max()
            
            else : 
                return 0
            
        except :
            return "rien"
    
    @classmethod
    def read_data(
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
        
        assert sl_tl in sl_tls.values()
        
        df = pd.read_table("./data/generalMT2022/" + sl_tl + "/mqm_generalMT2022_" + sl_tl + ".avg_seg_scores.tsv",
                        on_bad_lines = "skip")

        
        ## DROP NA VALUES :
        df = df.dropna()
        
        ## CHANGE TYPES OF VARIABLES :
        df['seg_id'] = df['seg_id'].astype(int)
        #df['score'] = df['score'].astype(float)
        
        ## SET INDEX :
        df = df.set_index("seg_id")
        
        ## GET SCORES :
        
        if sl_tl in ["ende", "zhen"]:
            
            df_rates = pd.read_table("./data/generalMT2022/" + sl_tl + "/mqm_generalMT2022_" + sl_tl + ".tsv",
                                 on_bad_lines = "skip").set_index("seg_id")
            
        
            df["score_test"] = df.apply(lambda x : WMT22.MQM_google_score(
                df_rates = df_rates,
                seg_id = x.name,
                sys = x.sys
            ), axis = 1)
        
        else :
            df_rates = pd.read_table("./data/generalMT2022/" + sl_tl + "/mqm_generalMT2022_" + sl_tl + ".tsv",
                                 on_bad_lines = "skip",
                                 index_col = False).set_index("seg_id")
            
            df["score_test"] = df.apply(lambda x : WMT22.MQM_unlabel_score(
                df_rates = df_rates,
                seg_id = x.name,
                sys = x.sys
            ), axis = 1)
        
        
        return df
    
    
    
        
        
    