# IMPORTS :
import pandas as pd
import tarfile

## DICTIONNAIRE DES COUPLES DE TRADUCTIONS :
sl_tls = {
    "English-German" : "en-de",
    "English-Chinese" : "en-zh",
    "Romanian-English" : "ro-en",
    "Estonian-English" : "et-en",
    "Nepalese-English" : "ne-en",
    "Sinhala-English" : "si-en"
}

def read_data(
    sl_tl : str, 
    set_type : str
    ):
    """collect the data for a given couple (sl, tl)

    Args:
        sl_tl (str): traduction couple (source, target)
        set_type (str): type of the set : train, test or dev

    Returns:
        df: DataFrame
    """
    
    assert sl_tl in sl_tls.values()
    assert set_type in ["train", "dev", "test"]
    
    ## ACCES AUX DONNEES DE TRAIN ET DEV :
    if set_type in ["train", "dev"]:
        
        with tarfile.open("data/"+sl_tl+".tar.gz", "r:*") as tar:
            separator = "\t"
            
            df = pd.read_csv(tar.extractfile(""+sl_tl+'/dev.'+sl_tl.replace("-","")+'.df.short.tsv'),
                                on_bad_lines='skip',
                                sep = separator,
                                index_col = "index")
        
    ## ACCES AUX DONNEES DE TEST :
    else:
        
        with tarfile.open("data/"+sl_tl+"_test.tar.gz", "r:*") as tar:
        
            separator = "\t"
            
            df = pd.read_csv(tar.extractfile(""+ sl_tl+"/test20."+sl_tl.replace("-", "")+".df.short.tsv"),
                                on_bad_lines='skip',
                                sep = separator,
                                index_col = "index")
    
    ## CHANGE TYPES OF VARIABLES :
    # coerce identifie les valeurs invalides par des NaN :
    df['mean'] = pd.to_numeric(df['mean'], errors='coerce')

    # retirer les observations invalides
    df = df.dropna()
    
    return df

def merge_bases(
    sl_tl : str
):
    """Concatenate train, dev, test datasets for a given traduction couple (sl, tl)

    Args:
        sl_tl (str): traduction couple (source, target)

    Returns:
        df: DataFrame
    """
    
    assert sl_tl in sl_tls.values()
    
    df = pd.DataFrame({})
    
    for set_type in ["train", "dev", "test"]:
        query = read_data(sl_tl = sl_tl, set_type = set_type)
        query["set_type"] = set_type
        df = pd.concat([df, query])
        assert list(query.columns) == list(df.columns)
        
    df = df.reset_index().drop(["index"], axis = 1)
        
    return df
    
    
    
        
        
    