import pandas as pd
import numpy as np
from scipy.stats import rankdata
import matplotlib.pyplot as plt
import seaborn as sns

sl_tls = {
    "ende" : "English-German",
    "enru" : "English-Russian",
    "zhen" : "Chinese-English" 
}

def compute_correlation(
    data : pd.DataFrame,
    metric_1 : str,
    metric_2 : str,
    correlation_type : str
):
    
    return data[[metric_1, metric_2]]\
        .corr(method = correlation_type)\
            .to_numpy()[0,1]

def domain_coverage(
    data : pd.DataFrame,
    candidate_metric :str,
    correlation_type : str,
    domain : str
):
    query = data[data.domain == domain]
    
    corr = compute_correlation(
        data = query,
        metric_1 = "gold_score",
        metric_2 = candidate_metric,
        correlation_type = correlation_type
    )
    
    return corr

def bad_quality_detection(
    data : pd.DataFrame,
    candidate_metric :str,
    correlation_type : str,
    alpha : float
):
    quantile_alpha = data.gold_score.quantile(alpha)
    
    query = data[data.gold_score <= quantile_alpha]
    
    corr = compute_correlation(
        data = query,
        metric_1 = "gold_score",
        metric_2 = candidate_metric,
        correlation_type = correlation_type
    )
    
    return corr

def segment_level_coverage(
    data : pd.DataFrame,
    candidate_metric : str,
    correlation_type : str
):
    
    query = data.groupby("seg_id")\
        [["gold_score", candidate_metric]]\
            .mean()
            
    corr = compute_correlation(
        data = query,
        metric_1 = "gold_score",
        metric_2 = candidate_metric,
        correlation_type = correlation_type
    )
    
    return corr
    

def Borda_count(
    data : pd.DataFrame,
    metrics : list,
    correlation_type : str
):
    
    L = len(data.domain.unique()) + 1 + 1
    N = len(metrics)
    ranks = np.empty((L, N))
    correlations = np.empty((L,N))

    for l in range(L-2):
        criterion_scores = []
        criterion_domain = data.domain.unique()[l]
        for n in range(N):
            candidate_metric = str(metrics[n].__name__)
            score = domain_coverage(
                        data = data,
                        candidate_metric = candidate_metric,
                        correlation_type = correlation_type,
                        domain = criterion_domain
                    )
            correlations[l,n] = score
    
    criterion_scores = []
    for n in range(N):
        candidate_metric = str(metrics[n].__name__)
        score = bad_quality_detection(
                    data = data,
                    candidate_metric = candidate_metric,
                    correlation_type = correlation_type,
                    alpha = 0.1
                )
        correlations[L-2,n] = score
    
    criterion_scores = []
    for n in range(N):
        candidate_metric = str(metrics[n].__name__)
        score = segment_level_coverage(
                    data = data,
                    candidate_metric = candidate_metric,
                    correlation_type = correlation_type,
                )
        correlations[L-1,n] = score
    
    criterion_ranks = np.apply_along_axis(rankdata, axis=1, arr=-correlations)
    sum_ranks = np.sum(criterion_ranks, axis=0)
    order = list(np.argsort(sum_ranks))
    
    borda_ranks = [order.index(n) + 1 for n in range(N)]

    
    return borda_ranks , correlations, criterion_ranks


def load_metrics_scores(
    metrics : dict
):
    
    data = pd.DataFrame([])

    for sl_tl in sl_tls.keys():
        
        query = pd.read_csv("./data/embedding_based_metrics_scores.csv",
                    index_col = 0)
        query = query[query.sltl == sl_tl]
        len_data = len(query)
        
        df = pd.DataFrame([])
        count = 0
        for metric_type in ["embedding_based","edit_based", "ngram_based"]:
            
            temp = pd.read_csv("./data/" + metric_type + "_metrics_scores.csv")
            temp = temp[temp.sltl == sl_tl][:len_data].reset_index()
            
            if count == 0:
                df[["sltl", "hyp", "ref", "domain", "seg_id", "gold_score"]] = \
                    temp[["sltl", "hyp", "ref", "domain", "seg_id", "gold_score"]]
            
            
            for metric in metrics[metric_type]:
                
                df[str(metric.__name__)] = temp[str(metric.__name__)]
            
            count +=1
        
        data = pd.concat([data, df])
    
    return data


def plot_rank_per_citerion(
    sl_tl : str,
    metrics : dict,
    path : str = None
):
    
    data = load_metrics_scores(metrics = metrics)
    
    metrics_all = [metric for sublist in metrics.values() for metric in sublist]
        
    query = data[data.sltl == sl_tl]
    
    borda_count_ranks, inv_correlations, criterion_ranks = Borda_count(
                            data = query,
                            metrics = metrics_all,
                            correlation_type = "spearman"
                        )

    plt.figure(figsize = (10, 5))
    
    cmap = sns.color_palette("RdYlGn_r", 10)
    x_labels = [str(metric.__name__) for metric in metrics_all]
    y_labels = ["DC " + domain for domain in data.domain.unique()] + ["BQD"] + ["SLC"]

    
    ax = sns.heatmap(criterion_ranks + 1,
                annot = True,
                cmap = cmap,
                xticklabels=x_labels,
                yticklabels=y_labels,
                annot_kws={"fontsize": 12}
                )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    
    if path is not None:
        plt.savefig(path, bbox_inches='tight')
    
    plt.title(sl_tls[sl_tl])
    plt.show()
    

def plot_borda_ranks(
    sl_tl : str,
    metrics : dict,
    path : str = None
):
    data = load_metrics_scores(metrics = metrics)
    
    metrics_all = [metric for sublist in metrics.values() for metric in sublist]

    query = data[data.sltl == sl_tl]
    
    borda_count_ranks, inv_correlations, criterion_ranks = Borda_count(
                            data = query,
                            metrics = metrics_all,
                            correlation_type = "spearman"
                        )

    plt.figure(figsize = (10,3))

    category_colors = {
        "ngram_based": "blue",
        "edit_based": "red",
        "embedding_based" : "green"
    }

    metrics_ranks = {}
    for n in range(len(metrics_all)):
        
        if metrics_all[n] in metrics["ngram_based"]:
            category = "ngram_based" 
        elif metrics_all[n] in metrics["edit_based"]:
            category = "edit_based" 
        else:
            category = "embedding_based" 
            
        metrics_ranks[str(metrics_all[n].__name__)] = (borda_count_ranks[n], category)

    sorted_scores = sorted(metrics_ranks.items(), key=lambda x: x[1][0], reverse=True)

    x = [sorted_scores[k][0] for k in range(len(metrics_all))]
    y = [len(metrics_all) - sorted_scores[k][1][0] for k in range(len(metrics_all))]
    colors = [category_colors[sorted_scores[k][1][1]] for k in range(len(metrics_all))]

    plt.bar(x, y, color=colors)

    for i, v in enumerate(sorted_scores):
        plt.text(i, y[i] + 0.1, str(v[1][0]), ha='center', fontweight='bold')

    plt.ylabel('Borda Rank')
    plt.xticks(rotation=45)
    plt.title(sl_tls[sl_tl])
    plt.savefig(path, bbox_inches='tight')
    plt.show()