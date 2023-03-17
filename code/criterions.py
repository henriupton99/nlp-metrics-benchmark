import pandas as pd
import numpy as np

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
            criterion_scores.append(-1*score)
        
        ranks[l,:] = np.argsort(criterion_scores)
    
    criterion_scores = []
    for n in range(N):
        candidate_metric = str(metrics[n].__name__)
        score = bad_quality_detection(
                    data = data,
                    candidate_metric = candidate_metric,
                    correlation_type = correlation_type,
                    alpha = 0.1
                )
        criterion_scores.append(-1*score)
    
    ranks[L-2,:] = np.argsort(criterion_scores)
    
    criterion_scores = []
    for n in range(N):
        candidate_metric = str(metrics[n].__name__)
        score = segment_level_coverage(
                    data = data,
                    candidate_metric = candidate_metric,
                    correlation_type = correlation_type,
                )
        criterion_scores.append(-1*score)
    
    ranks[L-1,:] = np.argsort(criterion_scores)

    metrics_sum_ranks = np.sum(ranks, axis = 0) 
    
    return np.argsort(metrics_sum_ranks)