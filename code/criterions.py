

def compute_correlation(
    data : pd.DataFrame,
    metric_1 : str,
    metric_2 : str,
    correlation_type : str
):
    
    return data[[metric_1, metric_2]].corr()

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