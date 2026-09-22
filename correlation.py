import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score

DATA_PATH = "processed_data.csv"   # output of preprocessing_2.py, NOT processed_only.csv
REFERENCE_DATE = pd.Timestamp("2026-07-07")  # set this to your dataset's scrape date
 

df = pd.read_csv(DATA_PATH)
df['price'] = df['price'].replace('[\$,]', '', regex=True).astype(float)

variables = ['review_scores_rating', 'price', 'amenities_count', 'host_listings_count', 'days_since_last_review']

df = df.dropna(subset=variables)
def compute_correlations():
    for v1, v2 in combinations(variables, 2):
    # Pearson correlation
        pearson_corr, _ = pearsonr(df[v1], df[v2])

        # Spearman correlation
        spearman_corr, _ = spearmanr(df[v1], df[v2])
            
            # Mutual Information
        mi = mutual_info_score(df[v1], df[v2])
            
            # Normalized Mutual Information
        nmi = normalized_mutual_info_score(df[v1], df[v2])
            
        print(f"Correlation between {v1} and {v2}")
        print(f"  Pearson: {pearson_corr:.4f}")
        print(f"  Spearman: {spearman_corr:.4f}")
        print(f"  Mutual Information: {mi:.4f}")
        print(f"  Normalized Mutual Information: {nmi:.4f}\n")


compute_correlations()