import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.preprocessing import KBinsDiscretizer
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score

DATA_PATH = "processed_data.csv"   # output of preprocessing_2.py, NOT processed_only.csv
REFERENCE_DATE = pd.Timestamp("2026-07-07")  # set this to your dataset's scrape date
 
AMENITY_INDICATORS = [
    "Wifi", "Kitchen", "Air conditioning", "Washer", "Heating", "Parking", "Self check-in"
]
REVIEW_FREQ_ORDER = ["No Reviews", "Low", "Medium", "High"]

df = pd.read_csv(DATA_PATH)
df['price'] = df['price'].replace('[\$,]', '', regex=True).astype(float)




# Creating an amernity count column from preproccessed data
has_cols = ["has_" + a.lower().replace(" ", "_") for a in AMENITY_INDICATORS]
df["amenity_count"] = df[has_cols].sum(axis=1)

# Reading in review frequency data and converting to an ordered categorical variable
df["review_frequency"] = pd.Categorical(
    df["review_frequency"], categories=REVIEW_FREQ_ORDER, ordered=True
).codes.astype(float)
df.loc[df["review_frequency"] < 0, "review_frequency"] = np.nan

variables = {"review_scores_rating": ("review_scores_rating", "numeric"),
    "price": ("price", "numeric"),
    "amenity_count": ("amenity_count", "numeric"),
    "host_listings_count": ("host_listings_count", "numeric"),
    "review_frequency": ("review_frequency", "ordinal")}

df = df.dropna(subset=list(variables.keys()))


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