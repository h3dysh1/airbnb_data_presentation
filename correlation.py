import numpy as np
import pandas as pd
import json
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

# Discretise numeric variables for MI/NMI calculations, and leaving the ordinal varaibles unchanged
def discretize(series, n_bins=4):
    x = series.to_numpy(dtype=float).reshape(-1, 1)
    if len(np.unique(x)) <= n_bins:
        return x.ravel()
    kbd = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
    return kbd.fit_transform(x).ravel()

def compute_correlations():
    results = []
    for v1, v2 in combinations(variables, 2):
    # Pearson correlation
        pearson_corr, _ = pearsonr(df[v1], df[v2])

        # Spearman correlation
        spearman_corr, _ = spearmanr(df[v1], df[v2])
            
            # Mutual Information
        # mi = mutual_info_score(df[v1], df[v2])
        x_disc, y_disc = discretize(df[v1]), discretize(df[v2])
        mi = mutual_info_score(x_disc, y_disc)

            
            # Normalized Mutual Information
        nmi = normalized_mutual_info_score(x_disc, y_disc)
            
        print(f"Correlation between {v1} and {v2}")
        print(f"  Pearson: {pearson_corr:.4f}")
        print(f"  Spearman: {spearman_corr:.4f}")
        print(f"  Mutual Information: {mi:.4f}")
        print(f"  Normalized Mutual Information: {nmi:.4f}\n")

        results.append({
            "var_1": v1,
            "var_2": v2,
            "pearson": round(pearson_corr, 4),
            "spearman": round(spearman_corr, 4),
            "mutual_information": round(mi, 4),
            "normalized_mutual_information": round(nmi, 4),
        })

    with open("correlation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Saved results to correlation_results.json")


compute_correlations()