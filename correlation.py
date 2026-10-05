import numpy as np
import pandas as pd
import json
from scipy.stats import pearsonr, spearmanr
from itertools import combinations
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import normalized_mutual_info_score

DATA_PATH = "processed_data.csv"   # output of preprocessing_2.py, NOT processed_only.csv

AMENITY_INDICATORS = [
    "Wifi", "Kitchen", "Air conditioning", "Washer", "Heating", "Parking", "Self check-in"
]
REVIEW_FREQ_ORDER = ["No Reviews", "Low", "Medium", "High"]

df = pd.read_csv(DATA_PATH)
df['price'] = df['price'].replace(r'[\$,]', '', regex=True).astype(float)




# Creating an amenity count column from preproccessed data
has_cols = ["has_" + a.lower().replace(" ", "_") for a in AMENITY_INDICATORS]
df["amenity_count"] = df[has_cols].sum(axis=1)

# Reading in review frequency data and converting to an ordered categorical variable
df["review_frequency"] = pd.Categorical(
    df["review_frequency"], categories=REVIEW_FREQ_ORDER, ordered=True
).codes.astype(float)
df.loc[df["review_frequency"] < 0, "review_frequency"] = np.nan

variables = {"review_scores_rating": ("review_scores_rating", "continuous"),
    "price": ("price", "continuous"),
    "amenity_count": ("amenity_count", "discrete"),
    "host_listings_count": ("host_listings_count", "discrete"),
    "review_frequency": ("review_frequency", "ordinal")}

DISCRETE_KINDS = {"discrete", "ordinal"}

df = df.dropna(subset=list(variables.keys()))

def mutual_information(x, y, kind_x, kind_y):
    """MI WITHOUT binning, using nearest-neighbour estimators"""
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=float)
    if kind_y == "ordinal":
        return mutual_info_classif(x, y.astype(int), discrete_features=False, random_state=42)[0]
    if kind_x == "ordinal":
        return mutual_info_classif(y.reshape(-1, 1), x.ravel().astype(int),
                                   discrete_features=False, random_state=42)[0]
    return mutual_info_regression(x, y, discrete_features=False, random_state=42)[0]
 
def compute_correlations():
    results = []
    for v1, v2 in combinations(variables, 2):
        kind1 = variables[v1][1]
        kind2 = variables[v2][1]

        # Pearson correlation
        pearson_corr, _ = pearsonr(df[v1], df[v2])

        # Spearman correlation
        spearman_corr, _ = spearmanr(df[v1], df[v2])
            
        # Mutual Information
        mi = mutual_information(df[v1], df[v2], kind1, kind2)
     
        # NMI: needs discrete labels, so only computed when BOTH variables are
        # already discrete
        if kind1 in DISCRETE_KINDS and kind2 in DISCRETE_KINDS:
            nmi = round(float(normalized_mutual_info_score(df[v1], df[v2])), 4)
        else:
            nmi = None
            
       
        print(f"Correlation between {v1} and {v2}")
        print(f"  Pearson: {pearson_corr:.4f}")
        print(f"  Spearman: {spearman_corr:.4f}")
        print(f"  Mutual Information: {mi:.4f}")
        print(f"  Normalized Mutual Information: {'n/a (continuous variable)' if nmi is None else f'{nmi:.4f}'}\n")
 
        results.append({
            "var_1": v1,
            "var_2": v2,
            "pearson": round(pearson_corr, 4),
            "spearman": round(spearman_corr, 4),
            "mutual_information": round(float(mi), 4),
            "normalized_mutual_information": nmi,
        })

    with open("correlation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Rows used for every pair: {len(df)}")
    print("Saved results to correlation_results.json")


compute_correlations()