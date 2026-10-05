# PCA + CLUSTERING

# Libraries --------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

# Defined variables -------------------------------------------------------
FEATURE_SETS = {
    "Geographic": [
        "latitude",
        "longitude",
        "amenity_count",
        "accommodates"
    ],
    
    "Price-tier": [
        "price",
        "accommodates",
        "bedrooms",
        "review_scores_rating"
    ],
    
    "Host operating profile": [
        "host_listings_count",
        "availability_365",
        "review_scores_rating"
    ],
    
    "Custom listing characteristics": [
        "price",
        "accommodates",
        "bedrooms",
        "amenity_count",
        "availability_365"
    ]
}

AMENITY_COLUMNS = [
    "has_wifi",
    "has_kitchen",
    "has_air_conditioning",
    "has_washer",
    "has_heating",
    "has_parking"
]

CLUSTER_FEATURES = [
    "price_log",
    "accommodates",
    "bedrooms",
    "amenity_count",
    "availability_365"
]

K = 6

DATA_PATH = "processed_data.csv"

# Functions ----------------------------------------------------------------

# Preparing data - might need to add to preprocessing ?
def prepare_data(raw_data):
    df = raw_data.copy()
 
    # Price processing
    if not pd.api.types.is_numeric_dtype(df["price"]):
        df["price"] = pd.to_numeric(
            df["price"].astype(str).str.replace(r"[$,]", "", regex=True),
            errors="coerce"
        )
 
    # Handles t/f text flags as well as booleans, missing counts as absent
    df[AMENITY_COLUMNS] = (
        df[AMENITY_COLUMNS]
        .replace({"t": 1, "f": 0})
        .fillna(0)
        .astype(int)
    )
    df["amenity_count"] = df[AMENITY_COLUMNS].sum(axis=1)
 
    # log1p keeps zero prices valid
    df["price_log"] = np.log1p(df["price"])
 
    return df

# Comparing summary statistics and missing values of each candidate set
def compare_candidate_feature_sets(df):
    df = df.copy()

    for name, features in FEATURE_SETS.items():
        
        print("=" * 60)
        print(name)
        print("=" * 60)
        
        print("Features:")
        print(features)
        
        print("\nMissing values:")
        print(df[features].isna().sum())
        
        print("\nSummary statistics:")
        print(df[features].describe())
        
        print()

# Creating the feature set, returns cluster_df and X_scaled for k_means
def feature_set(df, features):
    df = df.copy()

    # Creating separate data frame and removing missing values 
    cluster_df = df[features + ["review_scores_rating"]].copy() 
    cluster_df = cluster_df.dropna(subset=features).copy()

    # Distributions
    print(cluster_df[features].describe())

    # Histograms
    cluster_df[features].hist(
        figsize=(12, 8),
        bins=30
    )
    plt.tight_layout()

    # Standardise
    X = cluster_df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_scaled_df = pd.DataFrame(
        X_scaled,
        columns=features,
        index=cluster_df.index
    )

    print(X_scaled_df.head())

    return cluster_df, X_scaled

# Fit k-means for k = 2 to 10 on the scaled data and plot the inertia
# of each model so the elbow (where improvement levels off) can be found
def k_means(X):
    inertias = []

    k_values = range(2, 11)

    for k in k_values:
        
        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )
        
        kmeans.fit(X)
        
        inertias.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(
        k_values,
        inertias,
        marker="o"
    )

    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method for K-Means")
    plt.xticks(list(k_values))

# Fit the final k-means model with the chosen k and return a copy of the
# data frame with each row's cluster label in a new kmeans_cluster column
# K is chosen from elbow method and defined above
def final_kmeans(df, X, k):
    df = df.copy()

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    df["kmeans_cluster"] = kmeans.fit_predict(X)

    return df

# Compares review ratings by cluster
# Visualises differences using a bar graph
# Might wanna fix this (heavily right skewed data)
# Use our low/med/high categories instead somehow?
def compare_ratings_by_cluster(df):
    rating_by_cluster = (
        df
        .groupby("kmeans_cluster")["review_scores_rating"]
        .agg(["mean", "median", "count"])
    )

    print(rating_by_cluster.round(3))

    plt.figure(figsize=(8, 5))

    plt.bar(
        rating_by_cluster.index.astype(str),
        rating_by_cluster["mean"]
    )

    plt.xlabel("K-Means Cluster")
    plt.ylabel("Mean Review Score")
    plt.title("Mean Review Score by K-Means Cluster")

# Fit hierarchical (Ward) clustering with k clusters and attach the labels
def final_hierarchical(df, X, k):
    df = df.copy()

    hierarchical = AgglomerativeClustering(
        n_clusters=k
    )

    df["hierarchical_cluster"] = hierarchical.fit_predict(X)

    return df

# Summarise each cluster: size, share of listings and average feature values
def describe_clusters(df, features, cluster_col):
    cols = list(dict.fromkeys(features + ["review_scores_rating"]))
    summary = df.groupby(cluster_col)[cols].mean().round(2)

    counts = df[cluster_col].value_counts().sort_index()
    summary["size"] = counts
    summary["percent"] = (counts / counts.sum() * 100).round(2)

    summary["avg_price"] = np.expm1(summary["price_log"]).round(0)

    print(summary)
    return summary

# Produces adjusted rand index to compare k_means and hierarchial clustering
def compare_kmeans_hierarchial(df):
    ari = adjusted_rand_score(
        df["kmeans_cluster"],
        df["hierarchical_cluster"]
    )

    print("Adjusted Rand Index:", ari)

# Fit PCA on the scaled data, keeping every component
def run_pca(X):
    pca = PCA()
    X_pca = pca.fit_transform(X)

    return pca, X_pca

# Scree plot: variance per component with the cumulative total
def plot_explained_variance(pca):
    ratios = pca.explained_variance_ratio_
    components = np.arange(1, len(ratios) + 1)

    plt.figure(figsize=(8, 5))
    plt.bar(components, ratios, label="Individual")
    plt.plot(
        components,
        np.cumsum(ratios),
        marker="o",
        color="black",
        label="Cumulative"
    )

    plt.xlabel("Principal component")
    plt.ylabel("Proportion of variance explained")
    plt.title("PCA Explained Variance")
    plt.xticks(components)
    plt.legend()

# Show which original features drive each component
def pca_loadings(pca, features, n_components=3):
    loadings = pd.DataFrame(
        pca.components_[:n_components].T,
        columns=[f"PC{i + 1}" for i in range(n_components)],
        index=features
    ).round(2)

    print(loadings)
    return loadings

# Plot the first two components coloured by cluster label
def plot_pca_clusters(X_pca, labels, title):
    plt.figure(figsize=(8, 6))

    scatter = plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=labels,
        cmap="tab10",
        s=8,
        alpha=0.6
    )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.legend(*scatter.legend_elements(), title="Cluster")

# Everything together
if __name__ == "__main__":
    df = prepare_data(pd.read_csv(DATA_PATH))

    compare_candidate_feature_sets(df)
    cluster_df, X_scaled = feature_set(df, CLUSTER_FEATURES)
    k_means(X_scaled)

    clustered_df = final_kmeans(cluster_df, X_scaled, K)
    clustered_df = final_hierarchical(clustered_df, X_scaled, K)

    describe_clusters(clustered_df, CLUSTER_FEATURES, "kmeans_cluster")
    describe_clusters(clustered_df, CLUSTER_FEATURES, "hierarchical_cluster")
    compare_kmeans_hierarchial(clustered_df)
    compare_ratings_by_cluster(clustered_df)

    pca, X_pca = run_pca(X_scaled)
    plot_explained_variance(pca)
    pca_loadings(pca, CLUSTER_FEATURES)

    plot_pca_clusters(
        X_pca, clustered_df["kmeans_cluster"], "K-Means Clusters in PCA Space"
    )
    plot_pca_clusters(
        X_pca, clustered_df["hierarchical_cluster"], "Hierarchical Clusters in PCA Space"
    )

    plt.show()