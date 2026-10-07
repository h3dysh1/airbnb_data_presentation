from sklearn.feature_selection import mutual_info_classif
import pandas as pd

# mark which X_COLS are genuinely discrete/categorical vs continuous
discrete_mask = [col in (BOOL + CAT) for col in X_COLS]

mi_arr = mutual_info_classif(X_train, y_train, discrete_features=discrete_mask, random_state=1)
mi_scores = pd.Series(mi_arr, index=X_train.columns)

top3_filter = mi_scores.sort_values(ascending=False).head(3)
print("Top 3 (filter — Mutual Information):")
print(top3_filter)

comparison = pd.DataFrame({
    'Embedded (Tree importance)': importances,
    'Filter (MI)': mi_scores
}).sort_values('Filter (MI)', ascending=False)

print(comparison)
