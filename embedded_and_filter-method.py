import re
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import mutual_info_classif



# file to read
DATA_PATH = "processed_data.csv"
# Fixes the random shuffling, same split every run
SEED = 1
# target variable to predict
Y_COL = "review_category_rating"
# 20 features 
X_COLS = ['host_listings_count', 'latitude', 'longitude', 'accommodates',
          'number_of_reviews', 'availability_365', 'reviews_per_month', 'host_has_profile_pic',
          'hosts_time_as_user_years', 'host_is_superhost', 'host_identity_verified',
          'has_wifi', 'has_kitchen', 'has_air_conditioning', 'has_washer', 'has_heating',
          'has_parking', 'has_self_check-in', 'review_frequency', 'price']
# lists the true and false text columns
BOOL = ['host_is_superhost', 'host_identity_verified', 'host_has_profile_pic']


# seeing if there are amenities in the listing, and if so, which ones
BINARY_AMENITY = ['has_wifi', 'has_kitchen', 'has_air_conditioning', 'has_washer',
                  'has_heating', 'has_parking', 'has_self_check-in']
# sets real order of the review-frequency catetgories
REVIEW_FREQ_ORDER = ["No Reviews", "Low", "Medium", "High"]
#  rows that are missing are removed
FILTER_CATEGORIES = ['Missing']
# how many features print
TOP_N = 3



# Helps turn false/true text into 0/1 for boolean columns
def process_bool(val):
    if pd.isna(val):
        return np.nan
    return 1 if val[0].lower() == 't' else 0

# Converts the price column to a float, removing the $ and , characters
def parse_price(price):
    if pd.isna(price):
        return price
    return float(re.sub(r'[$,]', '', price))

# Reads in the data and processes the boolean and price columns
def load():
    df = pd.read_csv(DATA_PATH)
    for c in BOOL:
        df[c] = df[c].apply(process_bool)
    for c in BINARY_AMENITY:
        df[c] = df[c].astype(int)

    df['review_frequency'] = pd.Categorical(
        df['review_frequency'], categories=REVIEW_FREQ_ORDER, ordered=True).codes.astype(float)
    df.loc[df['review_frequency'] < 0, 'review_frequency'] = np.nan


    df['price'] = df['price'].apply(parse_price)
    for c in FILTER_CATEGORIES:
        df = df[df[Y_COL] != c]
    return df

# Helps split the data into training and testing sets, and imputes missing values
def tune_depth(X_train, y_train):
    best_depth, best_score = 0, -1
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for depth in range(1, 20):
        scores = []
        for tr, va in cv.split(X_train, y_train):
            m = DecisionTreeClassifier(max_depth=depth, random_state=42)
            m.fit(X_train.iloc[tr], y_train.iloc[tr])
            scores.append(accuracy_score(y_train.iloc[va], m.predict(X_train.iloc[va])))
        if np.mean(scores) > best_score:
            best_score, best_depth = np.mean(scores), depth
    return best_depth, best_score


def main():
    df = load()
    # Seperating the features and target variable, and splitting into training and testing sets
    X, y = df[X_COLS], df[Y_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)


    # Imputing missing values and scaling the data
    X_train, X_test = X_train.copy(), X_test.copy()
    cols = X_train.columns
    imp = SimpleImputer(strategy='median')
    X_train[cols] = imp.fit_transform(X_train[cols]); X_test[cols] = imp.transform(X_test[cols])

    discrete_cols = BOOL + BINARY_AMENITY + ['review_frequency']
    continuous_cols = [c for c in cols if c not in discrete_cols]

    sc = StandardScaler()
    X_train[continuous_cols] = sc.fit_transform(X_train[continuous_cols])
    X_test[continuous_cols] = sc.transform(X_test[continuous_cols])


    # Finds the best depth and trains the final model, then evaluates on the test set
    depth, cv_score = tune_depth(X_train, y_train)
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42).fit(X_train, y_train)
    test_acc = accuracy_score(y_test, tree.predict(X_test))

    # Prints the results of the model evaluation and feature importance
    importances = pd.Series(tree.feature_importances_, index=X_COLS).sort_values(ascending=False)


    print(f"Rows used: {len(df)} (train {len(X_train)}, test {len(X_test)})")
    print(f"Best max_depth = {depth} (CV accuracy {cv_score:.4f}); test accuracy = {test_acc:.4f}\n")
    print("Decision Tree feature importances (sum to 1):")
    print(importances.round(4).to_string())
    print(f"\nTop {TOP_N} features (embedded, Decision Tree):")
    for rank, (feat, val) in enumerate(importances.head(TOP_N).items(), start=1):
        print(f"  {rank}. {feat}: {val:.4f}")
 
    importances.rename("tree_importance").round(5).to_csv("decision_tree_importances.csv")
    print("\nSaved decision_tree_importances.csv")

      # Filter method: Mutual Information
    #Seperates the discrete and continuous features for mutual information calculation
    discrete_mask = [col in (BOOL + BINARY_AMENITY + ['review_frequency']) for col in X_COLS]

     # Calculates the mutual information scores for each feature and prints the top 3 features based on mutual information and decision tree importance
    mi_arr = mutual_info_classif(X_train, y_train, discrete_features=discrete_mask, random_state=SEED)
    mi_scores = pd.Series(mi_arr, index=X_train.columns)

    top3_filter = mi_scores.sort_values(ascending=False).head(3)
    top3_embedded = importances.head(TOP_N)

    print("Top 3 (filter — Mutual Information):")
    print(top3_filter.round(4))
    print("\nTop 3 (embedded — Decision Tree importance):")
    print(top3_embedded.round(4))

  
main()

