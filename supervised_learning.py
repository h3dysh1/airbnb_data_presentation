import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier
import re
 
DATA_PATH = 'processed_data.csv'
 
Y_COL = 'review_category_rating'
X_COLS = ['host_listings_count', 'latitude', 'longitude', 'accommodates',
          'number_of_reviews', 'availability_365', 'reviews_per_month', 'host_has_profile_pic',
          'hosts_time_as_user_years', 'host_is_superhost', 'host_identity_verified',
          'has_wifi', 'has_kitchen', 'has_air_conditioning', 'has_washer', 'has_heating',
          'has_parking', 'has_self_check-in', 'review_frequency', 'price']

BOOL = ['host_is_superhost', 'host_identity_verified', 'host_has_profile_pic',]
CAT = ['review_frequency', 'has_wifi', 'has_kitchen', 'has_air_conditioning', 'has_washer', 'has_heating',
          'has_parking', 'has_self_check-in']
STR = ['price']
 
FILTER_CATEGORIES = ['Missing']

TEST_SPLIT = 0.2

KNN_N_NEIGHBORS = 5


def main():
    df = pd.read_csv(DATA_PATH)
    # Process data
    df = process_df(df)
    df = filter_df(df) # Filter out the rows with unwanted values
 
    X = df[X_COLS]
    y = df[Y_COL]

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SPLIT,
        stratify=y,
        random_state=42
    )

    # Fill out the missings values (impute), scale the values
    X_train, X_test = impute_and_scale(X_train, X_test)

    #KNN
    knn = run_knn(X_train, X_test, y_train, y_test)

    # DEcision Tree
    dt = run_decision_tree(X_train, X_test, y_train, y_test)
    feature_importance(dt, X_train.columns)

    # Baseline
    baseline_score = baseline(X_train, X_test, y_train, y_test)

    knn_test_score = accuracy_score(y_test, knn.predict(X_test))
    dt_test_score = accuracy_score(y_test, dt.predict(X_test))

    print(f"\nKNN test accuracy: {knn_test_score} (baseline: {baseline_score}")
    print(f"Decision Tree test accuracy: {dt_test_score} (baseline: {baseline_score}")

    bootstrap_ci(knn, X_test, y_test)

    save_predictions(X_test, y_test, {'knn': knn, 'decision_tree': dt})


def process_df(df):
    df = df.copy()

    # Process Booelan Values
    for column in BOOL:
        df[column] = df[column].apply(process_bool)

    # Process Categorical Values
    for column in CAT:
        categories = list(set(df[column].values))
        cat_dict = {}

        for i in range(len(categories)):
            cat_dict[categories[i]] = i

        df[column] = df[column].apply(process_cat, cat_dict=cat_dict)


    # Process price
    df['price'] = df['price'].apply(parse_price)

    return df



def process_bool(val):
    if pd.isna(val):
        return np.nan

    if val[0].lower() == 't':
        return 1
    elif val[0].lower() == 'f':
        return 0



def process_cat(cat, cat_dict):
    return cat_dict[cat]




def parse_price(price):
    # Check if the current price is a string
    if not pd.isna(price):
        # Remove the `$` and `,` symbols from the string
        cleaned_price = re.sub(r'[$,]', '', price)
        return (float(cleaned_price))
    else:
        # If price is Nan, just keep it as NaN
        return price



def filter_df(df):
    df = df.copy()

    for category in FILTER_CATEGORIES:
        df = df[df[Y_COL] != category]
 
    return df


def impute_and_scale(X_train, X_test):
    num_cols = X_train.select_dtypes(include='number').columns

    # Impute with median values
    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    X_train[num_cols] = imputer.fit_transform(X_train[num_cols])
    X_test[num_cols] = imputer.transform(X_test[num_cols]) 

    # Scale the values for each column
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])
 
    return X_train, X_test



def run_knn(X_train, X_test, y_train, y_test):
    k_values = range(1, 200, 5)
    five_f_CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    best_k = 0
    best_score = -1

    for k in k_values:  
        results = []

        for train_idx, test_idx in five_f_CV.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[test_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[test_idx]

            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_tr, y_tr)

            y_pred = knn.predict(X_val)
            results.append(accuracy_score(y_val, y_pred))

        mean_score = np.mean(results)
        print(f"k={k}: mean accuracy = {mean_score:.4f}")

        if mean_score > best_score:
            best_score = mean_score
            best_k = k

    print(f"\nBest k: {best_k}, score: {best_score}")

    best_knn = KNeighborsClassifier(n_neighbors=best_k)
    best_knn.fit(X_train, y_train)

    return best_knn



def run_decision_tree(X_train, X_test, y_train, y_test):
    depth_values = range(1, 20)
    n = 5
    nf_CV = StratifiedKFold(n_splits=n, shuffle=True, random_state=42)

    best_depth = 0
    best_score = -1

    for depth in depth_values:
        results = []

        for train_idx, test_idx in nf_CV.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[test_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[test_idx]

            dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
            dt.fit(X_tr, y_tr)

            y_pred = dt.predict(X_val)
            results.append(accuracy_score(y_val, y_pred))

        mean_score = np.mean(results)
        print(f"max_depth={depth}: mean accuracy = {mean_score:.4f}")

        if mean_score > best_score:
            best_score = mean_score
            best_depth = depth

    print(f"\nBest max_depth: {best_depth}, score: {best_score:.4f}")

    best_dt = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
    best_dt.fit(X_train, y_train)

    return best_dt



# All code below has been AI slopped ill write it myself soonish

def baseline(X_train, X_test, y_train, y_test):
    dummy = DummyClassifier(strategy='most_frequent')
    dummy.fit(X_train, y_train)

    y_pred = dummy.predict(X_test)
    score = accuracy_score(y_test, y_pred)

    print(f"Baseline (majority class) accuracy: {score:.4f}")

    return score


def bootstrap_ci(model, X_test, y_test, n_iterations=10):
    scores = []
    n = len(X_test)

    for i in range(n_iterations):
        idx = np.random.choice(n, size=n, replace=True)
        X_sample = X_test.iloc[idx]
        y_sample = y_test.iloc[idx]

        y_pred = model.predict(X_sample)
        scores.append(accuracy_score(y_sample, y_pred))

    scores = np.array(scores)
    lower = np.percentile(scores, 2.5)
    upper = np.percentile(scores, 97.5)

    print(f"Bootstrap 95% CI: [{lower:.4f}, {upper:.4f}] (mean={scores.mean():.4f})")

    return lower, upper



def feature_importance(model, feature_names):
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=False)

    print("\nFeature importances (Decision Tree):")
    print(importances)

    return importances


def save_predictions(X_test, y_test, models: dict, path='predictions.csv'):
    output = pd.DataFrame(index=X_test.index)
    output['actual'] = y_test

    for name, model in models.items():
        output[f'predicted_{name}'] = model.predict(X_test)

    output.to_csv(path, index=True)
    print(f"\nSaved predictions to {path}")

    return output


main()