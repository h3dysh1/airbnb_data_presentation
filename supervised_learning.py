import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
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
    )

    # Fill out the missings values (impute), scale the values
    X_train, X_test = impute_and_scale(X_train, X_test)

    #KNN
    knn(X_train, X_test, y_train, y_test)

    # DEcision Tree
    decision_tree(X_train, X_test, y_train, y_test)


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



def knn(X_train, X_test, y_train, y_test):
    k_values = range(1, 50, 1)
    n = 5
    nf_CV = KFold(n_splits=n, shuffle=True, random_state=42)

    best_k = None
    best_score = -1

    for k in k_values:
        fold_scores = []

        for train_idx, test_idx in nf_CV.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[test_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[test_idx]

            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(X_tr, y_tr)

            y_pred = model.predict(X_val)
            fold_scores.append(accuracy_score(y_val, y_pred))

        mean_score = np.mean(fold_scores)
        print(f"k={k}: mean accuracy = {mean_score:.4f}")

        if mean_score > best_score:
            best_score = mean_score
            best_k = k

    print(f"\nBest k: {best_k}, score: {best_score:.4f}")

    best_knn = KNeighborsClassifier(n_neighbors=best_k)
    best_knn.fit(X_train, y_train)

    return best_knn



def decision_tree(X_train, X_test, y_train, y_test):
    depth_values = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, None]
    n = 5
    nf_CV = KFold(n_splits=n, shuffle=True, random_state=42)

    best_depth = None
    best_score = -1

    for depth in depth_values:
        fold_scores = []

        for train_idx, test_idx in nf_CV.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[test_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[test_idx]

            model = DecisionTreeClassifier(max_depth=depth, random_state=42)
            model.fit(X_tr, y_tr)

            y_pred = model.predict(X_val)
            fold_scores.append(accuracy_score(y_val, y_pred))

        mean_score = np.mean(fold_scores)
        print(f"max_depth={depth}: mean accuracy = {mean_score:.4f}")

        if mean_score > best_score:
            best_score = mean_score
            best_depth = depth

    print(f"\nBest max_depth: {best_depth}, score: {best_score:.4f}")

    best_dt = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
    best_dt.fit(X_train, y_train)

    return best_dt



main()