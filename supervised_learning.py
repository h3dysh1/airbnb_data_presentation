import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
 
DATA_PATH = 'processed_data.csv'
 
Y_COL = 'review_category_rating'
X_COLS = ['host_listings_count', 'latitude', 'longitude', 'accommodates',
          'number_of_reviews', 'availability_365', 'reviews_per_month']
 
FILTER_CATEGORIES = ['Missing']
 
TEST_SIZE = 0.2
RANDOM_STATE = 42
 
KNN_N_NEIGHBORS = 5
KNN_PARAM_GRID = {'n_neighbors': range(1, 31, 2)}  # odd numbers avoid ties
 
DT_PARAM_GRID = {
    'max_depth': [3, 5, 7, 10, 15, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 5, 10],
    'criterion': ['gini', 'entropy']
}
 
CV_FOLDS = 5
SCORING = 'f1_macro'
 
 
def main():
    df = pd.read_csv(DATA_PATH)
    df = filter_df(df)
 
    X = df[X_COLS]
    y = df[Y_COL]
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )
 
    X_train, X_test = impute_and_scale(X_train, X_test)
 
    run_knn(X_train, X_test, y_train, y_test)
    run_decision_tree(X_train, X_test, y_train, y_test)
 
 
def filter_df(df):
    df = df.copy()
 
    for category in FILTER_CATEGORIES:
        df = df[df[Y_COL] != category]
 
    return df
 
 
def impute_and_scale(X_train, X_test):
    num_cols = X_train.select_dtypes(include='number').columns
 
    num_imputer = SimpleImputer(strategy='median')
    X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
    X_test[num_cols] = num_imputer.transform(X_test[num_cols]) 
 
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])
 
    return X_train, X_test
 
 
def run_knn(X_train, X_test, y_train, y_test):
    knn = KNeighborsClassifier(n_neighbors=KNN_N_NEIGHBORS)
    knn.fit(X_train, y_train)
 
    y_pred = knn.predict(X_test)
    report_results('KNN (default)', y_test, y_pred)
 
    grid = GridSearchCV(
        KNeighborsClassifier(),
        KNN_PARAM_GRID,
        cv=CV_FOLDS,          # StratifiedKFold under the hood for classifiers
        scoring=SCORING,      # macro F1 treats all classes equally, good with imbalance
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
 
    print("Best k:", grid.best_params_)
    print("Best CV score:", grid.best_score_)
 
    best_knn = grid.best_estimator_
    y_pred = best_knn.predict(X_test)
    report_results('KNN (tuned)', y_test, y_pred)
 
    return best_knn
 
 
def run_decision_tree(X_train, X_test, y_train, y_test):
    dt = DecisionTreeClassifier(random_state=RANDOM_STATE)
    dt.fit(X_train, y_train)
 
    y_pred = dt.predict(X_test)
    report_results('Decision Tree (default)', y_test, y_pred)
 
    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        DT_PARAM_GRID,
        cv=CV_FOLDS,
        scoring=SCORING,
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
 
    print("Best params:", grid.best_params_)
    print("Best CV score:", grid.best_score_)
 
    best_dt = grid.best_estimator_
    y_pred = best_dt.predict(X_test)
    report_results('Decision Tree (tuned)', y_test, y_pred)
 
    importances = pd.Series(best_dt.feature_importances_, index=X_train.columns)
    importances = importances.sort_values(ascending=False)
    print("\nTop feature importances:\n", importances.head(15))
 
    return best_dt
 
 
def report_results(label, y_test, y_pred):
    print(f"\n--- {label} ---")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
 
 
main()
