# EODP Pre-Processing

import numpy as np
import pandas as pd
import re

DATA_PATH = "listings.csv"
VALID_TIME_FRAMES = ["ltm", "l30d", "ly"]
AMENITY_INDICATORS = [
    "Wifi",
    "Kitchen",
    "Air conditioning",
    "Washer",
    "Heating",
    "Parking",
    "Self check-in"
]
REVIEW_COLUMNS = [
    "rating",
    "accuracy",
    "cleanliness",
    "checkin",
    "communication",
    "location",
    "value"
]



def main():
    raw_data = pd.read_csv(DATA_PATH)

    # Create csv of processed data
    processed_data = preprocessing(raw_data)
    processed_data.to_csv("processed_data.csv", index=False)

    # Create csv of only the processed data
    processed_column_names = []

    for column in REVIEW_COLUMNS:
        processed_column_names.append("review_category_" + column)

    for amenity in AMENITY_INDICATORS:
        processed_column_names.append("has_" + amenity.lower().replace(" ", "_"))

    for time in VALID_TIME_FRAMES:
        processed_column_names.append("review_frequency_" + time)

    processed_column_names.append("review_frequency")

    processed_only = processed_data[processed_column_names]
    processed_only.to_csv("processed_only.csv", index=False)




def preprocessing(df):
    df = df.copy()

    # Discretise the review columns
    for column in REVIEW_COLUMNS:
        df = categorise_review_scores(df, column)

    # Categorise the 'number_of_reviews' column
    df = frequency_of_reviews(df)

    # Categorise all other number of reviews columns
    for time_frame in VALID_TIME_FRAMES:
        df = frequency_of_reviews(df, time_frame)

    # Determine if certain amenities exist
    df = amenity_indicators(df)

    return df



def categorise_review_scores(df, review_column):
    from_column = "review_scores_" + review_column
    to_column = "review_category_" + review_column

    df = df.copy()
    data = df[from_column]

    # Calculate the first and third quartile from the original column
    Q1 = np.nanpercentile(data, 25)
    Q3 = np.nanpercentile(data, 75)

    Q3_taken = min(5.0, Q3)

    # Categorise the original review column into Low, Medium, High, according to calculated quartiles, and place it into a new column
    df[to_column] = pd.cut(
        df[from_column],
        bins=[-float("inf"), Q1, Q3_taken, float("inf")],
        labels=["Low", "Medium", "High"],
        right=False   # fix: bins include lower bounds, this also affect Q1
    )

    # Fill in the missing values for the categorised column
    df[to_column] = df[to_column].cat.add_categories('Missing')
    df[to_column] = df[to_column].fillna("Missing")

    return df



def frequency_of_reviews(df, time_frame = ""):
    df = df.copy()

    # Categorise 'number_of_reviews'
    if time_frame == "":
        number_of_reviews = df["number_of_reviews"]
        category_name = "review_frequency"

    # Categroise other number of reviews columns
    elif time_frame in VALID_TIME_FRAMES:
        number_of_reviews = df["number_of_reviews_" + time_frame]
        category_name = "review_frequency_" + time_frame
    else:
        # This should never happen
        raise ValueError("Invalid time frame")

    # Calculate percentiles, without accounting for listings with 0 reviews
    Q1 = np.nanpercentile(number_of_reviews[number_of_reviews > 0], 25)
    Q3 = np.nanpercentile(number_of_reviews[number_of_reviews > 0], 75)

    # Create a new column to categorise
    df[category_name] = pd.cut(
        number_of_reviews,
        bins=[-float("inf"), 0, Q1, Q3, float("inf")],
        labels=["No Reviews","Low", "Medium", "High"]
    )

    return df



def amenity_indicators(df):
    df = df.copy()

    for amenity in AMENITY_INDICATORS:
        column_name = "has_" + amenity.lower().replace(" ", "_")

        # Create a new column determining whether each listing has our desired amenity
        df[column_name] = df["amenities"].apply(
            lambda x: bool(re.search(amenity, str(x), re.IGNORECASE))
        )

    return df



if __name__ == "__main__":
    main()