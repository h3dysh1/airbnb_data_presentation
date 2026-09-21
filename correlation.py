import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score

DATA_PATH = "processed_data.csv"   # output of preprocessing_2.py, NOT processed_only.csv
REFERENCE_DATE = pd.Timestamp("2026-07-07")  # set this to your dataset's scrape date
 
df = pd.read_csv(DATA_PATH)
 

 