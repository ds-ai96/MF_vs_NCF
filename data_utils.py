import numpy as np
import pandas as pd
import scipy.sparse as sp

def load_data():
    data = pd.read_csv("/data/ratings.dat", sep="::", header=None, names=["UserID", "MovieID", "Rating", "Timestamp"])

