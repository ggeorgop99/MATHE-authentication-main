import pandas as pd


def header(filepath):
  df = pd.read_csv(filepath)
  return df.head()

