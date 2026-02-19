import pandas as pd
import numpy as np                              
df = pd.read_csv("data_cleaning_assignment/dirty_dataset_1000.csv")
print(df.head())
print(df.isnull())
df.fillna(df.mean(numeric_only=True), inplace=True)
df.fillna("Unknown", inplace=True)
df.drop_duplicates(inplace=True)
print(df.info())
df.to_csv("cleaned_dataset.csv", index=False)

