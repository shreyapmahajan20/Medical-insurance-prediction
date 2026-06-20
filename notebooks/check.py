import pandas as pd
df = pd.read_csv("medical-insurance-prediction\data\insurance.csv")
print(df.shape)
print(df.head())
print(df.dtypes)