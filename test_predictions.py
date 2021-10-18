import pandas as pd
import numpy as np

df = pd.read_csv('pred.csv')
df2 = pd.read_csv('pred1.csv')

print(f'One: {df[df["target"] >= 0.5]["target"].count()}, Zero: {df[df["target"] < 0.5]["target"].count()}')

# df1 = df1['target'].apply(lambda x: round(x))
# df2 = df2['target'].apply(lambda x: round(x))

# print((df1 == df2).all())