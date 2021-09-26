import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')


X = df_train.drop('target', axis=1)
y = df_train['target']

X_test = df_test

print(X.shape)
print(y.shape)
print(X_test.shape)

lgbm = LGBMClassifier(objective='multiclass', random_state=5)

lgbm.fit(X, y)

y_pred = lgbm.predict(X_test)