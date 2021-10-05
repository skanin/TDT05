import pandas as pd
import numpy as np

from collections import namedtuple
from imblearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from catboost import CatBoostClassifier
class Dataset:
    
    hex_columns = ['f2', 'f3', 'f13', 'f18', 'f20', 'f26']
    bool_columns = ['f28', 'f22', 'f14']
    ordinal_columns = ['f6', 'f8', 'f16', 'f17', 'f19', 'f21', 'f25']
    categorical_columns = ['f1', 'f5', 'f7', 'f9', 'f11', 'f24']
    numerical_columns = ['f4']
    alphabet_columns = ['f0']
    ordinal_cat_columns = ['f0', 'f10', 'f12', 'f23', 'f27']
    binary_columns = ['f10']
    all_columns = ['f' + str(i) for i in range(0, 28)]
    removed_cols = []

    def __init__(self, train_path : str, test_path : str) -> None:
        self.train_path = train_path
        self.test_path = test_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y = None

        self.numeric_transformer = make_pipeline(
            IterativeImputer(random_state=42),
            StandardScaler()
        )

        self.categorical_transformer = make_pipeline(
            SimpleImputer(strategy='most_frequent'),
            OrdinalEncoder()
        )

        self.ordinal_transformer = make_pipeline(
            SimpleImputer(strategy='most_frequent'),
            OrdinalEncoder()
        )

        numerical_transformer_cols = self.numerical_columns + self.hex_columns + self.bool_columns + self.ordinal_columns
        categorical_transformer_cols = self.categorical_columns + self.alphabet_columns

        categorical_transformer_cols.remove('f11')

        self.preprocessor = make_column_transformer(
            (self.numeric_transformer, numerical_transformer_cols),
            (self.categorical_transformer, categorical_transformer_cols),
            (self.ordinal_transformer, self.ordinal_cat_columns)
        )

    def load(self) -> tuple:
        self.load_train(self.train_path)
        self.load_test(self.test_path)


        return self.X_train, self.y, self.X_test

    def load_train(self, dataset : str, delimiter = ',') -> None:
        self.df = pd.read_csv(dataset, delimiter=delimiter)
        self.X_train = self.df.drop(columns=['id', 'target'])
        self.y = self.df['target']
        self.X = self.df.drop(columns=['id', 'target'])

        self.preprocess_train(self.X_train)

    def load_test(self, dataset : str, delimiter = ',') -> None:
        self.df = pd.read_csv(dataset, delimiter = delimiter)
        self.X_test = self.df.drop(columns=['id'])

        self.preprocess_test(self.X_test)

    def preprocess_test(self, df : pd.DataFrame) -> None:
        df.loc[26648, 'f9'] = np.nan
        df.loc[20956, 'f15'] = np.nan
        df.loc[21034, 'f15'] = np.nan

        self.preprocess_train(df, test=True)

    def preprocess_train(self, df : pd.DataFrame, test = False):
        self.remove_duplicate_columns(df)
        self.remove_duplicates(df)
        
        self.conv_columns(df)
        # self.fillna(df)
        self.transform(df, test)
                
    def remove_duplicate_columns(self, df : pd.DataFrame) -> None:
        for col in self.categorical_columns:
            df[col] = df[col].factorize()[0]

        cols_to_drop = []
        cols = list(df.columns)
        for col in df.columns:
            if col in cols:
                cols.remove(col)
            for col2 in cols:
                if df[col].equals(df[col2]):
                    cols_to_drop.append(col2)

        self.removed_cols = list(set(self.removed_cols + cols_to_drop))
        df.drop(columns=cols_to_drop, inplace=True)

    def conv_columns(self, df : pd.DataFrame) -> None:
        self.conv_hex(df)
        self.conv_bool(df)
        self.conv_binary(df)
    
    def conv_hex(self, df : pd.DataFrame) -> None:
        def conv_hex_map(x):
            try:
                return int(x, 16)
            except ValueError as e:
                return np.nan
            except TypeError as e:
                return np.nan    
        
        for col in self.hex_columns:
            if col not in list(df.columns):
                continue
            col_loc = list(df.columns).index(col)
            df[col] = df[col].apply(lambda x: conv_hex_map(x))

    def conv_bool(self, df : pd.DataFrame):
        def conv_bool_map(x):
            try:
                if not type(x) == str:
                    return x
                if x.lower() == 'f':
                    return 0
                elif x.lower() == 't':
                    return 1
                return x
            except Exception as e:
                return np.nan
        
        for col in self.bool_columns:
            if col not in list(df.columns):
                continue
            col_loc = list(df.columns).index(col)
            df[col] = df[col].apply(lambda x: conv_bool_map(x))

    def conv_binary(self, df : pd.DataFrame):
        def binaryToDecimal(binary):
            try:
                binary = int(binary)
            except:
                return np.nan
            
            binary1 = binary
            decimal, i, n = 0, 0, 0
            while(binary != 0):
                dec = binary % 10
                decimal = decimal + dec * pow(2, i)
                binary = binary//10
                i += 1
            return decimal
        for col in self.binary_columns:
            if col not in list(df.columns):
                continue
        df[col] = df[col].apply(lambda x: binaryToDecimal(x))    
    # def fillna(self, df : pd.DataFrame) -> None:
    #     for col in self.numerical_columns + self.ordinal_columns + self.hex_columns:
    #         if col not in list(df.columns):
    #             continue
    #         df[col].fillna(np.mean(df[col]))
    
    def transform(self, df : pd.DataFrame, test : bool) -> None:
        if test:
            self.X_test = pd.DataFrame(self.preprocessor.transform(self.X_test))
            print(pd.DataFrame({'total_missing': self.X_test.isnull().sum(), 'perc_missing': (self.X_test.isnull().sum()/50000)*100}))
        else:
            # print(self.X_train.head())
            self.X_train = pd.DataFrame(self.preprocessor.fit_transform(self.X_train, self.y))
            print(pd.DataFrame({'total_missing': self.X_train.isnull().sum(), 'perc_missing': (self.X_train.isnull().sum()/50000)*100}))
    
    def remove_duplicates(self, df : pd.DataFrame) -> None:
        df.drop_duplicates(inplace=True)
        

def split(X : pd.DataFrame, y : pd.DataFrame, size=.2):
    return train_test_split(X, y, test_size=size, stratify=y)


def fit(data : pd.DataFrame, labels : pd.DataFrame, classifier : any) -> None:
    classifier.fit(data, labels)

    # return classifier.predict_proba(test)
    # return classifier.predict(test) if hasattr(classifier, "decision_function") else classifier.predict_proba(test)

def predict_proba(classifier : any, test : pd.DataFrame) -> np.array:
    return classifier.predict_proba(test)

def predict(classifier : any, test : pd.DataFrame) -> np.array:
    return classifier.predict(test)


def get_classifier(name : str) -> namedtuple:
    classifiers = {
        'adaboost': AdaBoostClassifier(n_estimators=100, random_state=0),
        'gnb': GaussianNB(),
        'lgbm': LGBMClassifier(),
        'rf': RandomForestClassifier(n_estimators=100),
    }

    if name not in list(classifiers.keys()):
        raise Exception(f'No classifier named {name}')
    
    return classifiers[name]

def print_accuracy(pred : np.array, test : pd.DataFrame, name : str):
    print(name + ' Model accuracy score: {0:0.4f}'.format(roc_auc_score(test, pred[:, 1])))

def save_predictions(preds : np.ndarray):
    pred_df = pd.DataFrame(preds.reshape(-1, 1))
    print(pred_df[0])



    keys = [i for i in range(50000, 100000)]

    # pred_df.set_index(keys)
    pred_df.insert(0, "id", keys)
    pred_df.rename(columns={0: 'target'}, inplace=True)

    pred_df.to_csv('./predictions10.csv', index=False)

if __name__ == '__main__':
    dataset = Dataset('tdt05-2021-challenge-1/train.csv', 'tdt05-2021-challenge-1/test.csv')
    
    X_train, y, X_test = dataset.load()
    
    X_train_split, X_test_split, y_train_split, y_test_split = split(X_train, y)

    # clf = get_classifier('adaboost')
    classifiers = {
        'adaboost': AdaBoostClassifier(n_estimators=100, random_state=0),
        'gnb': GaussianNB(),
        'lgbm': LGBMClassifier(),
        'rf': RandomForestClassifier(n_estimators=100),
        'catboost': CatBoostClassifier(iterations=10000, learning_rate=.1, depth=2)
    }

    # clf = StackingClassifier(estimators = list(classifiers.items()), final_estimator = CatBoostClassifier(iterations=10000, learning_rate=.1, depth=2))
    clf = classifiers['adaboost']
    print('Starting prediction')
    fit(X_train_split, y_train_split, clf)
    print('Prediction done')
    
    prediction = predict_proba(clf, X_test_split)

    print_accuracy(prediction, y_test_split, 'adaboost')

    real_pred = predict_proba(clf, X_test)

    f = open("./pred.csv", "w")
    f.write("id,target\n")
    id_nr = 50000
    for v in real_pred[:, 1]:
        f.write(f"{id_nr},{v}\n")
        id_nr += 1
    f.close()
    