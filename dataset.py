import pandas as pd
import numpy as np

from imblearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer

class Dataset:
    
    hex_columns = ['f2', 'f3', 'f13', 'f18', 'f20', 'f26']
    bool_columns = ['f28', 'f22', 'f14']
    ordinal_columns = ['f6', 'f8', 'f16', 'f17', 'f19', 'f21', 'f25']
    categorical_columns = ['f1', 'f5', 'f7', 'f9', 'f12', 'f23', 'f24', 'f27']
    numerical_columns = ['f4']
    alphabet_columns = ['f0']
    all_columns = ['f' + str(i) for i in range(0, 28)]

    def __init__(self, train_path : str, test_path : str) -> None:
        self.train_path = train_path
        self.test_path = test_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y = None

    def load(self) -> tuple:
        self.load_train(self.train_path)
        # self.load_test(self.test_path)
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
        self.conv_columns(df)
        self.fillna(df)
        self.transform(df, test)
    
                
    def remove_duplicate_columns(self, df : pd.DataFrame) -> None:
        cols_to_drop = []
        cols = list(df.columns)
        for col in df.columns:
            if col in cols:
                cols.remove(col)
            for col2 in cols:
                if df[col].equals(df[col2]):
                    cols_to_drop.append(col2)
        df.drop(columns=cols_to_drop, inplace=True)

    def conv_columns(self, df : pd.DataFrame) -> None:
        self.conv_hex(df)
        self.conv_bool(df)
    
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
            df.iloc[:,col_loc] = df[col].apply(lambda x: conv_hex_map(x)).values

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
            df.iloc[:,col_loc] = df[col].apply(lambda x: conv_bool_map(x)).values

    def fillna(self, df : pd.DataFrame) -> None:
        for col in self.numerical_columns + self.ordinal_columns + self.hex_columns:
            if col not in list(df.columns):
                continue
            df[col].fillna(np.mean(df[col]))
    
    def transform(self, df : pd.DataFrame, test : bool) -> None:
        numeric_transformer = make_pipeline(
            IterativeImputer(random_state=42),
            StandardScaler()
        )

        categorical_transformer = make_pipeline(
            SimpleImputer(strategy='constant', fill_value='missing'),
            OneHotEncoder(handle_unknown='ignore')
        )
            
        preprocessor = make_column_transformer( 
            (numeric_transformer, self.numerical_columns + self.hex_columns + self.bool_columns + self.ordinal_columns),
            (categorical_transformer, self.categorical_columns + self.alphabet_columns),
        )

        if test:
            self.X_test = pd.DataFrame(preprocessor.transform(self.X_test).toarray())
        else:
            # print(self.X_train.head())
            self.X_train = pd.DataFrame(preprocessor.fit_transform(self.X_train, self.y).toarray())
        

if __name__ == '__main__':
    dataset = Dataset('./train.csv', './test.csv')
    
    X_train, y, X_test = dataset.load()

    print(X_train[271].unique())