import numpy as np
import pandas as pd
from sklearn import preprocessing
import sklearn.model_selection
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def write_answer(answer, filename):
    test_df = pd.read_csv("test.csv")
    assert isinstance(answer, np.ndarray)
    assert answer.size == len(test_df)
    answer = answer.astype(int)
    answer_df = pd.DataFrame(
        {"PassengerId": test_df["PassengerId"], "Survived": answer}
    )
    answer_df.to_csv(filename, index=False)

def encode(df):
    del df["PassengerId"]
    for col in df.columns:
        if df[col].dtype is np.float64:
            continue
        if len(df[col].unique()) > 5:
            df = df.drop(columns=col)
            pass
        continue
    return df

# def preprocess(df):
#     del df["PassengerId"]
#     for col in df.columns:
#         if df[col].dtype is np.float64:

#     for col in ["Sex", "Ticket", "Cabin", "Embarked"]:
#         le = preprocessing.LabelEncoder()
#         le.fit(df[col].values)
#         df[col] = le.transform(df[col].values)
#     df["Age"] = df["Age"].fillna(df["Age"].mean())
#     df["Name"] = df["Name"].map(lambda s: len(s))
#     return df

df_train = pd.read_csv('train.csv')
df_test =  pd.read_csv('test.csv')

# s = df_train.describe(include="all")
# for c in s.columns:
#     print(s[c])
#     print()

# imp = IterativeImputer(max_iter=10, random_state=0).fit(df_train)
# new_train = imp.transform(df_train)

print(encode(df_train).describe())