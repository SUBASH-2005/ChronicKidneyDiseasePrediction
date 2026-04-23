import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_and_clean_data(path):

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    # remove id column
    if "id" in df.columns:
        df.drop("id", axis=1, inplace=True)

    # remove extra spaces
    for col in df.select_dtypes("object"):
        df[col] = df[col].str.strip()

    # fill missing values (CORRECT WAY)
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

    # encode categorical columns
    for col in df.select_dtypes(include="object"):
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    return df
