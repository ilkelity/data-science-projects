import sys
sys.path.append("C:/Users/lenovo/Desktop/ds lectures/helpers")
import eda
import data_prep as dp

import pandas as pd
df = pd.read_csv(r"C:\Users\lenovo\Desktop\ds lectures\helpers\titanic.csv")
df.head()

def titanic_data_prep(df):
    df.columns = [col.upper() for col in df.columns]

    #############################################
    # 1. Feature Engineering (Değişken Mühendisliği)
    #############################################

    # Cabin bool
    df["NEW_CABIN_BOOL"] = df["CABIN"].notnull().astype('int')
    # Name count
    df["NEW_NAME_COUNT"] = df["NAME"].str.len()
    # name word count
    df["NEW_NAME_WORD_COUNT"] = df["NAME"].apply(lambda x: len(str(x).split(" ")))
    # name dr
    df["NEW_NAME_DR"] = df["NAME"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
    # name title
    df['NEW_TITLE'] = df.NAME.str.extract(' ([A-Za-z]+)\.', expand=False)
    # family size
    df["NEW_FAMILY_SIZE"] = df["SIBSP"] + df["PARCH"] + 1
    # age_pclass
    df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]
    # is alone
    df.loc[((df['SIBSP'] + df['PARCH']) > 0), "NEW_IS_ALONE"] = "NO"
    df.loc[((df['SIBSP'] + df['PARCH']) == 0), "NEW_IS_ALONE"] = "YES"
    # age level
    df.loc[(df['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
    df.loc[(df['AGE'] >= 18) & (df['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
    df.loc[(df['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'
    # sex x age
    df.loc[(df['SEX'] == 'male') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
    df.loc[(df['SEX'] == 'male') & ((df['AGE'] > 21) & (df['AGE']) < 50), 'NEW_SEX_CAT'] = 'maturemale'
    df.loc[(df['SEX'] == 'male') & (df['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniormale'
    df.loc[(df['SEX'] == 'female') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
    df.loc[(df['SEX'] == 'female') & ((df['AGE'] > 21) & (df['AGE']) < 50), 'NEW_SEX_CAT'] = 'maturefemale'
    df.loc[(df['SEX'] == 'female') & (df['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniorfemale'

    df.head()
    df.shape

    # şimdi değişkenlerin türlerine göre isimlerini tutalım:
    cat_cols, num_cols, cat_but_car = eda.grab_col_names(df)

    num_cols = [col for col in num_cols if "PASSENGERID" not in col]

    for col in num_cols:
        print(col, dp.check_outlier(df, col))

    for col in num_cols:
        dp.replace_with_thresholds(df, col)

    for col in num_cols:
        print(col, dp.check_outlier(df, col))

    #############################################
    # 3. Missing Values (Eksik Değerler)
    #############################################

    dp.missing_values_table(df)

    df.drop("CABIN", inplace=True, axis=1)

    dp.missing_values_table(df)

    remove_cols = ["TICKET", "NAME"]
    df.drop(remove_cols, inplace=True, axis=1)

    dp.missing_values_table(df)

    df["AGE"] = df["AGE"].fillna(df.groupby("NEW_TITLE")["AGE"].transform("median"))

    df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]

    df.loc[(df['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
    df.loc[(df['AGE'] >= 18) & (df['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
    df.loc[(df['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'

    df.loc[(df['SEX'] == 'male') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
    df.loc[(df['SEX'] == 'male') & ((df['AGE'] > 21) & (df['AGE']) < 50), 'NEW_SEX_CAT'] = 'maturemale'
    df.loc[(df['SEX'] == 'male') & (df['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniormale'
    df.loc[(df['SEX'] == 'female') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
    df.loc[(df['SEX'] == 'female') & ((df['AGE'] > 21) & (df['AGE']) < 50), 'NEW_SEX_CAT'] = 'maturefemale'
    df.loc[(df['SEX'] == 'female') & (df['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniorfemale'

    dp.missing_values_table(df)

    df = df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0)

    #############################################
    # 4. Label Encoding
    #############################################

    binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
                   and df[col].nunique() == 2]

    for col in binary_cols:
        df = dp.label_encoder(df, col)

    #############################################
    # 5. Rare Encoding
    #############################################

    dp.rare_analyser(df, "SURVIVED", cat_cols)

    # Bu değişkene rare encoding yapacağız.
    # Sonrasında tüm kategorik değişkenlere OHE uygulayacağız.

    df = dp.rare_encoder(df, 0.01)
    df["NEW_TITLE"].value_counts()
    dp.rare_analyser(df, "SURVIVED", cat_cols)

    #############################################
    # 6. One-Hot Encoding
    #############################################

    ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

    df = dp.one_hot_encoder(df, ohe_cols)

    cat_cols, num_cols, cat_but_car = eda.grab_col_names(df)

    num_cols = [col for col in num_cols if "PASSENGERID" not in col]

    dp.rare_analyser(df, "SURVIVED", cat_cols)

    (df["SEX"].value_counts() / len(df) < 0.01).any()

    (df["NEW_NAME_WORD_COUNT_9"].value_counts() / len(df) < 0.01).any()

    useless_cols = [col for col in df.columns if df[col].nunique() == 2 and
                    (df[col].value_counts() / len(df) < 0.01).any(axis=None)]

    # df.drop(useless_cols, axis=1, inplace=True)

    #############################################
    # 7. Standart Scaler
    #############################################

    # Dikkat! Standartlaştırma her zaman gerekmiyor.
    # Ağaç yöntemlerinde özellikle gerekmiyor.
    from sklearn.preprocessing import StandardScaler

    num_cols

    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    df[num_cols].head()

    from helpers.eda import check_df
    check_df(df)
    return df

## TASK 3
df_ = titanic_data_prep(df)
# df.to_pickle(path="./datasets/titanic.pkl")



