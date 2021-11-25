# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

from sklearn.linear_model import BayesianRidge, Ridge
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.decomposition import PCA





df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')


y = df_train['SalePrice']
#df_train.drop('SalePrice', axis=1, inplace=True)
df_init = pd.concat([df_train, df_test], axis=0, sort=False)
print('df_init1:', df_init)


#  Formatting error

df_init["MSZoning"] = df_init["MSZoning"].replace('C (all)', 'C')
df_init["MasVnrType"] = df_init["MasVnrType"].fillna("None")
df_init["Alley"] = df_init["Alley"].fillna("None")
df_init["BsmtQual"] = df_init["BsmtQual"].fillna("None")
df_init["BsmtCond"] = df_init["BsmtCond"].fillna("None")
df_init["BsmtExposure"] = df_init["BsmtExposure"].fillna("None")
df_init["BsmtFinType1"] = df_init["BsmtFinType1"].fillna("None")
df_init["BsmtFinType2"] = df_init["BsmtFinType2"].fillna("None")
df_init["FireplaceQu"] = df_init["FireplaceQu"].fillna("None")
df_init["GarageType"] = df_init["GarageType"].fillna("None")
df_init["GarageFinish"] = df_init["GarageFinish"].fillna("None")
df_init["GarageQual"] = df_init["GarageQual"].fillna("None")
df_init["GarageCond"] = df_init["GarageCond"].fillna("None")
df_init["PoolQC"] = df_init["PoolQC"].fillna("None")
df_init["Fence"] = df_init["Fence"].fillna("None")
df_init["MiscFeature"] = df_init["MiscFeature"].fillna("None")
df_init2 = df_init.copy()
print('df_init2:', df_init2)


# NaN for Numericals
columns = df_init2.select_dtypes(include=['float64', 'int64']).columns
df_num = df_init2.loc[:, columns]
imputer = KNNImputer(n_neighbors=2)
df_num = pd.DataFrame(
    imputer.fit_transform(df_num),
    columns=list(
        df_num.columns))


# Types Float to Int64
col_names = ["Id",
             "BsmtFullBath",
             "BsmtHalfBath",
             "GarageYrBlt",
             "GarageCars",
             "YearBuilt",
             "YearRemodAdd",
             "FullBath",
             "HalfBath",
             "BedroomAbvGr",
             "KitchenAbvGr",
             "TotRmsAbvGrd",
             "Fireplaces",
             "MoSold",
             "YrSold",
             "MiscVal"]
df_num[col_names] = df_num[col_names].astype('int64')

# Int64 to object categorical
col_names = ["MSSubClass", "OverallQual", "OverallCond"]
for col in col_names:
    df_num[col] = df_num[col].astype(str, copy=False)


# Impute NaN in categoricals
df_cat = df_init2.select_dtypes(include=['object'])

to_impute = ['MSZoning',
             'Utilities',
             'Exterior1st',
             'Exterior2nd',
             'Electrical',
             'KitchenQual',
             'Functional',
             'SaleType']
for column in to_impute:
    df_cat[column].fillna(df_cat[column].mode()[0], inplace=True)

# Clean dataframe
df_cat = df_cat.reset_index()
df_cleaned = pd.concat([df_num, df_cat], axis=1)
df_cleaned = df_cleaned.drop(["index"], axis=1)
df_cleaned1 = df_cleaned.copy()
########################################################################

# Object to int64 categorical
columns_list = ["ExterQual", "ExterCond", "BsmtQual",
                "BsmtCond", "BsmtExposure", "HeatingQC",
                "KitchenQual", "FireplaceQu", "GarageQual",
                "GarageCond", "PoolQC"]

# Replaced by Int64 notation
for column in columns_list:
    df_cleaned1[column] = df_cleaned1[column].replace(
        ['Gd', 'TA', 'Ex', 'Fa', 'Po', 'None', 'No', 'Mn', 'Av'], [7, 5, 9, 3, 2, 0, 0, 4, 5])
    df_cleaned1["OverallCond"] = pd.to_numeric(df_cleaned1["OverallCond"])
# to int64
df_cleaned1["OverallQual"] = pd.to_numeric(df_cleaned1["OverallQual"])
df_cleaned1["OverallCond"] = df_cleaned1["OverallCond"].astype(int)
df_cleaned1["OverallQual"] = df_cleaned1["OverallQual"].astype(int)
df_cleaned2 = df_cleaned1.copy()
########################################################################


df_num1 = df_cleaned2.select_dtypes(include=[np.number])
df_num2 = pd.concat([df_num1, y], axis=1)
print("df_num2: ", df_num2.shape)
df_num2s = np.split(df_num2, [1460], axis=0)
print("df_num2s: ", df_num2s[0].shape)

# Drop outliers (1)
df_num2s[0] = df_num2s[0].loc[df_num2s[0][
    'GarageCars'] != df_num2s[0]['GarageCars'].max()]
for i in range(2):
    df_num2s[0] = df_num2s[0].loc[df_num2s[0][
        'GrLivArea'] != df_num2s[0]['GrLivArea'].max()]
for i in range(3):
    print(df_num2s[0].shape)
    df_num2s[0] = df_num2s[0].loc[df_num2s[0][
        'TotalBsmtSF'] != df_num2s[0]['TotalBsmtSF'].max()]
    print(df_num2s[0].shape)


# Drop outliers (2)

#df_num2[0] = df_num2[0].drop(["Id"],axis=1)
df_num2s[0] = df_num2s[0].set_index("Id")
df_num2s[0] = df_num2s[0].drop(["SalePrice"], axis=1)
print("df_num2s[0] : ", df_num2s[0].shape)

#---------Normalize-------------#
columns = df_num2s[0].columns
std_scale = preprocessing.StandardScaler()
std_scale.fit(df_num2s[0])
df_num2_sc = pd.DataFrame(
    std_scale.fit_transform(
        df_num2s[0]),
    index=df_num2s[0].index,
    columns=columns)
for column in df_num2_sc.columns:
    df_num2_sc = df_num2_sc.drop(
        df_num2_sc[df_num2_sc[column] > 25].index)  # pas < 25 car sinon NaN
print("df_num2_sc : ", df_num2_sc.shape)

# Table Transformation df_cleaned4 (2908, 89)
df_num2s[0] = df_num2s[0][df_num2s[0].index.isin(df_num2_sc.index.to_list())]

df_num2s[1] = df_num2s[1].set_index("Id")
df_num2s[1] = df_num2s[1].drop(["SalePrice"], axis=1)
df_num2 = pd.concat([df_num2s[0], df_num2s[1]], axis=0)
df_num3 = df_num2.copy()
print("df_num3:", df_num3.shape)
########################################################################

df_object = df_cleaned2.select_dtypes(include=['object'])

index_col = np.arange(1, 2920)
df_object["Id"] = index_col
df_object = df_object.set_index("Id")
df_object = df_object[df_object.index.isin(df_num3.index.to_list())]
df_cleaned3 = pd.concat([df_num3, df_object], axis=1)
df_cleaned4 = df_cleaned3.copy()
print("df_cleaned4:", df_cleaned4.shape)
########################################################################


# Add Features

df_cleaned4["SFPerRoom"] = df_cleaned4["GrLivArea"] / (df_cleaned4["TotRmsAbvGrd"] +
                                                       df_cleaned4["FullBath"] +
                                                       0.5 * df_cleaned4["HalfBath"] +
                                                       df_cleaned4["KitchenAbvGr"] +
                                                       df_cleaned4["BedroomAbvGr"])

df_cleaned4['TotOverall'] = df_cleaned4['OverallQual'] + \
    df_cleaned4['OverallCond']
df_cleaned4['TotGarage'] = df_cleaned4['GarageQual'] + \
    df_cleaned4['GarageCond']
df_cleaned4['TotExter'] = df_cleaned4['ExterQual'] + df_cleaned4['ExterCond']
df_cleaned4['TotBsmt'] = df_cleaned4['BsmtQual'] + df_cleaned4['BsmtCond']


df_cleaned4['TotBathrooms'] = (df_cleaned4['FullBath'] +
                               (0.5 *
                                df_cleaned4['HalfBath']) +
                               df_cleaned4['BsmtFullBath'] +
                               (0.5 *
                                df_cleaned4['BsmtHalfBath']))

df_cleaned4['BsmtFinType'] = df_cleaned4['BsmtFinType1'] + \
    df_cleaned4['BsmtFinType2']
df_cleaned4['BsmtFinSF'] = df_cleaned4['BsmtFinSF1'] + \
    df_cleaned4['BsmtFinSF2']
df_cleaned4['TotFlrSF'] = df_cleaned4['1stFlrSF'] + df_cleaned4['2ndFlrSF']
df_cleaned4['TotPorchSF'] = df_cleaned4['OpenPorchSF'] + \
    df_cleaned4['EnclosedPorch'] + df_cleaned4['3SsnPorch'] + df_cleaned4['ScreenPorch']
print("df_cleaned4:", df_cleaned4.shape)

# ADD y to DF
df_y = df_train[["Id", "SalePrice"]]

df_y = df_y.set_index("Id")

df_y = df_y[df_y.index.isin(df_cleaned4.index.to_list())]

df_cleaned4 = pd.concat([df_cleaned4, df_y], axis=1)
print("df_cleaned4:", df_cleaned4.shape)


# remove columns
cols_to_remove = ["GrLivArea",
                  "TotGarage",  # created variable
                  # "GarageCond",
                  "GarageCond",
                  "TotBsmt",  # created variable
                  "BsmtFinSF",  # created variable
                  "PoolArea",
                  "GarageArea",
                  "Fireplaces",
                  "TotExter",  # created variable
                  # "TotRmsAbvGrd",
                  "TotRmsAbvGrd",
                  "BsmtCond",
                  "GarageYrBlt",
                  "1stFlrSF",

                  # negative correlation to SalePrice
                  "BsmtFinSF2", "BsmtHalfBath", "LowQualFinSF", "YrSold",
                  "MiscVal", "OverallCond", "EnclosedPorch", "KitchenAbvGr"]


df_cleaned4 = df_cleaned4.drop(cols_to_remove, axis=1)
df_cleaned5 = df_cleaned4.copy()
########################################################################

# Log(X)

df_num5 = df_cleaned5.select_dtypes(include=['float64', 'int64'])
df_num5 = df_num5.drop(["SalePrice"], axis=1)

dictionary = {}
for column in df_num5.columns:
    dictionary[column] = abs(df_cleaned5[column].agg(['skew']))

    # Asymetric Distribution
df_skew = pd.DataFrame.from_dict(dictionary, orient='index', columns=['skew'])
df_skew = df_skew[df_skew['skew'] >= 0.5]
columns = df_skew.index.to_list()
df_cleaned5[columns] = df_cleaned5[columns].apply(lambda x: np.log1p(x))
df_cleaned6 = df_cleaned5.copy()
########################################################################


# cos(x)
df_cleaned6['MoSold'] = (-np.cos(0.5236 * df_cleaned6['MoSold']))
df_cleaned7 = df_cleaned6.copy()
########################################################################

# Encoding X
df_cleaned7 = df_cleaned7.drop(["SalePrice"], axis=1)
df_cleaned7 = pd.get_dummies(df_cleaned7, drop_first=True)
df_cleaned8 = df_cleaned7.copy()
########################################################################


# Normalize X
scaler = StandardScaler()
scaler.fit(df_cleaned8)
df_cleaned8 = pd.DataFrame(
    scaler.transform(df_cleaned8),
    index=df_cleaned8.index,
    columns=df_cleaned8.columns)
########################################################################

# X_pca
X = df_cleaned8.values
pca = PCA(n_components=0.8)
X_pca = pca.fit_transform(X)
X_f = np.concatenate((X, X_pca), axis=1)

# rename columns
df_X = pd.DataFrame(X_f)
dictionary = {}
new_name = df_cleaned8.columns.to_list()

for i in range(len(df_X.loc[:, 0:253].columns.to_list())):
    new_name = df_cleaned8.columns.to_list()
    column = df_X.loc[:, 0:253].columns.to_list()[i]
    dictionary[column] = new_name[i]
df_X.rename(columns=dictionary, inplace=True)
df_final = df_X.copy()
########################################################################


# log( y)
y = df_cleaned6['SalePrice']
log_y = np.log(y)
log_y = log_y.reset_index()  # first index line = 0 in place of 1
log_ys = np.split(log_y, [1449], axis=0)  # log_ys[0]["SalePrice"]

df_finals = np.split(df_final, [1449], axis=0)


# predict with the best hyper parameters
models = {
    "catboost": CatBoostRegressor(depth=4,
                                  iterations=5000,
                                  l2_leaf_reg=2,
                                  learning_rate=0.01),
    "gbr": GradientBoostingRegressor(learning_rate=0.01,
                                     max_depth=3,
                                     n_estimators=1000,
                                     subsample=0.5),
    "br": BayesianRidge(alpha_1=1e-06,
                        alpha_2=0.0001,
                        lambda_1=0.0001,
                        lambda_2=1e-06,
                        n_iter=200, tol=0),
    "lightgbm": LGBMRegressor(learning_rate=0.02,
                              max_depth=3,
                              n_estimators=570,
                              num_leaves=30),
    "ridge": Ridge(alpha=2, tol=0.0001)}


for name, model in models.items():
    model.fit(df_finals[0], log_ys[0]["SalePrice"])
    print(name + " trained.")


final_predictions = (
    0.4 * np.exp(models['catboost'].predict(df_finals[1])) +
    0.2 * np.exp(models['gbr'].predict(df_finals[1])) +
    0.2 * np.exp(models['br'].predict(df_finals[1])) +
    0.1 * np.exp(models['ridge'].predict(df_finals[1])) +
    0.1 * np.exp(models['lightgbm'].predict(df_finals[1]))
)
final_predictions


# to_csv
Id = df_test["Id"]
df_submited2 = pd.concat(
    [Id, pd.Series(final_predictions, name='SalePrice')], axis=1)
df_submited2.to_csv("test_submited.csv", index=False)
