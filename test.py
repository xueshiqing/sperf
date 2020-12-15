from sklearn.linear_model import LinearRegression
from sklearn import tree
from sklearn import svm
from sklearn import neighbors
from sklearn import ensemble
from sklearn import ensemble
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error  # 均方误差
from sklearn.metrics import mean_absolute_error  # 平方绝对误差
from sklearn.metrics import median_absolute_error  #中位数绝对误差
from sklearn.metrics import r2_score  # R square
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn2pmml import PMMLPipeline, sklearn2pmml


import matplotlib.pyplot as plt

from sklearn import tree

model_DecisionTreeRegressor = tree.DecisionTreeRegressor()
####3.2线性回归####
from sklearn import linear_model

model_LinearRegression = linear_model.LinearRegression()
####3.3SVM回归####
from sklearn import svm

model_SVR = svm.SVR()
####3.4KNN回归####
from sklearn import neighbors

model_KNeighborsRegressor = neighbors.KNeighborsRegressor()
####3.5随机森林回归####
from sklearn import ensemble

model_RandomForestRegressor = ensemble.RandomForestRegressor(n_estimators=50)  # 这里使用20个决策树
####3.6Adaboost回归####
from sklearn import ensemble

model_AdaBoostRegressor = ensemble.AdaBoostRegressor(n_estimators=100)  # 这里使用50个决策树
####3.7GBRT回归####
from sklearn import ensemble

model_GradientBoostingRegressor = ensemble.GradientBoostingRegressor(n_estimators=130)  # 这里使用100个决策树
####3.8Bagging回归####
from sklearn.ensemble import BaggingRegressor

model_BaggingRegressor = BaggingRegressor()
####3.9ExtraTree极端随机树回归####
from sklearn.tree import ExtraTreeRegressor

model_ExtraTreeRegressor = ExtraTreeRegressor()
####3.10ARD贝叶斯ARD回归
model_ARDRegression = linear_model.ARDRegression()
####3.11BayesianRidge贝叶斯岭回归
model_BayesianRidge = linear_model.BayesianRidge()
####3.12TheilSen泰尔森估算
model_TheilSenRegressor = linear_model.TheilSenRegressor()
####3.13RANSAC随机抽样一致性算法
model_RANSACRegressor = linear_model.RANSACRegressor()


def load_data():
    df = pd.read_csv('out.csv', encoding='utf-8')
    print(df)
    return df


def try_different_method(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    sklearn2pmml(model, "model.pmml")
    score = model.score(x_test, y_test)
    result = model.predict(x_test)
    joblib.dump(model, "model.m")
    print('========MODEL=========')
    print(type(model))
    print('========RMSE==========')
    print(np.sqrt(mean_squared_error(y_test, result)))
    print('=========MAE==========')
    print(mean_absolute_error(y_test, result))
    print('=========R^2==========')
    print(r2_score(y_test, result))
    # plt.figure(figsize=(15, 5))
    # plt.plot(np.arange(len(result)), y_test, 'go-', label='true value')
    # plt.plot(np.arange(len(result)), result, 'ro-', label='predict value')
    # plt.title('score: %f' % score)
    # plt.legend()
    # plt.show()
    model_load = joblib.load("model.m")
    # res1 = model.predict(np.array([[1, 2, 3, 4, 5]]))
    # print(res1)


df = load_data()
# print(df)
df_x = df.drop(['timeStamp', 'failureMessage'], axis=1)
df_x.dropna(inplace=True)
label_mapping = {
    'homepage': 1,
    'loginpage': 2,
    'login': 3,
    'inbox': 4,
    'inboxlist': 5,
    'mydemand': 6,
    'releasedemand': 7,
    'mydemandlist': 8,
    'logout-0': 9,
    'logout-1': 10,
    "login-0": 11,
    "login-1": 12,
    "login-2": 13,
    "logout-2": 14,
    "logout": 15,
    "sw-dataCheck": 20,
    "sw-dataCollection": 21,
    "sw-dataManage": 22,
    "sw-enterpriseNature": 23,
    "sw-getMainDataStatic": 24,
    "sw-indexInfo": 25,
    "sw-industrialStructure": 26,
    "sw-industryClassification": 27,
    "sw-login": 28,
    "sw-logout": 29,
    "sw-navigationTree": 30,
    "sw-systemAdmin": 31,
    "sw-systemPersonal": 32
}

success_mapping = {
    True: 1,
    False: 2
}

df_x['label'] = df_x['label'].map(label_mapping)
df_x['success'] = df_x['success'].map(success_mapping)
# print(df_x[['label', 'allThreads', 'success', 'grpThreads']])
df_x.dropna(inplace=True)
X_train, X_test, Y_train, Y_test = train_test_split(
    df_x[['label', 'allThreads', 'success', 'mem_free', 'service', 'batch']],
    df_x['elapsed'], test_size=0.2, random_state=0)
# print(X_test)
# print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)


pipeline = PMMLPipeline([
    ("regressor", model_RandomForestRegressor)
])


try_different_method(pipeline, X_train, X_test, Y_train, Y_test)

# lgb_train = lgb.Dataset(X_train,Y_train)
# lgb_valid = lgb.Dataset(X_test,Y_test,reference=lgb_train)
#
# param = {'max_depth':10, 'objective':'binary','num_threads':8,
#          'learning_rate':0.1,'bagging':0.7,'feature_fraction':0.7,
#          'lambda_l1':0.1,'lambda_l2':0.2,'seed':123454}
# gbm = lgb.train(param, lgb_train, num_boost_round=150,early_stopping_rounds=100, valid_sets=[lgb_valid])
# y_train_binary = gbm.predict(X_train, num_iteration=gbm.best_iteration)  # type:np.numarray
# result = gbm.predict(X_test, num_iteration=gbm.best_iteration)  # type:np.numarray
# print('========MODEL=========')
# print("LGB")
# print('========RMSE==========')
# print(np.sqrt(mean_squared_error(Y_test, result)))
# print('=========MAE==========')
# print(mean_absolute_error(Y_test, result))
# print('=========R^2==========')
# print(r2_score(Y_test, result))