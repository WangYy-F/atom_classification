import pandas as pd
import numpy as np
from numpy.random import normal, randint
import re
import os
from os import path
from sklearn.base import clone, BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor, ExtraTreesRegressor, AdaBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split, KFold
from scipy.special import softmax

class stacking(BaseEstimator, RegressorMixin, TransformerMixin):
    ##=============== 参数说明 ================##
    # mod --- 堆叠过程的第一层中的算法
    # meta_model --- 堆叠过程的第二层中的算法，也称次学习器

    def __init__(self, mod, meta_model):
        self.saved_model = None
        self.data = None
        self.mod = mod  # 首层学习器模型
        self.meta_model = meta_model  # 次学习器模型
        self.kf = KFold(n_splits=5, random_state=42, shuffle=True)  # 这就是堆叠的最大特征进行了几折的划分

    ## 训练函数
    def fit(self, X, y):
        self.data = np.mean(X[np.where(y==1), :], axis=0)
        self.saved_model = [list() for i in self.mod]  # self.saved_model包含所有第一层学习器
        oof_train = np.zeros((X.shape[0], len(self.mod)))  # 维度：训练样本数量*模型数量，训练集的首层预测值
        for i, model in enumerate(self.mod):  # 返回的是索引和模型本身
            for train_index, val_index in self.kf.split(X, y):  # 返回的是数据分割成分（训练集和验证集对应元素）的索引
                renew_model = clone(model)  # 模型的复制
                renew_model.fit(X[train_index], y[train_index])  # 对分割出来的训练集数据进行训练
                self.saved_model[i].append(renew_model)  # 把模型添加进去
                # oof_train[val_index,i] = renew_model.predict(X[val_index]).reshape(-1,1) #用来预测验证集数据
                val_prediction = renew_model.predict(X[val_index]).reshape(-1, 1)  # 验证集的预测结果，注：结果是没有索引的
                for temp_index in range(val_prediction.shape[0]):
                    oof_train[val_index[temp_index], i] = val_prediction[temp_index]  # 用来预测验证集数据的目标值
        self.meta_model.fit(oof_train, y)  # 次学习器模型训练，这里只是用到了首层预测值作为特征
        return self

    ## 预测函数
    def predict(self, X):
        whole_test = np.column_stack([np.column_stack(model.predict(X) for model in single_model).mean(axis=1)
                                      for single_model in self.saved_model])  # 得到的是整个测试集的首层预测值
        return self.meta_model.predict(whole_test)  # 返回次学习器模型对整个测试集的首层预测值特征的最终预测结果

    ## 获取首层学习结果的堆叠特征
    def get_oof(self, X, y, test_X):
        oof = np.zeros((X.shape[0], len(self.mod)))  # 初始化为0
        test_single = np.zeros((test_X.shape[0], 5))  # 初始化为0
        # display(test_single.shape)
        test_mean = np.zeros((test_X.shape[0], len(self.mod)))
        for i, model in enumerate(self.mod):  # i是模型
            for j, (train_index, val_index) in enumerate(self.kf.split(X, y)):  # j是所有划分好的的数据
                clone_model = clone(model)  # 克隆模块，相当于把模型复制一下
                clone_model.fit(X[train_index], y[train_index])  # 把分割好的数据进行训练
                val_prediction = clone_model.predict(X[val_index]).reshape(-1, 1)  # 验证集的预测结果，注：结果是没有索引的
                for temp_index in range(val_prediction.shape[0]):
                    oof[val_index[temp_index], i] = val_prediction[temp_index]  # 用来预测验证集数据
                test_prediction = clone_model.predict(test_X).reshape(-1, 1)  # 对测试集进行预测
                test_single[:, j] = test_prediction[:, 0]
            test_mean[:, i] = test_single.mean(axis=1)  # 测试集算好均值
        return oof, test_mean

    def sampling(self, n):
        start = np.mean(self.data, axis = 0)
        output = np.array([start])
        i = 0
        u = 0
        q = 0
        print("sampling goes...")
        if sum(start[5:9]) <= 1e-5:
            dim = 5
        else:
            dim = 10
        while i < n / 2 and q < n:
            for j in range(dim):
                q = q + 1
                value = start
                value[j] = normal(start[j], 0.005, 1)
                if self.predict([value]):
                    start = value
                    output = np.append(output, [start], axis = 0)
                    i = i + 1
                    if i % 500 == 0:
                        print(start)
                else:
                    u = u + 1
                    if u >= 200:
                        start = np.mean(self.data, axis=0)
                        u = 0
        if i < n/2:
            print("copy supp")
            for q in range(i, int(n/2)):
                output = np.append(output, [np.mean(self.data, axis = 0)], axis = 0)
        return output


class models:

    def __init__(self, n):
        self.models = []
        for i in range(n):
            #mod = [BaggingRegressor(), ExtraTreesRegressor(), AdaBoostRegressor(), KNeighborsRegressor()]
            #meta_model = LogisticRegression()
            self.models.append(self.create())

    def create(self):
        return stacking([BaggingRegressor(), ExtraTreesRegressor(), AdaBoostRegressor(), KNeighborsRegressor()],
                        LGBMRegressor())

    def fit(self, X, y):
        assert len(self.models) == y.shape[1]
        for i, model in enumerate(self.models):
            print("fit model", i)
            model.fit(X,y[:,i])

    def test(self, X):
        result = []
        y = np.zeros((X.shape[0], len(self.models)))
        for i, model in enumerate(self.models):
            y[:,i] = model.predict(X).ravel()
        for i in range(X.shape[0]):
            #if sum(y[i,:]) == 0:
            #    result.append(-1)
            #elif sum(y[i,:]) == 1:
            result.append(np.argmax(y[i,:]))
            #else:
            #    y[i, 6] = 0
            #    if sum(y[i, :]) == 1:
            #        result.append(np.argmax(y[i, :]))
            #    else:
            #        result.append(-2)
        return np.array(result), y

    def accuracy(self, X, y):
        pred, re = self.test(X)
        return sum(pred != np.argmax(y, axis=1)), X.shape[0], re

    def append(self, X):
        y = np.zeros((X.shape[0])) + 1
        for model in self.models:
            X = np.append(X, model.data, axis = 0)
            y = np.append(y, np.zeros((model.data.shape[0])))
        mod = stacking(self.models[0].mod, self.models[0].meta_model)
        mod.fit(X, y)
        self.models.append(mod)

    def extend(self, x_dir, y_dir):
        x = np.loadtxt(x_dir)
        y = np.loadtxt(y_dir)
        n  = x.shape[0]
        for mod in self.models:
            new_dat = mod.sampling(n)
            x = np.append(x, new_dat, axis = 0)
            y = np.append(y, np.zeros(new_dat.shape[0]), axis = 0)
        model = self.create()
        model.fit(x, y)
        self.models.append(model)