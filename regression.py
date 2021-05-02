# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 15:56:37 2021

@author: AMD_PC
"""
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# python方程式求解
dataset = finaldata
X = dataset[:, 1:]
y = dataset[:, 0]
best = dict({'R2': 0, '常數項': 0, '係數': 0,'預測':0,'測試':0})
for k in range(0, 3000):
    print(k)
    score = 0
    while (score > 0.99 or score < 0.01):
        # Importing the dataset
        # Splitting the dataset into the Training set and Test set
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3)
        # random_state = 0

        # Fitting Multiple Linear Regression to the Training set
        regressor = LinearRegression()
        regressor.fit(X_train, y_train)

        # Predicting the Test set results
        y_pred = regressor.predict(X_test)
        # y= regressor.predict(X)

        score = r2_score(y_test, y_pred)
        alpha = regressor.intercept_
        beta = regressor.coef_

        point_list = []
        point = dict({'R2': score, '常數項': alpha,
                      '係數': beta,'預測':y_pred,'測試':y_test})
        if best['R2'] < point['R2']:
            best = point.copy()
# 運算各個參數
print('迴圈執行了 %d次' % k)
print('參數')
print('決定係數', best['R2'])
for i in range(0, number):
    print(polynomial[i],"係數為", best['係數'][i])
print("常數項為", best['常數項'])

