# -*- coding: utf-8 -*-
"""
Created on Tue May  4 14:55:34 2021

@author: AMD_PC
"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from value_testfunction import value_normal, value_ln_normal, value_plus, value_minus, value_ln_plus, value_ln_minus, value_divide, value_ln_divide
from polynomialfunction import substance
from polynomialfunction import normal, ln_normal, plus, minus, ln_plus, ln_minus, divide, ln_divide
import numpy as np
import pandas as pd
from scipy.stats import stats
from timeout import set_time_limit
#number = int(input("特徵數量(整數):"))
best = dict({'R2': 0, '常數項': 0, '係數': 0, '預測': 0, '測試': 0, '特徵組合': 0})

for j in range(1, 11):  # 依序前1到前10特徵參數的迴圈
    number = j
    print(number)
    df = pd.read_excel('data.xlsx', index_col=False)
    b = df['blue']
    g = df['green']
    r = df['red']
    edge = df['rededge']
    n = df['nir']

    data = pd.Series({  # 進行皮爾森指數比較的資料庫
        'highest': 1,
        'b': normal(b),
        'g': normal(g),
        'r': normal(r),
        're': normal(edge),
        'n': normal(n),
        'ln(b)': ln_normal(b),
        'ln(b)': ln_normal(g),
        'ln(r)': ln_normal(r),
        'ln(re)': ln_normal(edge),
        'ln(n)': ln_normal(n),
        '(b+g)': plus(b, g),
        '(b+r)': plus(b, r),
        '(b+re)': plus(b, edge),
        '(b+n)': plus(b, n),
        '(g+r)': plus(g, r),
        '(g+re)': plus(g, edge),
        '(g+n)': plus(g, n),
        '(r+re)': plus(r, edge),
        '(r+n)': plus(r, n),
        '(re+n)': plus(edge, n),
        '(b-g)': minus(b, g),
        '(b-r)': minus(b, r),
        '(b-re)': minus(b, edge),
        '(b-n)': minus(b, n),
        '(g-r)': minus(g, r),
        '(g-re)': minus(g, edge),
        '(g-n)': minus(g, n),
        '(r-re)': minus(r, edge),
        '(r-n)': minus(r, n),
        '(re-n)': minus(edge, n),
        'ln(b+g)': ln_plus(b, g),
        'ln(b+r)': ln_plus(b, r),
        'ln(b+re)': ln_plus(b, edge),
        'ln(b+n)': ln_plus(b, n),
        'ln(g+r)': ln_plus(g, r),
        'ln(g+re)': ln_plus(g, edge),
        'ln(g+n)': ln_plus(g, n),
        'ln(r+re)': ln_plus(r, edge),
        'ln(r+n)': ln_plus(r, n),
        'ln(re+n)': ln_plus(edge, n),
        'ln(b-g+2)': ln_minus(b, g),
        'ln(b-r+2)': ln_minus(b, r),
        'ln(b-re+2)': ln_minus(b, edge),
        'ln(b-n+2)': ln_minus(b, n),
        'ln(g-r+2)': ln_minus(g, r),
        'ln(g-re+2)': ln_minus(g, edge),
        'ln(g-n+2)': ln_minus(g, n),
        'ln(r-re+2)': ln_minus(r, edge),
        'ln(r-n+2)': ln_minus(r, n),
        'ln(re-n+2)': ln_minus(edge, n),
        '(b/g)': divide(b, g),
        '(b/r)': divide(b, r),
        '(b/re)': divide(b, edge),
        '(b/n)': divide(b, n),
        '(g/b)': divide(g, b),
        '(g/r)': divide(g, r),
        '(g/re)': divide(g, edge),
        '(g/n)': divide(g, n),
        '(r/b)': divide(r, b),
        '(r/g)': divide(r, g),
        '(r/re)': divide(r, edge),
        '(r/n)': divide(r, n),
        '(re/b)': divide(edge, b),
        '(re/g)': divide(edge, g),
        '(re/r)': divide(edge, r),
        '(re/n)': divide(edge, n),
        '(n/b)': divide(n, b),
        '(n/g)': divide(n, g),
        '(n/r)': divide(n, r),
        '(n/re)': divide(n, edge),
        'ln(b/g)': ln_divide(b, g),
        'ln(b/r)': ln_divide(b, r),
        'ln(b/re)': ln_divide(b, edge),
        'ln(b/n)': ln_divide(b, n),
        'ln(g/r)': ln_divide(g, r),
        'ln(g/re)': ln_divide(g, edge),
        'ln(g/n)': ln_divide(g, n),
        'ln(r/re)': ln_divide(r, edge),
        'ln(r/n)': ln_divide(r, n),
        'ln(re/n)': ln_divide(edge, n)})

    # 將重複的數值清除只留一個(keep='last')保留重複最後的名稱 (keep=False)將所有重複的數值刪去
    drop_series = data.drop_duplicates()  # 將重複值刪除只保留一個
    biggest = drop_series.nlargest(number+1)
    #print("前幾高pearson", biggest)
    polynomial = biggest.index[1:]  # 將前幾高的特徵組合名稱儲存起來
    print(data[polynomial])
    print(polynomial)

    data = pd.Series({  # 這邊是全部10個點的數據庫
        'highest': 1,
        'b': value_normal(b),
        'g': value_normal(g),
        'r': value_normal(r),
        're': value_normal(edge),
        'n': value_normal(n),
        'ln(b)': value_ln_normal(b),
        'ln(b)': value_ln_normal(g),
        'ln(r)': value_ln_normal(r),
        'ln(re)': value_ln_normal(edge),
        'ln(n)': value_ln_normal(n),
        '(b+g)': value_plus(b, g),
        '(b+r)': value_plus(b, r),
        '(b+re)': value_plus(b, edge),
        '(b+n)': value_plus(b, n),
        '(g+r)': value_plus(g, r),
        '(g+re)': value_plus(g, edge),
        '(g+n)': value_plus(g, n),
        '(r+re)': value_plus(r, edge),
        '(r+n)': value_plus(r, n),
        '(re+n)': value_plus(edge, n),
        '(b-g)': value_minus(b, g),
        '(b-r)': value_minus(b, r),
        '(b-re)': value_minus(b, edge),
        '(b-n)': value_minus(b, n),
        '(g-r)': value_minus(g, r),
        '(g-re)': value_minus(g, edge),
        '(g-n)': value_minus(g, n),
        '(r-re)': value_minus(r, edge),
        '(r-n)': value_minus(r, n),
        '(re-n)': value_minus(edge, n),
        'ln(b+g)': value_ln_plus(b, g),
        'ln(b+r)': value_ln_plus(b, r),
        'ln(b+re)': value_ln_plus(b, edge),
        'ln(b+n)': value_ln_plus(b, n),
        'ln(g+r)': value_ln_plus(g, r),
        'ln(g+re)': value_ln_plus(g, edge),
        'ln(g+n)': value_ln_plus(g, n),
        'ln(r+re)': value_ln_plus(r, edge),
        'ln(r+n)': value_ln_plus(r, n),
        'ln(re+n)': value_ln_plus(edge, n),
        'ln(b-g+2)': value_ln_minus(b, g),
        'ln(b-r+2)': value_ln_minus(b, r),
        'ln(b-re+2)': value_ln_minus(b, edge),
        'ln(b-n+2)': value_ln_minus(b, n),
        'ln(g-r+2)': value_ln_minus(g, r),
        'ln(g-re+2)': value_ln_minus(g, edge),
        'ln(g-n+2)': value_ln_minus(g, n),
        'ln(r-re+2)': value_ln_minus(r, edge),
        'ln(r-n+2)': value_ln_minus(r, n),
        'ln(re-n+2)': value_ln_minus(edge, n),
        '(b/g)': value_divide(b, g),
        '(b/r)': value_divide(b, r),
        '(b/re)': value_divide(b, edge),
        '(b/n)': value_divide(b, n),
        '(g/b)': value_divide(g, b),
        '(g/r)': value_divide(g, r),
        '(g/re)': value_divide(g, edge),
        '(g/n)': value_divide(g, n),
        '(r/b)': value_divide(r, b),
        '(r/g)': value_divide(r, g),
        '(r/re)': value_divide(r, edge),
        '(r/n)': value_divide(r, n),
        '(re/b)': value_divide(edge, b),
        '(re/g)': value_divide(edge, g),
        '(re/r)': value_divide(edge, r),
        '(re/n)': value_divide(edge, n),
        '(n/b)': value_divide(n, b),
        '(n/g)': value_divide(n, g),
        '(n/r)': value_divide(n, r),
        '(n/re)': value_divide(n, edge),
        'ln(b/g)': value_ln_divide(b, g),
        'ln(b/r)': value_ln_divide(b, r),
        'ln(b/re)': value_ln_divide(b, edge),
        'ln(b/n)': value_ln_divide(b, n),
        'ln(g/r)': value_ln_divide(g, r),
        'ln(g/re)': value_ln_divide(g, edge),
        'ln(g/n)': value_ln_divide(g, n),
        'ln(r/re)': value_ln_divide(r, edge),
        'ln(r/n)': value_ln_divide(r, n),
        'ln(re/n)': value_ln_divide(edge, n)})
    #qwe = data[polynomial[2]]
    value_list = [substance, ]  # 將檢測物跟所選擇的特徵組合 變成一個矩陣 為了後面迴歸公式的演算

    for i in range(0, number, 1):
        value = data[polynomial[i]]
        value_list += [value]

    finaldata = np.transpose(value_list)

    # 回歸方程式求解
    dataset = finaldata
    X = dataset[:, 1:]
    y = dataset[:, 0]
    for k in range(0, 3000):  # 每個計算都算3000次 算是經驗次數
        print(k)
        score = 0
        while (score > 1 or score < 0.00001):  # 設定回歸公式的決定係數 分數介於1~0.00001 之間算一次運算
            # Importing the dataset
            # Splitting the dataset into the Training set and Test set
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3)  # 取30%當測試 70%當作訓練
            # random_state = 0

            # Fitting Multiple Linear Regression to the Training set
            regressor = LinearRegression()
            regressor.fit(X_train, y_train)

            # Predicting the Test set results
            y_pred = regressor.predict(X_test)  # 預測
            # y= regressor.predict(X)

            score = r2_score(y_test, y_pred)  # 分數
            alpha = regressor.intercept_  # 常數項
            beta = regressor.coef_  # 各特徵參數的係數

            point_list = []
            point = dict({'R2': score, '常數項': alpha,  # 給定名稱為了儲存資料  best的定義在最上面
                          '係數': beta, '預測': y_pred, '測試': y_test, '特徵組合': polynomial})
            if best['R2'] < point['R2']:
                best = point.copy()

print('決定係數', best['R2'])

# 運算各個參數
#  print('迴圈執行了 %d次' % k)

# 將所得到的迴歸公式  輸出
regression_function = ''
for i in range(0, len(best['特徵組合'])):
    regression_function += best['特徵組合'][i]+"*"+"("+str(best['係數'][i])+")"+"+"
regression_function = regression_function+"("+str(best['常數項'])+")"
