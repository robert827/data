# -*- coding: utf-8 -*-
"""
Created on Fri May  7 16:43:55 2021

@author: AMD_PC
"""
import pandas as pd
from polynomialfunction import normal, ln_normal, plus, minus, ln_plus, ln_minus, divide, ln_divide
from polynomialfunction import substance, name
from value_testfunction import value_normal, value_ln_normal, value_plus, value_minus, value_ln_plus, value_ln_minus, value_divide, value_ln_divide
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import glob
import os
import imageio
import micasense.imageutils as imageutils
import matplotlib.pyplot as plt
import numpy as np
import cv2


def data_point(image, x, y):
    # 取樣點座標
    # x = 800
    # y = 1200

    m = []

    # 取樣像素格數，只能是基數
    for pixelnumber in range(5, 12, 6):

        blurry = 41  # 基數，濾波範圍

        pixelleft = (pixelnumber-1)/2
        pixelright = (pixelnumber+1)/2
        xright = x+int(pixelright)
        xleft = x-int(pixelleft)
        yright = y+int(pixelright)
        yleft = y-int(pixelleft)

        # 熱像座標
        # xthermal = int(x*(160/2029))
        # ythermal = int(y*(120/1480))

        import numpy as np
        # 讀取校正後反射率
        blue = (np.load(image+'_blue.npy'))
        green = (np.load(image+'_green.npy'))
        red = (np.load(image+'_red.npy'))
        rededge = (np.load(image+'_rededge.npy'))
        nir = (np.load(image+'_nir.npy'))
        # thermal = np.load('thermal.npy')

        # 濾波取平均
        def values(band):
            band = cv2.GaussianBlur(band, (blurry, blurry), 0)
            band = band[yleft:yright, xleft:xright]
            band = band.mean()
            return band

        blue = values(blue)
        green = values(green)
        red = values(red)
        rededge = values(rededge)
        nir = values(nir)
        # thermal = values(thermal)

        np1 = np.array([blue, green, red, rededge, nir])
        # np2 = np1.reshape(5,1)

        m.append((pixelnumber, np1))
        return "b,g,r,re,n", np1


def correction(image_path, image_name, panel_path, panel_name):
    import micasense.capture as capture
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
    # 輸入照片位置
    get_ipython().run_line_magic('matplotlib', 'inline')

    panelNames = None

    # This is an altum image with RigRelatives and a thermal band
    imageNames = glob.glob(os.path.join(image_path, image_name+'_*.tif'))
    panelNames = glob.glob(os.path.join(panel_path, panel_name+'_*.tif'))

    if panelNames is not None:
        panelCap = capture.Capture.from_filelist(panelNames)
    else:
        panelCap = None

    capture = capture.Capture.from_filelist(imageNames)

    for img in capture.images:
        if img.rig_relatives is None:
            raise ValueError(
                "Images must have RigRelatives tags set which requires updated firmware and calibration. See the links in text above")

    if panelCap is not None:
        if panelCap.panel_albedo() is not None:
            panel_reflectance_by_band = panelCap.panel_albedo()
        else:
            # RedEdge band_index order
            panel_reflectance_by_band = [0.67, 0.69, 0.68, 0.61, 0.67]
        panel_irradiance = panelCap.panel_irradiance(panel_reflectance_by_band)
        img_type = "reflectance"
        capture.plot_undistorted_reflectance(panel_irradiance)
    else:
        if False:  # capture.dls_present():
            img_type = 'reflectance'
            capture.plot_undistorted_reflectance(capture.dls_irradiance())
        else:
            img_type = "radiance"
            capture.plot_undistorted_radiance()

    # 相片對齊
    warp_mode = cv2.MOTION_HOMOGRAPHY
    warp_matrices = capture.get_warp_matrices()

    cropped_dimensions, edges = imageutils.find_crop_bounds(
        capture, warp_matrices)
    im_aligned = imageutils.aligned_capture(
        capture, warp_matrices, warp_mode, cropped_dimensions, None, img_type=img_type)

    print("warp_matrices={}".format(warp_matrices))

    rgb_band_indices = [2, 1, 0]

    # Create an empty normalized stack for viewing
    im_display = np.zeros(
        (im_aligned.shape[0], im_aligned.shape[1], capture.num_bands+1), dtype=np.float32)

    # modify with these percentilse to adjust contrast
    im_min = np.percentile(im_aligned[:, :, 0:2].flatten(),  0.1)
    # for many images, 0.5 and 99.5 are good values
    im_max = np.percentile(im_aligned[:, :, 0:2].flatten(), 99.9)

    for i in range(0, im_aligned.shape[2]):
        if img_type == 'reflectance':
            # for reflectance images we maintain white-balance by applying the same display scaling to all bands
            im_display[:, :, i] = imageutils.normalize(
                im_aligned[:, :, i], im_min, im_max)
        elif img_type == 'radiance':
            # for radiance images we do an auto white balance since we don't know the input light spectrum by
            # stretching each display band histogram to it's own min and max
            im_display[:, :, i] = imageutils.normalize(im_aligned[:, :, i])

    rgb = im_display[:, :, rgb_band_indices]

    # 合成出rgb圖
    # Create an enhanced version of the RGB render using an unsharp mask
    gaussian_rgb = cv2.GaussianBlur(rgb, (9, 9), 10.0)
    gaussian_rgb[gaussian_rgb < 0] = 0
    gaussian_rgb[gaussian_rgb > 1] = 1
    unsharp_rgb = cv2.addWeighted(rgb, 1.5, gaussian_rgb, -0.5, 0)
    unsharp_rgb[unsharp_rgb < 0] = 0
    unsharp_rgb[unsharp_rgb > 1] = 1

    # Apply a gamma correction to make the render appear closer to what our eyes would see
    gamma = 1.2
    gamma_corr_rgb = unsharp_rgb**(1.0/gamma)
    plt.imshow(gamma_corr_rgb, aspect='equal')
    plt.axis('off')
    plt.show()
    # 匯出rgb圖
    imtype = 'png'  # or 'jpg'
    imageio.imwrite(image_name+'_rgb.'+imtype,
                    (255*gamma_corr_rgb).astype('uint8'))

    blue_band = capture.band_names_lower().index('blue')
    green_band = capture.band_names_lower().index('green')
    red_band = capture.band_names_lower().index('red')
    nir_band = capture.band_names_lower().index('nir')
    rededge_band = capture.band_names_lower().index('red edge')

    blue = im_aligned[:, :, blue_band]
    blue = blue.astype('float64')
    np.save(image_name+'_blue', blue)

    green = im_aligned[:, :, green_band]
    green = green.astype('float64')
    np.save(image_name+'_green', green)

    red = im_aligned[:, :, red_band]
    red = red.astype('float64')
    np.save(image_name+'_red', red)

    rededge = im_aligned[:, :, rededge_band]
    red = rededge.astype('float64')
    np.save(image_name+'_rededge', rededge)

    nir = im_aligned[:, :, nir_band]
    nir = nir.astype('float64')
    np.save(image_name+'_nir', nir)
    return print("all bands npy file save")
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
    # 輸入照片位置
    get_ipython().run_line_magic('matplotlib', 'inline')

    panelNames = None

    # This is an altum image with RigRelatives and a thermal band
    imagePath = os.path.join('.', 'data')
    imageNames = glob.glob(os.path.join(imagePath,  'IMG_0134_*.tif'))
    panelNames = glob.glob(os.path.join(imagePath,  'IMG_0003_*.tif'))

    if panelNames is not None:
        panelCap = capture.Capture.from_filelist(panelNames)
    else:
        panelCap = None

    capture = capture.Capture.from_filelist(imageNames)

    for img in capture.images:
        if img.rig_relatives is None:
            raise ValueError(
                "Images must have RigRelatives tags set which requires updated firmware and calibration. See the links in text above")

    if panelCap is not None:
        if panelCap.panel_albedo() is not None:
            panel_reflectance_by_band = panelCap.panel_albedo()
        else:
            # RedEdge band_index order
            panel_reflectance_by_band = [0.67, 0.69, 0.68, 0.61, 0.67]
        panel_irradiance = panelCap.panel_irradiance(panel_reflectance_by_band)
        img_type = "reflectance"
        capture.plot_undistorted_reflectance(panel_irradiance)
    else:
        if False:  # capture.dls_present():
            img_type = 'reflectance'
            capture.plot_undistorted_reflectance(capture.dls_irradiance())
        else:
            img_type = "radiance"
            capture.plot_undistorted_radiance()

    # 相片對齊
    warp_mode = cv2.MOTION_HOMOGRAPHY
    warp_matrices = capture.get_warp_matrices()

    cropped_dimensions, edges = imageutils.find_crop_bounds(
        capture, warp_matrices)
    im_aligned = imageutils.aligned_capture(
        capture, warp_matrices, warp_mode, cropped_dimensions, None, img_type=img_type)

    print("warp_matrices={}".format(warp_matrices))

    rgb_band_indices = [2, 1, 0]

    # Create an empty normalized stack for viewing
    im_display = np.zeros(
        (im_aligned.shape[0], im_aligned.shape[1], capture.num_bands+1), dtype=np.float32)

    # modify with these percentilse to adjust contrast
    im_min = np.percentile(im_aligned[:, :, 0:2].flatten(),  0.1)
    # for many images, 0.5 and 99.5 are good values
    im_max = np.percentile(im_aligned[:, :, 0:2].flatten(), 99.9)

    for i in range(0, im_aligned.shape[2]):
        if img_type == 'reflectance':
            # for reflectance images we maintain white-balance by applying the same display scaling to all bands
            im_display[:, :, i] = imageutils.normalize(
                im_aligned[:, :, i], im_min, im_max)
        elif img_type == 'radiance':
            # for radiance images we do an auto white balance since we don't know the input light spectrum by
            # stretching each display band histogram to it's own min and max
            im_display[:, :, i] = imageutils.normalize(im_aligned[:, :, i])

    rgb = im_display[:, :, rgb_band_indices]

    # 合成出rgb圖
    # Create an enhanced version of the RGB render using an unsharp mask
    gaussian_rgb = cv2.GaussianBlur(rgb, (9, 9), 10.0)
    gaussian_rgb[gaussian_rgb < 0] = 0
    gaussian_rgb[gaussian_rgb > 1] = 1
    unsharp_rgb = cv2.addWeighted(rgb, 1.5, gaussian_rgb, -0.5, 0)
    unsharp_rgb[unsharp_rgb < 0] = 0
    unsharp_rgb[unsharp_rgb > 1] = 1

    # Apply a gamma correction to make the render appear closer to what our eyes would see
    gamma = 1.2
    gamma_corr_rgb = unsharp_rgb**(1.0/gamma)
    plt.imshow(gamma_corr_rgb, aspect='equal')
    plt.axis('off')
    plt.show()
    # 匯出rgb圖
    imtype = 'png'  # or 'jpg'
    imageio.imwrite('rgb.'+imtype, (255*gamma_corr_rgb).astype('uint8'))

    blue_band = capture.band_names_lower().index('blue')
    green_band = capture.band_names_lower().index('green')
    red_band = capture.band_names_lower().index('red')
    nir_band = capture.band_names_lower().index('nir')
    rededge_band = capture.band_names_lower().index('red edge')

    blue = im_aligned[:, :, blue_band]
    blue = blue.astype('float64')
    np.save('blue', blue)

    green = im_aligned[:, :, green_band]
    green = green.astype('float64')
    np.save('green', green)

    red = im_aligned[:, :, red_band]
    red = red.astype('float64')
    np.save('red', red)

    rededge = im_aligned[:, :, rededge_band]
    red = rededge.astype('float64')
    np.save('rededge', rededge)

    nir = im_aligned[:, :, nir_band]
    nir = nir.astype('float64')
    np.save('nir', nir)


times = 11


def regression(b,g,r,re,n):
    best = dict({'R2': 0, '常數項': 0, '係數': 0, '預測': 0, '測試': 0, '特徵組合': 0})
    for j in range(1, times):  # 依序前1到前10特徵參數的迴圈
        number = j
        print(number)
        data = pd.Series({  # 進行皮爾森指數比較的資料庫
            'highest': 1,
            'b': normal(b),
            'g': normal(g),
            'r': normal(r),
            're': normal(re),
            'n': normal(n),
            'ln(b)': ln_normal(b),
            'ln(g)': ln_normal(g),
            'ln(r)': ln_normal(r),
            'ln(re)': ln_normal(re),
            'ln(n)': ln_normal(n),
            '(b+g)': plus(b, g),
            '(b+r)': plus(b, r),
            '(b+re)': plus(b, re),
            '(b+n)': plus(b, n),
            '(g+r)': plus(g, r),
            '(g+re)': plus(g, re),
            '(g+n)': plus(g, n),
            '(r+re)': plus(r, re),
            '(r+n)': plus(r, n),
            '(re+n)': plus(re, n),
            '(b-g)': minus(b, g),
            '(b-r)': minus(b, r),
            '(b-re)': minus(b, re),
            '(b-n)': minus(b, n),
            '(g-r)': minus(g, r),
            '(g-re)': minus(g, re),
            '(g-n)': minus(g, n),
            '(r-re)': minus(r, re),
            '(r-n)': minus(r, n),
            '(re-n)': minus(re, n),
            'ln(b+g)': ln_plus(b, g),
            'ln(b+r)': ln_plus(b, r),
            'ln(b+re)': ln_plus(b, re),
            'ln(b+n)': ln_plus(b, n),
            'ln(g+r)': ln_plus(g, r),
            'ln(g+re)': ln_plus(g, re),
            'ln(g+n)': ln_plus(g, n),
            'ln(r+re)': ln_plus(r, re),
            'ln(r+n)': ln_plus(r, n),
            'ln(re+n)': ln_plus(re, n),
            'ln(b-g+2)': ln_minus(b, g),
            'ln(b-r+2)': ln_minus(b, r),
            'ln(b-re+2)': ln_minus(b, re),
            'ln(b-n+2)': ln_minus(b, n),
            'ln(g-r+2)': ln_minus(g, r),
            'ln(g-re+2)': ln_minus(g, re),
            'ln(g-n+2)': ln_minus(g, n),
            'ln(r-re+2)': ln_minus(r, re),
            'ln(r-n+2)': ln_minus(r, n),
            'ln(re-n+2)': ln_minus(re, n),
            '(b/g)': divide(b, g),
            '(b/r)': divide(b, r),
            '(b/re)': divide(b, re),
            '(b/n)': divide(b, n),
            '(g/b)': divide(g, b),
            '(g/r)': divide(g, r),
            '(g/re)': divide(g, re),
            '(g/n)': divide(g, n),
            '(r/b)': divide(r, b),
            '(r/g)': divide(r, g),
            '(r/re)': divide(r, re),
            '(r/n)': divide(r, n),
            '(re/b)': divide(re, b),
            '(re/g)': divide(re, g),
            '(re/r)': divide(re, r),
            '(re/n)': divide(re, n),
            '(n/b)': divide(n, b),
            '(n/g)': divide(n, g),
            '(n/r)': divide(n, r),
            '(n/re)': divide(n, re),
            'ln(b/g)': ln_divide(b, g),
            'ln(b/r)': ln_divide(b, r),
            'ln(b/re)': ln_divide(b, re),
            'ln(b/n)': ln_divide(b, n),
            'ln(g/r)': ln_divide(g, r),
            'ln(g/re)': ln_divide(g, re),
            'ln(g/n)': ln_divide(g, n),
            'ln(r/re)': ln_divide(r, re),
            'ln(r/n)': ln_divide(r, n),
            'ln(re/n)': ln_divide(re, n)})

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
            're': value_normal(re),
            'n': value_normal(n),
            'ln(b)': value_ln_normal(b),
            'ln(g)': value_ln_normal(g),
            'ln(r)': value_ln_normal(r),
            'ln(re)': value_ln_normal(re),
            'ln(n)': value_ln_normal(n),
            '(b+g)': value_plus(b, g),
            '(b+r)': value_plus(b, r),
            '(b+re)': value_plus(b, re),
            '(b+n)': value_plus(b, n),
            '(g+r)': value_plus(g, r),
            '(g+re)': value_plus(g, re),
            '(g+n)': value_plus(g, n),
            '(r+re)': value_plus(r, re),
            '(r+n)': value_plus(r, n),
            '(re+n)': value_plus(re, n),
            '(b-g)': value_minus(b, g),
            '(b-r)': value_minus(b, r),
            '(b-re)': value_minus(b, re),
            '(b-n)': value_minus(b, n),
            '(g-r)': value_minus(g, r),
            '(g-re)': value_minus(g, re),
            '(g-n)': value_minus(g, n),
            '(r-re)': value_minus(r, re),
            '(r-n)': value_minus(r, n),
            '(re-n)': value_minus(re, n),
            'ln(b+g)': value_ln_plus(b, g),
            'ln(b+r)': value_ln_plus(b, r),
            'ln(b+re)': value_ln_plus(b, re),
            'ln(b+n)': value_ln_plus(b, n),
            'ln(g+r)': value_ln_plus(g, r),
            'ln(g+re)': value_ln_plus(g, re),
            'ln(g+n)': value_ln_plus(g, n),
            'ln(r+re)': value_ln_plus(r, re),
            'ln(r+n)': value_ln_plus(r, n),
            'ln(re+n)': value_ln_plus(re, n),
            'ln(b-g+2)': value_ln_minus(b, g),
            'ln(b-r+2)': value_ln_minus(b, r),
            'ln(b-re+2)': value_ln_minus(b, re),
            'ln(b-n+2)': value_ln_minus(b, n),
            'ln(g-r+2)': value_ln_minus(g, r),
            'ln(g-re+2)': value_ln_minus(g, re),
            'ln(g-n+2)': value_ln_minus(g, n),
            'ln(r-re+2)': value_ln_minus(r, re),
            'ln(r-n+2)': value_ln_minus(r, n),
            'ln(re-n+2)': value_ln_minus(re, n),
            '(b/g)': value_divide(b, g),
            '(b/r)': value_divide(b, r),
            '(b/re)': value_divide(b, re),
            '(b/n)': value_divide(b, n),
            '(g/b)': value_divide(g, b),
            '(g/r)': value_divide(g, r),
            '(g/re)': value_divide(g, re),
            '(g/n)': value_divide(g, n),
            '(r/b)': value_divide(r, b),
            '(r/g)': value_divide(r, g),
            '(r/re)': value_divide(r, re),
            '(r/n)': value_divide(r, n),
            '(re/b)': value_divide(re, b),
            '(re/g)': value_divide(re, g),
            '(re/r)': value_divide(re, r),
            '(re/n)': value_divide(re, n),
            '(n/b)': value_divide(n, b),
            '(n/g)': value_divide(n, g),
            '(n/r)': value_divide(n, r),
            '(n/re)': value_divide(n, re),
            'ln(b/g)': value_ln_divide(b, g),
            'ln(b/r)': value_ln_divide(b, r),
            'ln(b/re)': value_ln_divide(b, re),
            'ln(b/n)': value_ln_divide(b, n),
            'ln(g/r)': value_ln_divide(g, r),
            'ln(g/re)': value_ln_divide(g, re),
            'ln(g/n)': value_ln_divide(g, n),
            'ln(r/re)': value_ln_divide(r, re),
            'ln(r/n)': value_ln_divide(r, n),
            'ln(re/n)': value_ln_divide(re, n)})
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
        patient_max = 10000
        for k in range(0, 3000):  # 每個計算都算3000次 算是經驗次數
            print(k)
            score = 0
            patient = 0  # 程式除錯運算次數名稱
            while (score > 1 or score < 0.5):  # 設定回歸公式的決定係數 分數介於1~0.00001 之間算一次運算
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

                point = dict({'R2': score, '常數項': alpha,  # 給定名稱為了儲存資料  best的定義在最上面
                              '係數': beta, '預測': y_pred, '測試': y_test, '特徵組合': polynomial})
                if best['R2'] < point['R2']:
                    best = point.copy()

                patient += 1  # 程式除錯運算次數疊加

                if patient == patient_max:
                    print("FAILED")
                    break
                elif patient % 1000 == 0:
                    print("tried {}/{}".format(patient, patient_max))
            if patient == patient_max:
                break

    regression_function = ''
    for i in range(0, len(best['特徵組合'])):
        regression_function += best['特徵組合'][i] + \
            "*"+"("+str(best['係數'][i])+")"+"+"

    regression_function = name+"="+regression_function+"("+str(best['常數項'])+")"
    return best, regression_function
