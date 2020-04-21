# DEEPLEARNING2020
```
官方網址: 
https://www.tensorflow.org/

教學範例:
https://www.tensorflow.org/tutorials

API說明:
TensorFlow Core v2.1.0
```
```
極詳細+超深入：最新版TensorFlow 1.x/2.x完整工程實作
作者： 李金洪  出版社：深智數位  
出版日期：2020/01/20
```
# 20200423  Marketing Data science +Kaggle專案
```
Google Analytics Customer Revenue Prediction
Predict how much GStore customers will spend


https://www.kaggle.com/c/ga-customer-revenue-prediction/data

http://www.7daixie.com/2020012122929031.html


他山之石可以估錯　
Google Analytics Customer Revenue Prediction　github

https://github.com/abdkumar/Google-Analytics-Customer-Revenue-Prediction/blob/master/customer%20revenue%20prediction.ipynb

https://github.com/Subhankar29/Google-Analytics-Customer-Revenue-Prediction
```
```
Data Science for Marketing Analytics
Tommy Blanchard, Debasish Behera, Et al
March 30, 2019

https://www.packtpub.com/big-data-and-business-intelligence/data-science-marketing-analytics
```
```
Marketing Data Science: Modeling Techniques in Predictive Analytics with R and Python (Hardcover)
Thomas W. Miller
Prentice Hall


```
```
https://medium.com/marketingdatascience/%E7%94%A8python%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%E6%AD%A5%E6%AD%A5%E6%89%93%E9%80%A0-%E8%87%AA%E5%B7%B1%E7%9A%84%E5%AE%A2%E6%88%B6%E7%B2%BE%E6%BA%96%E5%90%8D%E5%96%AE-%E9%99%84python%E7%A8%8B%E5%BC%8F%E7%A2%BC-e499041c4edd
```

# 20200312 資料分析
### NUMPY
```
NumPy 高速運算徹底解說：六行寫一隻程式？你真懂深度學習？手工算給你看！
現場で使える! NumPyデータ処理入門
作者： 吉田拓真, 尾原颯   譯者： 吳嘉芳, 蒲宗賢
編者： 施威銘研究室  旗標出版社
```

### PANDAS
```
Learning Pandas - Second Edition
https://github.com/PacktPublishing/Learning-Pandas-Second-Edition
```
```
Python資料分析 第二版  Python for Data Analysis, 2nd Edition
作者： Wes McKinney   譯者： 張靜雯  歐萊禮出版社
出版日期：2018/10/03

```
### PPT
```
https://github.com/MyDearGreatTeacher/uTaipei2019
```
# 20200409 Tensorflow 2.0
```
import pandas as pd
print("pandas version: %s" % pd.__version__)

import matplotlib
print("matplotlib version: %s" % matplotlib.__version__)

import numpy as np
print("numpy version: %s" % np.__version__)

import sklearn
print("scikit-learn version: %s" % sklearn.__version__)

import tensorflow as tf
print("tensorflow version: %s" % tf.__version__)

import torch
print("PyTorch version: %s" %torch.__version__)
print("2020年3月PyTorch version最新版本 是1.4 請參閱https://pytorch.org/")
```
```
20200409:
matplotlib version: 3.2.1
numpy version: 1.18.2
scikit-learn version: 0.22.2.post1
tensorflow version: 2.2.0-rc2
PyTorch version: 1.4.0
2020年3月PyTorch version最新版本 是1.4 請參閱https://pytorch.org/
```
### Google colab版本切換
```
[1]重開一個Notebook
[2]輸入並執行 
    %tensorflow_version 1.x
```

![Google colab版本切換](pic/TENSORFLOW_20200409_1.png)
```
在colab中使用tensorflow2.1或2.0
https://blog.csdn.net/qq_42145862/article/details/104217873
```
### Neural Network
```
# coding: utf-8
import numpy as np


def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

if __name__ == '__main__':
    for xs in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        y = AND(xs[0], xs[1])
        print(str(xs) + " -> " + str(y))
```
![]()

### Neural Network with Learning capability==regression
```


```

# 作業:閱讀報告
```
LeCun, Y., Bengio, Y. and Hinton, G. E. (2015)
Deep Learning
Nature, Vol. 521, pp 436-444. 
https://www.cs.toronto.edu/~hinton/absps/NatureDeepReview.pdf
```
