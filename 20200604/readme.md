# 推薦教材
```
tf.keras 技術者們必讀！深度學習攻略手冊
施威銘研究室 著
出版商:旗標科技
出版日期:2020-02-13

7-4 遷移學習 - 以預訓練好的經典模型 VGG16 為例
7-4-0 什麼是遷移學習 (transfer learning)
7-4-1 萃取出資料的特徵
7-4-2 將經典 CNN 移植到新模型之中
7-4-3 模型的微調 (fine-tuning)
```
#
```
# -*- coding: utf-8 -*-
"""first.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1J7en8f4hkelOn_RLazt2N5OB2V-VUi-l
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from google.colab import drive
drive.mount('/content/drive')

!pwd

import os
os.chdir('drive')

!pwd

import os
os.chdir('TF2020')

!ls

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#model = tf.keras.applications.InceptionV3(include_top=True, weights='imagenet')
model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')

model.summary()

!wget https://upload.wikimedia.org/wikipedia/commons/f/f9/Zoorashia_elephant.jpg

from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.applications.inception_v3 import decode_predictions

def read_img(img_path, resize=(299,299)):
    img_string = tf.io.read_file(img_path)  # 讀取檔案
    img_decode = tf.image.decode_image(img_string)  # 將檔案以影像格式來解碼
    img_decode = tf.image.resize(img_decode, resize)  # 將影像resize到網路輸入大小
    # 將影像格式增加到4維(batch, height, width, channels)，模型預測要求格式
    img_decode = tf.expand_dims(img_decode, axis=0)
    return img_decode

!ls

img_path = 'Zoorashia_elephant.jpg'
img = read_img(img_path)  # 透過剛創建的函式讀取影像
plt.imshow(tf.cast(img, tf.uint8)[0])

img = preprocess_input(img)  # 影像前處理
preds = model.predict(img)  # 預測圖片
print("Predicted:", decode_predictions(preds))
#print("Predicted:", decode_predictions(preds, top=3)[0])

```
