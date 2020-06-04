# ---- 建立 VGG16 卷積基底 ---- #

from tensorflow.keras.applications import VGG16

vgg16 = VGG16(include_top=False,
                    weights='imagenet',
                    input_shape=(150,150,3),
              )
vgg16.summary()

# ---- ImageDataGenerator ---- #
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# -- 訓練資料生成器 -- #
gobj = ImageDataGenerator(rescale=1./255, validation_split=0.75)  # 8000*0.25 = 2000 訓練資料


batch_size = 20   # 需注意批次數
trn_gen = gobj.flow_from_directory( #←建立生成訓練資料的生成器
    'cat_dog/train',                #←指定目標資料夾
    target_size=(150, 150),         #←調整所有影像大小成 150x150
    batch_size=batch_size,          #←每批次要生成多少筆資料
    class_mode='binary',            #←指定分類方式, 這裡是設為二元分類
    subset='training')              #←只生成前 75% 的訓練資料

samples = batch_size * len(trn_gen)
print(f'共有 {samples} 筆訓練資料')
print(f'可分為 {len(trn_gen)} 批次')

features = np.zeros(shape=(samples, 4, 4, 512))    # 用來儲存萃取出的特徵資料
labels = np.zeros(shape=(samples))                 # 用來儲存標籤

b_num = 0   # 計數批次數
for inputs_batch, labels_batch in trn_gen:
  features_batch = vgg16.predict(inputs_batch)          # 使用 VGG16 進行特徵萃取
  features[b_num * batch_size : (b_num + 1) * batch_size] = features_batch
  labels[b_num * batch_size : (b_num + 1) * batch_size] = labels_batch
  b_num += 1
  print(f'處理完 {b_num} 批次資料', end='\r')
  if b_num == len(trn_gen):
        print('')
        print('訓練資料萃取完成')
        break

print(features.shape)
print(labels.shape)

# ---- 存成 pickle 檔 ---- #

import pickle

with open('train.pickle','wb') as f:    # 將訓練資料與標籤存成 pickle 檔
    pickle.dump(features, f)       # 第 1 次 dump 訓練資料
    pickle.dump(labels, f)         # 第 2 次 dump 訓練標籤     

# -- 訓練驗證生成器 -- #
gobj = ImageDataGenerator(rescale=1./255)   # 全部都做為驗證資料

val_gen = gobj.flow_from_directory( #←建立生成驗證資料的生成器
    'cat_dog/test',          #←指定要讀取測試資料夾
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='binary')

samples = batch_size * len(val_gen)
print(f'共有 {samples} 筆驗證資料 ')
print(f'可分為 {len(val_gen)} 批次')

features = np.zeros(shape=(samples, 4, 4, 512))    # 用來儲存萃取出的特徵資料
labels = np.zeros(shape=(samples))                 # 用來儲存標籤

b_num = 0   # 計數批次數
for inputs_batch, labels_batch in val_gen:
  features_batch = vgg16.predict(inputs_batch)          # 使用 VGG16 進行特徵萃取
  features[b_num * batch_size : (b_num + 1) * batch_size] = features_batch
  labels[b_num * batch_size : (b_num + 1) * batch_size] = labels_batch
  b_num += 1
  print(f'處理完 {b_num} 批次資料', end='\r')
  if b_num == len(val_gen):
        print('')
        print('訓練資料萃取完成')
        break

print(features.shape)
print(labels.shape)

# ---- 存成 pickle 檔 ---- #

import pickle

with open('val.pickle','wb') as f:    # 將訓練資料與標籤存成 pickle 檔
    pickle.dump(features, f)       # 第 1 次 dump 驗證資料
    pickle.dump(labels, f)         # 第 2 次 dump 標籤     