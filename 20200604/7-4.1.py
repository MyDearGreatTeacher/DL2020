
# ---- 載入 pickle 檔 ---- #

import pickle

with open('train.pickle','rb') as f1:    # 將訓練資料與標籤存成 pickle 檔
    train_features = pickle.load(f1)       # 第 1 次 load 為訓練資料
    train_labels = pickle.load(f1)         # 第 2 次 load 為訓練標籤   
    print(train_features.shape)             # (2000, 4, 4, 512)
    print(train_labels.shape)               # (2000)

with open('val.pickle','rb') as f2:    # 將驗證資料與標籤存成 pickle 檔
    val_features = pickle.load(f2)        # 第 1 次 load 為驗證資料
    val_labels = pickle.load(f2)          # 第 2 次 load 為驗證標籤
    print(val_features.shape)               # (2000, 4, 4, 512)
    print(val_labels.shape)                 # (2000)


# ---- 將資料展平 ---- #
import numpy as np
train_features = np.reshape(train_features, (2000, 4 * 4 * 512))
val_features = np.reshape(val_features, (2000, 4 * 4 * 512))

# ---- 建立並訓練密集層分類器 ---- #

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop

model = Sequential()
model.add(Dense(512, activation='relu', input_dim=4 * 4 * 512))
model.add(Dropout(0.5))  # 丟棄法
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer=RMSprop(lr=2e-5),   # 學習速率降低一點
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(train_features, 
                    train_labels,
                    epochs=100,
                    batch_size=50,
                    validation_data=(val_features, val_labels))

# --  繪製圖表 -- # 
import util7 as u    	# 匯入自訂的繪圖工具模組

u.plot( history.history,   # 繪製準確率與驗證準確度的歷史線圖
        ('acc', 'val_acc'),
        'Training & Vaildation Acc',
        ('Epoch','Acc'), 
        )     

u.plot( history.history,   #  繪製損失及驗證損失的歷史線圖
        ('loss', 'val_loss'),
        'Training & Vaildation Loss',
        ('Epoch','Loss'), 
        )





