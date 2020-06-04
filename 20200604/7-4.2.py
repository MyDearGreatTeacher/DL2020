# ---- 建立並訓練密集層分類器 ---- #

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.applications import VGG16

vgg16 = VGG16(include_top=False,
              weights='imagenet',
              input_shape=(150,150,3))

model = Sequential()
model.add(vgg16)    # 將 vgg16 做為一層
model.add(Flatten())
model.add(Dense(512, activation='relu', input_dim=4 * 4 * 512))
model.add(Dropout(0.5))  # 丟棄法
model.add(Dense(1, activation='sigmoid'))

vgg16.trainable = False     # 凍結權重
model.summary()

model.compile(optimizer=RMSprop(lr=2e-5),   # 學習速率降低一點
              loss='binary_crossentropy',
              metrics=['acc'])
# ---- 設定資料擴增 ---- #
from tensorflow.keras.preprocessing.image import ImageDataGenerator

###使用資料擴增法生成訓練資料
gobj = ImageDataGenerator(rescale=1./255, validation_split=0.75,
    rotation_range=40,      #←隨機旋轉 -40~40 度
    width_shift_range=0.2,  #←隨機向左或右平移 20% 寬度以內的像素
    height_shift_range=0.2, #←隨機向上或下平移 20% 高度以內的像素
    shear_range=10,         #←隨機順時針傾斜影像 0~10 度
    zoom_range=0.2,         #←隨機水平或垂直縮放影像 20% (80%~120%)
    horizontal_flip=True)   #←隨機水平翻轉影像

trn_gen = gobj.flow_from_directory( #←建立生成訓練資料的走訪器
    'cat_dog/train',         #←指定目標資料夾
    target_size=(150, 150),  #←調整所有影像大小成 150x150
    batch_size=50,        #←每批次要生成多少筆資料
    class_mode='binary',     #←指定分類方式, 這裡是設為二元分類
    subset='training')       #←只生成前 75% 的訓練資料

gobj = ImageDataGenerator(rescale=1./255)   # 驗證資料不使用資料擴增

val_gen = gobj.flow_from_directory( #←建立生成驗證資料的走訪器
    'cat_dog/test',          #←指定要讀取測試資料夾
    target_size=(150, 150),
    batch_size=50,
    class_mode='binary')

history = model.fit(trn_gen,        #←指定訓練用的走訪器
                    epochs=30, verbose=2,
                    validation_data=val_gen) #←指定驗證用的走訪器



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
