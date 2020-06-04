from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.applications import VGG16

vgg16 = VGG16(include_top=False,    # 建立 vgg16 卷積基底
              weights='imagenet',
              input_shape=(150,150,3))

unfreeze = ['block5_conv1', 'block5_conv2', 'block5_conv3'] # 最後 3 層的名稱

for layer in vgg16.layers:
    if layer.name in unfreeze:
        layer.trainable = True  # 最後 3 層解凍
    else:
        layer.trainable = False # 其他凍結權重

vgg16.summary() # 再次查看 vgg16 模型資訊

# ---- 建立分類模型 ---- #

model = Sequential()
model.add(vgg16)    # 將 vgg16 做為一層
model.add(Flatten())
model.add(Dense(512, activation='relu', input_dim=4 * 4 * 512))
model.add(Dropout(0.5))  # 丟棄法
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=RMSprop(lr=1e-5),   # 學習速率從 2e-5 -> 1e-5
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
                    epochs=50, verbose=2,
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

def to_EMA(points, a=0.3):  #←將歷史資料中的數值轉為 EMA 值
  ret = []          # 儲存轉換結果的串列
  EMA = points[0]   # 第 0 個 EMA 值
  for pt in points:
    EMA = pt*a + EMA*(1-a)  # 本期EMA = 本期值*0.3 + 前期EMA * 0.7
    ret.append(EMA)         # 將本期EMA加入串列中
  return ret

hv = to_EMA(history.history['val_acc'])  # 將 val_acc 歷史資料的值轉成 EMA 值

history.history['ema_acc'] = hv
u.plot(history.history, ('acc','val_acc', 'ema_acc'),    # 繪製準確度歷史線圖
        'Training & Validation accuracy', ('Epochs', 'Accuracy'))
