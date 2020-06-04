from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential  #← 匯入 Keras 的序列式模型類別
from tensorflow.keras.layers import Dense       #← 匯入 Keras 的密集層類別

#傳回預處理好的 MNIST 資料集： (x_train, x_test), (y_train, y_test)
def mnist_data():
    # 載入 MNIST 資料集並預處理樣本 & 標籤資料
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    x_train = train_images.reshape((60000, 28 * 28)) #←將 (60000,28,28) 轉換成 (60000,784)
    x_train = x_train.astype('float32') / 255    #←再將 0~255 的像素值轉換成 0~1 的浮點數
    x_test = test_images.reshape((10000, 28 * 28))  #}←將 10000 筆測試樣本做同樣的轉換
    x_test = x_test.astype('float32') / 255         #}
    y_train = to_categorical(train_labels)  #←將訓練標籤做 One-hot 編碼
    y_test  = to_categorical(test_labels)   #←將測試標籤做 One-hot 編碼
    return (x_train, x_test), (y_train, y_test)

#傳回新建立並編譯好的 MNIST 模型
def mnist_model():
    model = Sequential()                 #← 建立序列模型物件
    model.add(Dense(512, activation='relu', input_dim= 784)) #← 加入第一層
    model.add(Dense(10, activation='softmax'))               #← 加入第二層
    model.compile(optimizer='rmsprop',             #← 指定優化器
                  loss='categorical_crossentropy', #← 指定損失函數
                  metrics=['acc'])                 #← 指定評量準則
    return model

###################################################################

import matplotlib.pyplot as plt

# 繪製線圖 (可將訓練時所傳回的損失值或準確率等歷史記錄繪製成線圖)
# history: 內含一或多筆要繪資料的字典, 例如：{'loss': [4,2,1,…], 'acc': [2,3,5,…]}
# keys: 以 tuple 或串列指定 history 中要繪製的 key 值, 例如：('loss', 'acc')
# title: 以字串指定圖表的標題文字
# xyLabel: 以 tuple 或串列指定 x, y 軸的說明文字, 例如：('epoch', 'Accuracy')
# ylim: 以 tuple 或串列指定 y 軸的最小值及最大值, 例如 (1, 3), 超出範圍的值會被忽略
# size: 以 tuple 指定圖的尺寸, 預設為 (6, 4) (即寬 6 高 4 英吋)
def plot(history_dict, keys, title=None, xyLabel=[], ylim=(), size=()):
    lineType = ('-', '--', '.', ':')    # 線條的樣式, 畫多條線時會依序採用
    if len(ylim)==2: plt.ylim(*ylim)    # 設定 y 軸最小值及最大值
    if len(size)==2: plt.gcf().set_size_inches(*size)  # size預設為 (6,4)
    epochs = range(1, len(history_dict[keys[0]])+1)  # 計算有幾週期的資料
    for i in range(len(keys)):   # 走訪每一個 key (例如 'loss' 或 'acc' 等)
        plt.plot(epochs, history_dict[keys[i]], lineType[i])  # 畫出線條
    if title:   # 是否顯示標題欄
        plt.title(title)
    if len(xyLabel)==2:  # 是否顯示 x, y 軸的說明文字
        plt.xlabel(xyLabel[0])
        plt.ylabel(xyLabel[1])
    plt.legend(keys, loc='best') # 顯示圖例 (會以 key 為每條線的說明)
    plt.show()  # 顯示出畫好的圖

#繪製多個影像資料及其對應的標籤值
#參數：
#  imgs、labels 為影像資料及標籤值
#  grid 為 (水平, 垂直) 個子圖表, 預設為水平 1 排中有 10 個子圖表
#  size 為圖表的 (寬, 高), 預設為 (20英吋, 2英吋)
def plot_img(imgs, labels=[], grid=(1, 10), size=(20, 2)):
    n = len(imgs)                     #←計算樣本數
    plt.gcf().set_size_inches(size)   #←指定圖表的寬度,高度 (英吋=2.54公分)
    for i in range(n):                #←依樣本數走訪
        ax = plt.subplot(grid[0], grid[1], i+1)   #←取得對應位置的子圖表 (共有 (1x樣本數) 個子圖表)
        ax.imshow(imgs[i])            #←顯示樣本 (影像) 內容
        if len(labels):
            ax.set_title(f'Label={labels[i]}')    #←以對應的標籤值做為子圖表的標題
        ax.set_xticks([]); ax.set_yticks([]) #←設定 X, Y 軸不顯示刻度
    plt.show()      #←實際顯示圖片
