#
```
TensorFlow 2.0 捲積神經網絡實戰
王曉華  清華大學  簡體中文
ISBN:7302540659
ISBN-13:9787302540656


https://www.tenlong.com.tw/products/9787302540656?list_name=srh
```

```
第1章 Python和TensorFlow 2.0的安裝 1

1.1 Python基本安裝和用法 1
1.1.1 Anaconda的下載與安裝 2
1.1.2 Python編譯器PyCharm 的安裝 5
1.1.3 使用Python計算softmax函數 8

1.2 TensorFlow 2.0 GPU版本的安裝 9
1.2.1 檢測Anaconda中的TensorFlow版本 9
1.2.2 TensorFlow 2.0 GPU版本基礎顯卡推薦和前置軟件安裝 10
1.3 Hello TensorFlow 13
1.4 本章小結 14

第2章 簡化代碼的複雜性：TensorFlow 2.0基礎與進階 15
2.1 配角轉成主角——從TensorFlow Eager Execution轉正談起 16
2.1.1 Eager簡介與調用 16
2.1.2 讀取數據 18
2.1.3 使用TensorFlow 2.0模式進行線性回歸的一個簡單的例子 20

2.2 Hello TensorFlow & Keras 22
2.2.1 MODEL！MODEL！MODEL！還是MODEL 23
2.2.2 使用Keras API實現鳶尾花分類的例子（順序模式） 24
2.2.3 使用Keras 函數式編程實現鳶尾花分類的例子（重點） 27
2.2.4 使用保存的Keras模式對模型進行複用 30
2.2.5 使用TensorFlow 2.0標準化編譯對Iris模型進行擬合 31
2.2.6 多輸入單一輸出TensorFlow 2.0編譯方法（選學） 35
2.2.7 多輸入多輸出TensorFlow 2.0編譯方法（選學） 39

2.3 全連接層詳解 41
2.3.1 全連接層的定義與實現 41
2.3.2 使用TensorFlow 2.0自帶的API實現全連接層 43
2.3.3 打印顯示TensorFlow 2.0設計的Model結構和參數 46
2.4 本章小結 48

 

第3章 TensorFlow 2.0語法基礎 49
3.1 BP神經網絡簡介 49

3.2 BP神經網絡兩個基礎算法詳解 53
3.2.1 最小二乘法（LS算法）詳解 53
3.2.2 道士下山的故事——梯度下降算法 56

3.3 反饋神經網絡反向傳播算法介紹 59
3.3.1 深度學習基礎 59
3.3.2 鍊式求導法則 61
3.3.3 反饋神經網絡原理與公式推導 62
3.3.4 反饋神經網絡原理的激活函數 68
3.3.5 反饋神經網絡原理的Python實現 70

3.4 本章小結 74


第4章 卷積層詳解與MNIST實戰 75
4.1 卷積運算基本概念 75
4.1.1 卷積運算 76
4.1.2 TensorFlow 2.0中卷積函數實現詳解 78
4.1.3 池化運算 80
4.1.4 softmax激活函數 81
4.1.5 卷積神經網絡原理 83

4.2 TensorFlow 2.0編程實戰—MNIST手寫體識別 86
4.2.1 MNIST數據集 86
4.2.2 MNIST數據集特徵和標籤介紹 88
4.2.3 TensorFlow 2.0編程實戰MNIST數據集 90
4.2.4 使用自定義的捲積層實現MNIST識別 95
4.3 本章小結 98


第5章 TensorFlow 2.0 Dataset使用詳解 99
5.1 Dataset API基本結構和內容 99
5.1.1 Dataset API數據種類 100
5.1.2 Dataset API基礎使用 101
5.2 Dataset API高級用法 102
5.2.1 Dataset API數據轉換方法 104
5.2.2 一個讀取圖片數據集的例子 108

5.3 使用TFRecord API創建和使用數據集 108
5.3.1 TFRecord詳解 109
5.3.2 TFRecord的創建 111
5.3.3 TFRecord的讀取 115

5.4 TFRecord實戰——帶有處理模型的完整例子 121
5.4.1 創建數據集 121
5.4.2 創建解析函數 122
5.4.3 創建數據模型 123
5.4 本章小結 124

 

第6章 從冠軍開始：ResNet 125
6.1 ResNet基礎原理與程序設計基礎 125
6.1.1 ResNet誕生的背景 126
6.1.2 模塊工具的TensorFlow實現——不要重複造輪子 129
6.1.3 TensorFlow高級模塊layers用法簡介 129

6.2 ResNet實戰CIFAR-100數據集分類 137
6.2.1 CIFAR-100數據集簡介 137
6.2.2 ResNet殘差模塊的實現 140
6.2.3 ResNet網絡的實現 142
6.2.4 使用ResNet對CIFAR-100進行分類 145

6.3 ResNet的兄弟——ResNeXt 147
6.3.1 ResNeXt誕生的背景 147
6.3.2 ResNeXt殘差模塊的實現 149
6.3.3 ResNeXt網絡的實現 150
6.3.4 ResNeXt和ResNet的比較 151

6.4 其他的捲積神經模型簡介 152
6.4.1 SqueezeNet模型簡介 153
6.4.2 Xception模型簡介 155
6.5 本章小結 156

 

第7章 Attention is all we need! 157

7.1 簡單的理解注意力機制 158
7.1.1 何為“注意力” 158
7.1.2 “hard or soft？”—注意力機制的兩種常見形式 159
7.1.3 “Spatial and Channel！”—注意力機制的兩種實現形式 160

7.2 SENet 和CBAM注意力機制的經典模型 163
7.2.1 最後的冠軍—SENet 163
7.2.2 結合了Spatial和Channel的CBAM模型 166
7.2.3 注意力的前沿研究—基於細粒度的圖像注意力機制 171
7.3 本章小結 173

CBAM: Convolutional Block Attention Module
Sanghyun Woo, Jongchan Park, Joon-Young Lee, In So Kweon
https://arxiv.org/abs/1807.06521
We propose Convolutional Block Attention Module (CBAM), 
a simple yet effective attention module for feed-forward convolutional neural networks. 

第8章 卷積神經網絡實戰：識文斷字我也可以 174

8.1 文本數據處理 175
8.1.1 數據集介紹和數據清洗 175
8.1.2 停用詞的使用 177
8.1.3 詞向量訓練模型word2vec使用介紹 180
8.1.4 文本主題的提取—基於TF-IDF（選學） 183
8.1.5 文本主題的提取—基於TextRank（選學） 187

8.2 針對文本的捲積神經網絡模型簡介—字符卷積 190
8.2.1 字符（非單詞）文本的處理 191
8.2.2 卷積神經網絡文本分類模型的實現—Conv1D（一維卷積） 199

8.3 針對文本的捲積神經網絡模型簡介—詞卷積 201
8.3.1 單詞的文本處理 201
8.3.2 卷積神經網絡文本分類模型的實現—Conv2D（二維卷積） 203
8.4 使用卷積對文本分類的補充內容 207
8.4.1 漢字的文本處理 207
8.4.2 其他的一些細節 210
8.5 本章小結 211
```
