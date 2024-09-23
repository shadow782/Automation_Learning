'''
好好学习
天天向上
'''
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Dense, GRU, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import ctc_batch_cost


# 数据预处理函数
def preprocess_data(images, labels, img_width, img_height, max_text_len, char_list):
    # 将图像调整到统一大小
    images = [tf.image.resize(img, (img_height, img_width)) for img in images]
    images = np.array(images)

    # 将标签转换为字符索引
    char_to_num = {char: i for i, char in enumerate(char_list)}
    labels = [[char_to_num[char] for char in label] for label in labels]

    # 将标签填充到相同长度
    labels = tf.keras.preprocessing.sequence.pad_sequences(labels, maxlen=max_text_len, padding='post')

    return images, labels


# 构建OCR模型
def build_ocr_model(img_width, img_height, max_text_len, char_list):
    input_img = Input(shape=(img_height, img_width, 1), name='image_input')
    labels = Input(shape=(max_text_len,), name='label_input')

    # 卷积层提取特征
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    # 将特征图转换为序列
    x = Reshape((img_width // 4, (img_height // 4) * 64))(x)

    # 双向GRU处理序列
    x = Bidirectional(GRU(128, return_sequences=True))(x)
    x = Bidirectional(GRU(128, return_sequences=True))(x)

    # 全连接层输出字符概率分布
    x = Dense(len(char_list) + 1, activation='softmax')(x)

    # 定义模型
    model = Model(inputs=[input_img, labels], outputs=x)

    return model


# 自定义CTC损失函数
def ctc_loss(y_true, y_pred):
    input_length = np.ones(y_pred.shape[0]) * y_pred.shape[1]
    label_length = np.ones(y_true.shape[0]) * y_true.shape[1]

    return ctc_batch_cost(y_true, y_pred, input_length, label_length)


# 数据集参数
img_width, img_height = 128, 32
max_text_len = 32
char_list = 'abcdefghijklmnopqrstuvwxyz'

# 构建模型
model = build_ocr_model(img_width, img_height, max_text_len, char_list)
model.compile(optimizer=Adam(), loss=ctc_loss)

# 加载数据并进行预处理
# images, labels = load_data()  # 加载数据集函数需自行实现
# images, labels = preprocess_data(images, labels, img_width, img_height, max_text_len, char_list)

# 训练模型
# model.fit([images, labels], labels, batch_size=32, epochs=50, validation_split=0.2)

# 保存模型
# model.save('ocr_model.h5')
