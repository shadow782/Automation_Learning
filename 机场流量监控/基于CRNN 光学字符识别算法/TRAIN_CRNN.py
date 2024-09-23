import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Dense, GRU, Bidirectional, BatchNormalization, Activation
from tensorflow.keras.models import Model

# 构建CRNN模型
def build_crnn_model(img_width, img_height, num_classes):
    input_img = Input(shape=(img_height, img_width, 1), name='image_input')

    # 卷积层提取特征
    x = Conv2D(32, (3, 3), padding='same')(input_img)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)

    # 将特征图转换为序列
    x = Reshape((img_width // 8, (img_height // 8) * 128))(x)

    # 双向GRU处理序列
    x = Bidirectional(GRU(256, return_sequences=True))(x)
    x = Bidirectional(GRU(256, return_sequences=True))(x)

    # 全连接层输出字符概率分布
    x = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=input_img, outputs=x)
    return model

# 数据集参数
img_width, img_height = 128, 32
num_classes = len(char_list) + 1  # 包括一个空白字符

# 构建CRNN模型
crnn_model = build_crnn_model(img_width, img_height, num_classes)
crnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
# crnn_model.fit(train_data, train_labels, validation_data=(val_data, val_labels), batch_size=32, epochs=50)
# crnn_model.save('crnn_model.h5')
