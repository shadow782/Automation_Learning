from tensorflow.keras.preprocessing import image
import numpy as np

# 加载训练好的CRNN模型
crnn_model = tf.keras.models.load_model('crnn_model.h5')


# 预处理图像
def preprocess_image(img_path, img_width, img_height):
    img = image.load_img(img_path, target_size=(img_height, img_width), color_mode='grayscale')
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img


# 字符列表
char_list = 'abcdefghijklmnopqrstuvwxyz'


# 将预测的字符索引转换为字符
def decode_predictions(pred):
    decoded = ''
    for p in pred:
        if p != -1:
            decoded += char_list[p]
    return decoded


# 加载并预处理文本区域图像
for i in range(len(boxes)):
    if i in indexes:
        img_path = f"text_region_{i}.jpg"
        img = preprocess_image(img_path, img_width, img_height)

        # 进行预测
        preds = crnn_model.predict(img)
        pred_text = decode_predictions(np.argmax(preds[0], axis=1))

        print(f"Detected text: {pred_text}")
