import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model

model = load_model('tomato_classifier_3classes.h5')
print("Đã tải mô hình thành công!")


# Hàm dự đoán cho một ảnh mới
def predict_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    
    return class_names[predicted_class], prediction[0][predicted_class]


# Gọi hàm và lưu kết quả
class_name, confidence = predict_image("tomatoes-data\\Ripe\\r (428).jpg")

# In kết quả
print(f"Loại cà chua: {class_name}")
print(f"Độ tin cậy: {confidence:.2f}")
