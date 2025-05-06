import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model

# Đường dẫn đến thư mục dữ liệu (đã cập nhật chỉ còn 3 lớp)
data_dir = "tomatoes-data"

# Kích thước ảnh đầu vào
img_size = 128
batch_size = 32

# Tạo trình tăng cường dữ liệu để tạo thêm biến thể và chống overfitting
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Dùng 20% dữ liệu làm validation
)

# Tạo generator cho training và validation
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Lấy tên các lớp (bây giờ chỉ có 3 lớp)
class_names = list(train_generator.class_indices.keys())
print("Các lớp cà chua:", class_names)

# Kiểm tra số lớp
assert len(class_names) == 3, "Số lớp không phải là 3, vui lòng kiểm tra thư mục dữ liệu!"


# Hiển thị vài ảnh mẫu
def show_batch(image_batch, label_batch):
    plt.figure(figsize=(10, 10))
    for i in range(min(9, len(image_batch))):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(image_batch[i])
        plt.title(class_names[np.argmax(label_batch[i])])
        plt.axis("off")
    plt.show()


# Lấy một batch để hiển thị
images, labels = next(train_generator)
show_batch(images, labels)

# Tạo mô hình CNN
model = Sequential([
    # Khối 1
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(img_size, img_size, 3)),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    
    # Khối 2
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    
    # Khối 3
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    
    # Khối fully connected
    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(3, activation='softmax')  # Đầu ra cho 3 lớp cà chua
])

# Biên dịch mô hình
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Tóm tắt mô hình
model.summary()

# Callbacks để cải thiện việc huấn luyện
callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.1, patience=5)
]

# Huấn luyện mô hình
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=50,
    callbacks=callbacks
)

# Vẽ biểu đồ accuracy và loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Lưu mô hình
model.save('tomato_classifier_3classes.h5')
print("Đã lưu mô hình tại 'tomato_classifier_3classes.h5'")

# model = load_model('tomato_classifier_3classes.h5')
# print("Đã tải mô hình thành công!")

# # Hàm dự đoán cho một ảnh mới
# def predict_image(img_path):
#     img = cv2.imread(img_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, (img_size, img_size))
#     img = img / 255.0
#     img = np.expand_dims(img, axis=0)
    
#     prediction = model.predict(img)
#     predicted_class = np.argmax(prediction)
    
#     return class_names[predicted_class], prediction[0][predicted_class]

# # Gọi hàm và lưu kết quả
# class_name, confidence = predict_image("tomatoes-data\\Ripe\\r (428).jpg")

# # In kết quả
# print(f"Loại cà chua: {class_name}")
# print(f"Độ tin cậy: {confidence:.2f}")
