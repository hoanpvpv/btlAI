import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.models import load_model
import tensorflow as tf

# Đường dẫn đến thư mục dữ liệu
data_dir = "tomatoes-data"

# Kích thước ảnh đầu vào
img_size = 256  # Đã nâng lên 256x256
batch_size = 16  # Giảm batch size để tránh lỗi bộ nhớ

# Tạo trình tăng cường dữ liệu
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

# Tạo generator cho training và validation
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_size, img_size),  # 256x256
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_size, img_size),  # 256x256
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Lấy tên các lớp
class_names = list(train_generator.class_indices.keys())
print("Các lớp cà chua:", class_names)
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

# Tạo mô hình CNN cho ảnh 256x256
model = Sequential([
    # Khối 1
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(img_size, img_size, 3)),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    
    # Khối 2
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    
    # Khối 3
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    
    # Khối 4 (thêm cho ảnh lớn hơn)
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    
    # Khối 5 (thêm cho ảnh lớn hơn)
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    
    # Khối fully connected
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(3, activation='softmax')
])

# Biên dịch mô hình
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Tóm tắt mô hình
model.summary()

# Callbacks
# Callbacks
callbacks = [
    EarlyStopping(patience=15, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.1, patience=7),
    ModelCheckpoint('tomato_classifier_256_best.keras', save_best_only=True, monitor='val_accuracy')
]
# Huấn luyện mô hình với điều chỉnh steps_per_epoch
steps_per_epoch = train_generator.samples // batch_size + (1 if train_generator.samples % batch_size != 0 else 0)
validation_steps = validation_generator.samples // batch_size + (1 if validation_generator.samples % batch_size != 0 else 0)

# Huấn luyện mô hình
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    epochs=50,
    callbacks=callbacks
)

# Vẽ biểu đồ
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

# Lưu mô hình với định dạng h5
# Sau khi huấn luyện
# Lưu mô hình cuối cùng với định dạng h5
tf.keras.models.save_model(model, 'tomato_classifier_256x256.h5', save_format='h5')
print("Đã lưu mô hình tại 'tomato_classifier_256x256.h5'")


# Hàm dự đoán cho ảnh kích thước 256x256
def predict_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))  # Resize về 256x256
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    
    print(f"Loại cà chua: {class_names[predicted_class]}")
    print(f"Độ tin cậy: {prediction[0][predicted_class]:.2f}")
    
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.resize(cv2.imread(img_path), (img_size, img_size))[:,:,::-1])
    plt.title(f"{class_names[predicted_class]} ({prediction[0][predicted_class]:.2f})")
    plt.axis('off')
    plt.show()
    
    return class_names[predicted_class], prediction[0][predicted_class]
