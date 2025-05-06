import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import time

# Thông số cấu hình
IMG_SIZE = 128
MODEL_PATH = 'tomato_classifier_3classes.h5'

# Tên các lớp phân loại (cần khớp với thứ tự trong lúc train)
CLASS_NAMES = ['Damaged', 'Ripe', 'Unripe']  # Thay đổi các lớp phù hợp với dataset của bạn

# Load model đã train
print("Đang tải model...")
model = load_model(MODEL_PATH)
print("Đã tải model thành công!")

# Khởi tạo webcam
cap = cv2.VideoCapture(1)  # 0 là camera mặc định

if not cap.isOpened():
    print("Không thể mở camera!")
    exit()


def preprocess_image(image):
    # Resize và chuẩn hóa ảnh giống như lúc train
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0  # Chuẩn hóa pixel values về khoảng [0,1]
    return image


def detect_tomato(frame):
    # Chuyển sang không gian màu HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Định nghĩa dải màu đỏ cho cà chua (có thể điều chỉnh)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    # Tạo mask
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2
    
    # Loại bỏ nhiễu
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Tìm các contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Lọc các contour quá nhỏ
    min_area = 1000  # Diện tích tối thiểu (có thể điều chỉnh)
    tomato_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    return tomato_contours


print("Bắt đầu phân loại cà chua qua camera. Nhấn 'q' để thoát.")

while True:
    # Đọc frame từ camera
    ret, frame = cap.read()
    
    if not ret:
        print("Không thể nhận frame từ camera. Đang thoát...")
        break
    
    # Tạo bản sao frame gốc để hiển thị
    display_frame = frame.copy()
    
    # Xử lý ảnh cho model
    processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Chuyển BGR sang RGB
    processed_frame = preprocess_image(processed_frame)
    
    # Dự đoán
    prediction = model.predict(np.expand_dims(processed_frame, axis=0), verbose=0)
    predicted_class_index = np.argmax(prediction[0])
    predicted_class = CLASS_NAMES[predicted_class_index]
    confidence = float(prediction[0][predicted_class_index])
    
    # Tìm vùng có cà chua
    tomato_contours = detect_tomato(frame)
    
    # Vẽ khung và nhãn cho từng cà chua phát hiện được
    for contour in tomato_contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # Vẽ khung hình chữ nhật
        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Hiển thị tên lớp bên trong khung
        label = f"{predicted_class}: {100*confidence:.1f}%"
        cv2.putText(display_frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Hiển thị thông tin tổng thể
    info_text = f"Tổng số cà chua: {len(tomato_contours)}"
    cv2.putText(display_frame, info_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # Hiển thị frame
    cv2.imshow('Phân loại cà chua', display_frame)
    
    # Thoát nếu nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
print("Đã kết thúc phân loại.")
