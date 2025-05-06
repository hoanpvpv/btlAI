import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from glob import glob


def detect_and_crop_tomato(image_path, output_folder=None, show_result=True):
    """
    Phát hiện và cắt cận quả cà chua từ ảnh sử dụng phương pháp phát hiện hình cầu
    
    Parameters:
        image_path (str): Đường dẫn đến ảnh cần xử lý
        output_folder (str): Thư mục lưu ảnh đã cắt, None nếu không lưu
        show_result (bool): Hiển thị kết quả xử lý
        
    Returns:
        numpy.ndarray: Ảnh cà chua đã cắt. Trả về None nếu không tìm thấy cà chua
    """
    # Đọc ảnh gốc
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Không thể đọc ảnh từ {image_path}")
        return None
    
    # Tạo bản sao để hiển thị
    display_image = original_image.copy()
    
    # Chuyển đổi sang ảnh thang độ xám
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    
    # Làm mờ ảnh để giảm nhiễu
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Phát hiện biên cạnh sử dụng Canny edge detector
    edges = cv2.Canny(blurred, 50, 150)
    
    # Thực hiện phép co giãn để kết nối các điểm biên
    kernel = np.ones((5, 5), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Tìm các đường viền (contours) từ hình ảnh biên
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Nếu không tìm thấy contour nào
    if not contours:
        print(f"Không tìm thấy cà chua trong ảnh {image_path}")
        return None
    
    # Lọc các contour theo độ tròn và tỉ lệ cạnh
    filtered_contours = []
    for contour in contours:
        # Bỏ qua các contour quá nhỏ
        if cv2.contourArea(contour) < 1000:
            continue
        
        # Tính độ tròn: 4*π*Diện tích/Chu vi^2
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        
        # Kiểm tra tỉ lệ cạnh (elliptical check)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h > 0 else 0
        
        # Hình cầu trong ảnh 2D có thể là hình tròn hoặc hình elip
        # Giảm ngưỡng độ tròn xuống 0.4 và chấp nhận elip có tỉ lệ cạnh từ 0.5 đến 2.0
        if (circularity > 0.4) and (0.5 <= aspect_ratio <= 2.0):
            filtered_contours.append((contour, circularity, aspect_ratio))
    
    if not filtered_contours:
        print(f"Không tìm thấy đối tượng hình cầu nào trong ảnh {image_path}")
        return None
    
    # Sắp xếp contours theo độ tròn và kích thước để ưu tiên những hình gần tròn nhất
    # Dùng điểm số = diện tích * độ tròn để ưu tiên contour lớn và tròn
    sorted_contours = sorted(filtered_contours, key=lambda x: cv2.contourArea(x[0]) * x[1], reverse=True)
    best_contour = sorted_contours[0][0]
    
    # Fit ellipse để xử lý cả hình cầu nhìn từ góc nghiêng
    if len(best_contour) >= 5:  # Cần ít nhất 5 điểm để fit ellipse
        ellipse = cv2.fitEllipse(best_contour)
        cv2.ellipse(display_image, ellipse, (0, 255, 0), 2)
        
        # Lấy thông số ellipse
        center, axes, angle = ellipse
        major_axis = max(axes)
        minor_axis = min(axes)
        
        # Tạo vùng cắt vuông quanh ellipse (tạo padding = 20% trục lớn)
        padding = int(major_axis * 0.2)
        x1 = max(0, int(center[0] - major_axis - padding))
        y1 = max(0, int(center[1] - major_axis - padding))
        x2 = min(original_image.shape[1], int(center[0] + major_axis + padding))
        y2 = min(original_image.shape[0], int(center[1] + major_axis + padding))
    else:
        # Fallback to bounding rect
        x, y, w, h = cv2.boundingRect(best_contour)
        padding = int(max(w, h) * 0.2)
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(original_image.shape[1], x + w + padding)
        y2 = min(original_image.shape[0], y + h + padding)
        
        # Vẽ hình chữ nhật
        cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Cắt ảnh
    cropped_image = original_image[y1:y2, x1:x2]
    
    # Kiểm tra nếu ảnh cắt không bị rỗng
    if cropped_image.size == 0:
        print(f"Không thể cắt ảnh từ {image_path}")
        return None
    
    if show_result:
        # Hiển thị quá trình và kết quả
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        plt.title('Ảnh gốc')
        plt.axis('off')
        
        plt.subplot(2, 2, 2)
        plt.imshow(edges, cmap='gray')
        plt.title('Phát hiện biên')
        plt.axis('off')
        
        plt.subplot(2, 2, 3)
        plt.imshow(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB))
        plt.title('Phát hiện cà chua hình cầu')
        plt.axis('off')
        
        plt.subplot(2, 2, 4)
        plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        plt.title('Cà chua đã cắt')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    # Lưu ảnh nếu cần
    if output_folder is not None:
        # Tạo thư mục nếu chưa tồn tại
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # Tạo tên file đầu ra
        filename = os.path.basename(image_path)
        output_path = os.path.join(output_folder, f"cropped_{filename}")
        
        # Lưu ảnh đã cắt
        cv2.imwrite(output_path, cropped_image)
        print(f"Đã lưu ảnh đã cắt tại {output_path}")
    
    return cropped_image


def process_directory(input_dir, output_dir, patterns=["*.jpg", "*.jpeg", "*.png"]):
    """
    Xử lý tất cả ảnh trong thư mục đầu vào và lưu kết quả vào thư mục đầu ra
    """
    # Tạo thư mục đầu ra nếu chưa tồn tại
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Tìm tất cả file ảnh phù hợp với pattern
    image_files = []
    for pattern in patterns:
        image_files.extend(glob(os.path.join(input_dir, pattern)))
    
    if not image_files:
        print(f"Không tìm thấy ảnh trong thư mục {input_dir}")
        return
    
    print(f"Tìm thấy {len(image_files)} ảnh để xử lý")
    
    # Xử lý từng ảnh
    success_count = 0
    for image_path in image_files:
        print(f"Đang xử lý {image_path}...")
        cropped = detect_and_crop_tomato(image_path, output_dir, show_result=False)
        if cropped is not None:
            success_count += 1
    
    print(f"Xử lý hoàn tất: {success_count}/{len(image_files)} ảnh đã được xử lý thành công")


# Phần chạy demo khi chạy trực tiếp file này
if __name__ == "__main__":
    # Xử lý một ảnh cụ thể
    image_path = "tomatoes-data/Ripe/r (428).jpg"  # Thay đổi đường dẫn này
    detect_and_crop_tomato(image_path, output_folder="cropped_tomatoes")
    
    # Hoặc xử lý tất cả ảnh trong một thư mục
    process_directory("GOOD", "cropped_tomatoes/Ripe")
    
    # Xử lý tất cả các loại cà chua
    # for class_name in ["Ripe", "Unripe", "Damaged"]:
    #     input_dir = f"tomatoes-data/{class_name}"
    #     output_dir = f"cropped_tomatoes/{class_name}"
    #     process_directory(input_dir, output_dir)
