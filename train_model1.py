#DIP
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
from torchvision import transforms
from PIL import Image
import cv2
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

# Tạo mô hình CNN
class CNNModel(nn.Module):
    def __init__(self, img_size=128):
        super(CNNModel, self).__init__()
        # Các lớp Convolutional
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Tính kích thước đầu vào cho fc1
        conv_output_size = img_size // 2 // 2  # Mỗi MaxPool giảm kích thước một nửa
        self.fc1 = nn.Linear(64 * conv_output_size * conv_output_size, 128)
        self.fc2 = nn.Linear(128, 3)  # 3 loại: animal, plant, other

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * (x.size(2) * x.size(3)))  # Flatten layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Các bước DIP chính
def apply_edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    return edges

def apply_morphology(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morphed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return morphed

def apply_threshold(image):
    _, thresholded = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    return thresholded

# Phân loại theo màu sắc chủ đạo
def classify_based_on_color(image):
    # Chuyển đổi ảnh sang không gian màu HSV để dễ phân biệt màu sắc
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Tính toán trung bình giá trị H, S, V
    mean_hue = np.mean(hsv_image[:, :, 0])  # Hue channel
    mean_saturation = np.mean(hsv_image[:, :, 1])  # Saturation channel
    mean_value = np.mean(hsv_image[:, :, 2])  # Value channel

    # Quy tắc phân loại màu
    if (mean_hue >= 35 and mean_hue <= 85) and (mean_saturation > 50):  # Màu xanh lá
        return "plant"
    elif (mean_value < 50) and (mean_saturation < 50):  # Màu xám, đen
        return "other"
    else:  # Các màu còn lại
        return "animal"

# Data Augmentation
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ToTensor()
])

# Hàm tiền xử lý ảnh với phân loại theo màu
def preprocess_image(image, img_name, save_path=None):
    # Áp dụng các bước DIP
    edges = apply_edge_detection(image)
    morphed_edges = apply_morphology(edges)
    thresholded = apply_threshold(morphed_edges)

    # Chuyển đổi grayscale sang RGB
    if len(thresholded.shape) == 2:
        thresholded = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2RGB)

    # Phân loại dựa trên màu sắc chủ đạo
    category = classify_based_on_color(thresholded)

    # Nếu cần lưu ảnh đã xử lý, có thể lưu tại đây
    if save_path:
        save_img_path = os.path.join(save_path, img_name)
        cv2.imwrite(save_img_path, thresholded)

    return transforms.ToTensor()(Image.fromarray(thresholded)), category

# Hàm chuẩn bị dữ liệu
def prepare_data(data_dir, categories, img_size, batch_size, save_path=None):
    data = []
    labels = []

    for category in categories:
        folder_path = os.path.join(data_dir, category)
        label = categories.index(category)

        for img_name in os.listdir(folder_path):
            try:
                img_path = os.path.join(folder_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                
                img_resized = cv2.resize(img, (img_size, img_size))
                
                # Lấy dữ liệu đã xử lý từ hàm tiền xử lý
                processed_img, img_category = preprocess_image(img_resized, img_name, save_path)
                
                # Gán nhãn cho dữ liệu
                data.append(processed_img)
                labels.append(categories.index(img_category))  # Thêm nhãn cho phân loại theo màu
            except Exception as e:
                print(f"Lỗi khi xử lý ảnh {img_name}: {e}")

    data = np.array(data)
    labels = np.array(labels)

    # Chia tập train/test
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# Hàm huấn luyện mô hình
def train_model(train_loader, model, criterion, optimizer, device, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100*correct/total:.2f}%')

# Đánh giá mô hình
def evaluate_model_with_metrics(test_loader, model, device, class_names):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    precision = precision_score(all_labels, all_preds, average='weighted', labels=range(len(class_names)))
    recall = recall_score(all_labels, all_preds, average='weighted', labels=range(len(class_names)))
    f1 = f1_score(all_labels, all_preds, average='weighted', labels=range(len(class_names)))
    print(f'Accuracy on test data: {100 * correct / total:.2f}%')
    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, labels=range(len(class_names))))

# Tải mô hình và nhãn
def load_model_and_labels(model_path, label_path):
    model = CNNModel()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    with open(label_path, "r") as f:
        labels = f.read().splitlines()

    return model, labels

# Tiền xử lý ảnh cho dự đoán
def preprocess_for_prediction(image, img_size):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return transform(image).unsqueeze(0)

# Phát hiện và phân loại Pokémon
def detect_pokemon(image_path, model, labels, img_size=128):
    image = cv2.imread(image_path)
    if image is None:
        print("Không thể đọc ảnh.")
        return

    # Tiền xử lý ảnh
    input_tensor = preprocess_for_prediction(image, img_size)

    # Chạy mô hình
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        class_idx = predicted.item()
        class_label = labels[class_idx]

    # Hiển thị kết quả
    h, w, _ = image.shape
    cv2.rectangle(image, (10, 10), (w - 10, h - 10), (0, 255, 0), 2)
    cv2.putText(
        image,
        class_label,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.imshow("Pokemon Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Chạy chương trình
if __name__ == "__main__":
    IMG_SIZE = 128
    DATA_DIR = "D:\\Pokemon\\Pokemon\\dataset\\pokemon\\sorted"
    CATEGORIES = ["animal", "other", "plant"]
    BATCH_SIZE = 32
    SAVE_PATH = "D:\\Pokemon\\Pokemon\\processed_images"
    
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    # Chuẩn bị dữ liệu
    train_loader, test_loader = prepare_data(DATA_DIR, CATEGORIES, IMG_SIZE, BATCH_SIZE, SAVE_PATH)

    # Khởi tạo mô hình
    model = CNNModel().to('cuda' if torch.cuda.is_available() else 'cpu')

    # Tiêu chuẩn và optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Huấn luyện mô hình
    train_model(train_loader, model, criterion, optimizer, 'cuda' if torch.cuda.is_available() else 'cpu', num_epochs=10)

    # Đánh giá mô hình
    evaluate_model_with_metrics(test_loader, model, 'cuda' if torch.cuda.is_available() else 'cpu', CATEGORIES)
    # Lưu mô hình
    torch.save(model.state_dict(), os.path.join(SAVE_PATH, "cnn_model.pth"))
    print("Mô hình đã được lưu!")

    # Dự đoán cho ảnh mẫu
    #MODEL_PATH = "D:/Pokemon/Pokemon/adjusted_model.pth"
    #LABEL_PATH = "D:/Pokemon/Pokemon/labels.txt"
    #IMAGE_PATH = "D:/Pokemon/Pokemon/sample2.png"
    #model, labels = load_model_and_labels(MODEL_PATH, LABEL_PATH)
    #detect_pokemon(IMAGE_PATH, model, labels)