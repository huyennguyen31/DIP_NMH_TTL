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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

#Tạo model CNN
class CNNModel(nn.Module):
    def __init__(self, img_size=128):
        super(CNNModel, self).__init__()

        # Các lớp Convolutional và Pooling gộp lại bằng nn.Sequential
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # Tính lại kích thước đầu vào cho fully connected layer sau pooling
        self.conv_output_size = img_size // 4  # Mỗi MaxPool giảm kích thước một nửa (2x2)
        self.fc1 = nn.Linear(64 * self.conv_output_size * self.conv_output_size, 128)
        self.fc2 = nn.Linear(128, 3)  # 3 loại: animal, plant, other

    def forward(self, x):
        # Xử lý qua các lớp Convolutional và Pooling
        x = self.conv_layers(x)
        
        # Flatten layer
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # Softmax có thể được áp dụng khi tính loss hoặc trong quá trình huấn luyện
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
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mean_hue = np.mean(hsv_image[:, :, 0])
    mean_saturation = np.mean(hsv_image[:, :, 1])
    mean_value = np.mean(hsv_image[:, :, 2])

    if (mean_hue >= 35 and mean_hue <= 85) and (mean_saturation > 50):
        return "plant"
    elif (mean_value < 50) and (mean_saturation < 50):
        return "other"
    else:
        return "animal"

# Data Augmentation
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ToTensor()
])

# Hàm tiền xử lý ảnh với phân loại theo màu
def preprocess_image(image, img_name, save_path=None):
    edges = apply_edge_detection(image)
    morphed_edges = apply_morphology(edges)
    thresholded = apply_threshold(morphed_edges)

    if len(thresholded.shape) == 2:
        thresholded = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2RGB)

    category = classify_based_on_color(thresholded)

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
                processed_img, img_category = preprocess_image(img_resized, img_name, save_path)
                data.append(processed_img)
                labels.append(categories.index(img_category))
            except Exception as e:
                print(f"Lỗi khi xử lý ảnh {img_name}: {e}")

    data = np.array(data)
    labels = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# Hàm vẽ biểu đồ hiệu năng
def plot_performance_metrics(train_losses, train_accuracies, test_losses, test_accuracies):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(15, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, test_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
    plt.plot(epochs, test_accuracies, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Hàm vẽ confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes,
                yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Hàm huấn luyện mô hình
def train_model(train_loader, test_loader, model, criterion, optimizer, device, num_epochs=100):
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
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
        
        train_loss = running_loss/len(train_loader)
        train_acc = 100 * correct/total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        y_true = []
        y_pred = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
        
        test_loss = running_loss/len(test_loader)
        test_acc = 100 * correct/total
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')
        print('-' * 60)
    
    return train_losses, train_accuracies, test_losses, test_accuracies, y_true, y_pred

# Hàm đánh giá mô hình
def evaluate_model(test_loader, model, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f'Accuracy on test data: {100 * correct / total:.2f}%')

if __name__ == "__main__":
    # Thiết lập các thông số
    IMG_SIZE = 128
    DATA_DIR = "D:\\Pokemon\\Pokemon\\dataset\\pokemon\\sorted"
    CATEGORIES = ["animal", "other", "plant"]
    BATCH_SIZE = 32
    SAVE_PATH = "D:\\Pokemon\\Pokemon\\processed_images"
    NUM_EPOCHS = 5
    
    # Tạo thư mục lưu ảnh đã xử lý nếu chưa tồn tại
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    # Chuẩn bị dữ liệu
    train_loader, test_loader = prepare_data(DATA_DIR, CATEGORIES, IMG_SIZE, BATCH_SIZE, SAVE_PATH)

    # Thiết lập device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Khởi tạo mô hình
    model = CNNModel().to(device)

    # Thiết lập loss function và optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Huấn luyện mô hình và thu thập metrics
    train_losses, train_accuracies, test_losses, test_accuracies, y_true, y_pred = train_model(
        train_loader, test_loader, model, criterion, optimizer, device, NUM_EPOCHS
    )

    # Vẽ biểu đồ hiệu năng
    plot_performance_metrics(train_losses, train_accuracies, test_losses, test_accuracies)

    # Vẽ confusion matrix
    plot_confusion_matrix(y_true, y_pred, CATEGORIES)

    # Lưu mô hình
    torch.save(model.state_dict(), os.path.join(SAVE_PATH, "cnn_model.pth"))
    print("Mô hình đã được lưu!")