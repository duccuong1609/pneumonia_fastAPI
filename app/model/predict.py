import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet18, resnet50, densenet121, vgg16
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score
import glob

import torch
import torch.nn as nn
from torchvision.models import resnet18, densenet121

class CNNBinaryClassifier(nn.Module):
    def __init__(self, model_type='resnet18', pretrained=True):
        super(CNNBinaryClassifier, self).__init__()
        self.model_type = model_type.lower()
        self.pretrained = pretrained
        self.features = None

        # Khởi tạo mô hình dựa trên model_type
        if self.model_type == 'resnet18':
            self.cnn = resnet18(pretrained=pretrained)
            num_ftrs = self.cnn.fc.in_features
            self.cnn.fc = nn.Linear(num_ftrs, 1)
            self.grad_layer = 'layer4'
        elif self.model_type == 'densenet121':
            self.cnn = densenet121(pretrained=pretrained)
            num_ftrs = self.cnn.classifier.in_features
            self.cnn.classifier = nn.Linear(num_ftrs, 1)
            self.grad_layer = 'norm5' 
        elif self.model_type == 'custom':
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(512, 1)
            )
            self.grad_layer = 'conv4'  
        else:
            raise ValueError(f"model_type '{model_type}' không được hỗ trợ. Chọn từ: resnet18, densenet121, custom")

        if pretrained and self.model_type != 'custom':
            for param in self.cnn.parameters():
                param.requires_grad = False
            # Mở khóa tầng cuối và layer Grad-CAM
            if self.model_type == 'resnet18':
                for param in self.cnn.layer4.parameters():
                    param.requires_grad = True
                for param in self.cnn.fc.parameters():
                    param.requires_grad = True
            elif self.model_type == 'densenet121':
                for param in self.cnn.features.norm5.parameters():
                    param.requires_grad = True
                for param in self.cnn.classifier.parameters():
                    param.requires_grad = True
        elif self.model_type == 'custom':
            for param in self.cnn.parameters():
                param.requires_grad = True

        self.register_hooks()

    def register_hooks(self):
        def forward_hook(module, input, output):
            if output is not None and isinstance(output, torch.Tensor):
                self.features = output
            else:
                print(f"Cảnh báo: Output của layer {self.grad_layer} không hợp lệ: {output}")

        # Đăng ký hook cho layer cụ thể
        if self.model_type in ['resnet18', 'resnet50']:
            self.cnn.layer4.register_forward_hook(forward_hook)
            for param in self.cnn.layer4.parameters():
                param.requires_grad = True
        elif self.model_type == 'densenet121':
            self.cnn.features.norm5.register_forward_hook(forward_hook) 
            for param in self.cnn.features.norm5.parameters():
                param.requires_grad = True
        elif self.model_type == 'vgg16':
            self.cnn.features[-1].register_forward_hook(forward_hook) 
            for param in self.cnn.features[-1].parameters():
                param.requires_grad = True
        elif self.model_type == 'custom':
            self.cnn[-4].register_forward_hook(forward_hook)
            for param in self.cnn[-4].parameters():
                param.requires_grad = True
        else:
            raise ValueError(f"Không thể đăng ký hook cho model_type '{self.model_type}'")

    def forward(self, x):
        logits = self.cnn(x)
        return torch.sigmoid(logits), logits
    
def inspect_and_fix_state_dict(model, checkpoint_path, device='cpu'):
    """
    Kiểm tra và sửa state_dict của checkpoint để tải vào mô hình.
    
    Args:
        model: Mô hình PyTorch.
        checkpoint_path: Đường dẫn đến file checkpoint.
        device: Thiết bị để tải checkpoint.
    
    Returns:
        state_dict đã được sửa (hoặc nguyên bản nếu không cần sửa).
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    
    print("Các khóa trong checkpoint:")
    for key in list(state_dict.keys())[:10]:  # In 10 khóa đầu tiên
        print(key)
    print("Các khóa trong mô hình:")
    for key in list(model.state_dict().keys())[:10]:
        print(key)
    
    # Loại bỏ tiền tố 'module.' nếu có
    if any(key.startswith('module.') for key in state_dict.keys()):
        print("Loại bỏ tiền tố 'module.' từ state_dict")
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # Ánh xạ các khóa nếu cần
    key_mapping = {
        'features.conv0.weight': 'cnn.features.conv0.weight',
        'features.norm0.weight': 'cnn.features.norm0.weight',
        'features.norm0.bias': 'cnn.features.norm0.bias',
        'features.norm0.running_mean': 'cnn.features.norm0.running_mean',
        'features.norm0.running_var': 'cnn.features.norm0.running_var',
        'classifier.weight': 'cnn.classifier.weight',
        'classifier.bias': 'cnn.classifier.bias',
    }
    
    model_dict = model.state_dict()
    new_state_dict = {key_mapping.get(k, k): v for k, v in state_dict.items() if key_mapping.get(k, k) in model_dict}
    model_dict.update(new_state_dict)
    
    return model_dict

# 10. Hàm tính Grad-CAM
def get_gradcam_heatmap(model, image, class_idx=0, device='cuda'):
    try:
        model.eval()
        image = image.unsqueeze(0).to(device)
        if not image.requires_grad:
            image.requires_grad_(True)
        
        print(f"Kích thước ảnh đầu vào: {image.shape}")
        
        # Forward pass
        sigmoid_outputs, logits = model(image)
        print(f"Kích thước sigmoid outputs: {sigmoid_outputs.shape}, Giá trị: {sigmoid_outputs}")
        print(f"Kích thước logits: {logits.shape}, Giá trị: {logits}")
        
        if model.features is None:
            raise ValueError("Feature maps không được ghi lại. Kiểm tra forward hook.")
        
        features = model.features
        print(f"Kích thước feature maps: {features.shape}")
        
        model.zero_grad()
        target_logit = logits[:, class_idx]
        target_logit.backward()
        
        # Kiểm tra gradient của feature maps
        gradients = features.grad
        if gradients is None:
            print("Lỗi: Gradient của feature maps là None. Kiểm tra layer và requires_grad.")
            return np.zeros((7, 7))
        
        print(f"Kích thước gradients: {gradients.shape}, Min/max: {gradients.min()}, {gradients.max()}")
        
        # Tính trọng số
        weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
        print(f"Kích thước weights: {weights.shape}, Min/max: {weights.min()}, {weights.max()}")
        
        # Tính heatmap
        heatmap = torch.sum(weights * features, dim=1).squeeze()
        heatmap = torch.relu(heatmap)
        print(f"Kích thước heatmap trước chuẩn hóa: {heatmap.shape}, Min/max: {heatmap.min()}, {heatmap.max()}")
        
        # Chuẩn hóa heatmap
        heatmap_max = torch.max(heatmap)
        if heatmap_max > 0:
            heatmap = heatmap / (heatmap_max + 1e-8)
        else:
            print("Lỗi: Heatmap toàn số 0. Kiểm tra feature maps và gradients.")
            return np.zeros((7, 7))
        
        heatmap = heatmap.detach().cpu().numpy()
        print(f"Kích thước heatmap cuối: {heatmap.shape}, Min/max: {heatmap.min()}, {heatmap.max()}")
        
        return heatmap
    
    except Exception as e:
        print(f"Lỗi khi tính Grad-CAM: {e}")
        return np.zeros((7, 7))

# 11. Hàm vẽ Grad-CAM
def visualize_gradcam(image, heatmap, title="Grad-CAM", alpha=0.6):
    try:
        image = image.cpu().numpy().transpose(1, 2, 0)
        image = (image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1)
        
        heatmap = np.uint8(255 * heatmap)
        heatmap = Image.fromarray(heatmap).resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)
        heatmap = np.array(heatmap) / 255.0
        
        heatmap_color = plt.cm.jet(heatmap)[:, :, :3]
        superimposed_img = heatmap_color * alpha + image * (1 - alpha)
        superimposed_img = superimposed_img.clip(0, 1)
        
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title("Ảnh gốc")
        plt.axis('off')
        plt.subplot(1, 3, 2)
        plt.imshow(heatmap, cmap='jet')
        plt.title("Heatmap Grad-CAM")
        plt.axis('off')
        plt.subplot(1, 3, 3)
        plt.imshow(superimposed_img)
        plt.title("Ảnh chồng lấn")
        plt.axis('off')
        plt.suptitle(title)
        plt.show()
    
    except Exception as e:
        print(f"Lỗi khi vẽ Grad-CAM: {e}")
        plt.figure()
        plt.imshow(image)
        plt.title("Ảnh gốc (Vẽ thất bại)")
        plt.axis('off')
        plt.show()

def predict(image, model , model_path, device='cuda', class_names=['NORMAL', 'PNEUMONIA']):
    try:
        device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Khởi tạo mô hình
        
        
        # === PHẦN CẦN SỬA ===
        # Sử dụng hàm phụ để kiểm tra và sửa state_dict
        fixed_state_dict = inspect_and_fix_state_dict(model, model_path, device)
        model.load_state_dict(fixed_state_dict, strict=False)
        # === KẾT THÚC PHẦN CẦN SỬA ===
        
        model.eval()
        # print(f"Đã tải mô hình từ {model_path}")
        
        # Chuẩn bị ảnh đầu vào
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image)
        image_tensor_input = image_tensor.unsqueeze(0).to(device)
        
        # Dự đoán
        with torch.no_grad():
            sigmoid_outputs, _ = model(image_tensor_input)
            probability = sigmoid_outputs.item()
            predicted = 1 if probability > 0.5 else 0
            predicted_label = class_names[predicted]
        
        result = {
            'label': predicted_label,
            'probability': probability,
            'message': f"Dự đoán: {predicted_label} với xác suất {probability:.4f}"
        }
        print(result['message'])
        
        # Tính và vẽ Grad-CAM
        # heatmap = get_gradcam_heatmap(model, image_tensor, class_idx=predicted, device=device)
        # visualize_gradcam(image_tensor, heatmap, title=f"Grad-CAM cho {predicted_label}")
        
        return result
    
    except Exception as e:
        print(f"Lỗi khi xử lý ảnh hoặc Grad-CAM: {e}")
        return {
            'label': None,
            'probability': None,
            'message': f"Lỗi: {str(e)}"
        }
    
# Thiết lập thiết bị
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# os.environ["TORCH_DYNAMO_DISABLE"] = "1"

# image_path = "D:/CV/model/gr1.jpg"
# result = predict_and_visualize_gradcam(image_path, model_path='D:/CV/model/pneumonia_cnn_resnet_model.pth', device=device)
# print(result)