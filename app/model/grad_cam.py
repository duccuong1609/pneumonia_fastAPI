# Mã trước đó (setup_dataloaders, XRayDataset, train_model, evaluate_model, 
# ModelCheckpoint, EarlyStopping, ReduceLROnPlateau) giữ nguyên từ artifact version ID cf94bf8d-9389-4e6f-bbf8-4f3a830de4df.
# Thêm hàm predict_and_visualize_gradcam để dự đoán và vẽ Grad-CAM cho một ảnh.

import os
import torch
import torch.nn as nn
from transformers import ViTForImageClassification
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

# Cấu hình môi trường Kaggle
os.environ["TORCH_DYNAMO_DISABLE"] = "1"

# 6. Xây dựng mô hình ViT
class ViTBinaryClassifier(nn.Module):
    def __init__(self):
        super(ViTBinaryClassifier, self).__init__()
        self.vit = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
            num_labels=1,
            ignore_mismatched_sizes=True,
            output_hidden_states=True,
        )
        # Đóng băng các tầng ViT
        for param in self.vit.vit.parameters():
            param.requires_grad = False

        # Mở khóa 6 tầng cuối
        num_layers = len(self.vit.vit.encoder.layer)
        for layer in self.vit.vit.encoder.layer[-6:]:
            for param in layer.parameters():
                param.requires_grad = True
        
        # Mở khóa CLS token
        self.vit.vit.embeddings.cls_token.requires_grad = True
    
    def forward(self, pixel_values, output_attentions=False):
        outputs = self.vit(pixel_values=pixel_values, output_hidden_states=True, output_attentions=output_attentions)
        logits = outputs.logits
        return torch.sigmoid(logits), logits  # Trả về sigmoid và logit thô

# Hàm vẽ Grad-CAM (giả định từ pipeline)
def visualize_gradcam(image, heatmap, title="Grad-CAM", alpha=0.6):
    """
    Vẽ ảnh gốc, heatmap Grad-CAM, và ảnh chồng lấn.
    
    Parameters:
    - image: Tensor ảnh đầu vào [3, H, W]
    - heatmap: Mảng numpy [H, W] (ví dụ: [14, 14])
    - title: Tiêu đề biểu đồ
    - alpha: Độ trong suốt của heatmap khi chồng lấn
    """
    try:
        # Chuyển tensor ảnh về numpy và chuẩn hóa
        image = image.cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]
        image = (image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1)
        
        # Resize heatmap về kích thước ảnh
        heatmap = np.uint8(255 * heatmap)  # Chuẩn hóa về [0, 255]
        heatmap = Image.fromarray(heatmap).resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)
        heatmap = np.array(heatmap) / 255.0
        
        # Tạo heatmap màu
        heatmap_color = plt.cm.jet(heatmap)[:, :, :3]  # [H, W, 3]
        
        # Ảnh chồng lấn
        superimposed_img = heatmap_color * alpha + image * (1 - alpha)
        superimposed_img = superimposed_img.clip(0, 1)
        
        # Vẽ ba biểu đồ
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title("Original Image")
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(heatmap, cmap='jet')
        plt.title("Grad-CAM Heatmap")
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(superimposed_img)
        plt.title("Superimposed")
        plt.axis('off')
        
        plt.suptitle(title)
        plt.show()
    
    except Exception as e:
        print(f"Error visualizing Grad-CAM: {e}")
        plt.figure()
        plt.imshow(image)
        plt.title("Original Image (Visualization Failed)")
        plt.axis('off')
        plt.show()

# Hàm tính Grad-CAM
def get_gradcam_heatmap(model, image, class_idx=0, device='cuda'):
    """
    Tính heatmap Grad-CAM cho mô hình ViT bằng gradient CLS hoặc trọng số attention.
    
    Parameters:
    - model: Mô hình ViTBinaryClassifier đã huấn luyện
    - image: Tensor hình ảnh đầu vào [3, H, W]
    - class_idx: Chỉ số lớp để tính gradient (mặc định: 0 cho phân loại nhị phân)
    - device: Thiết bị tính toán ('cuda' hoặc 'cpu')
    
    Returns:
    - numpy array: Heatmap Grad-CAM (ví dụ: [14, 14]), hoặc mảng zero nếu thất bại
    """
    try:
        model.eval()
        image = image.unsqueeze(0).to(device)  # Shape: [1, 3, H, W]
        if not image.requires_grad:
            image.requires_grad_(True)
        
        print(f"Input image shape: {image.shape}")
        
        sigmoid_outputs, raw_logits = model(pixel_values=image, output_attentions=True)
        
        print(f"Sigmoid outputs shape: {sigmoid_outputs.shape}, Sigmoid value: {sigmoid_outputs}")
        print(f"Raw logits shape: {raw_logits.shape}, Raw logits value: {raw_logits}")
        
        outputs = model.vit(pixel_values=image, output_hidden_states=True, output_attentions=True)
        if not hasattr(outputs, 'hidden_states') or outputs.hidden_states is None:
            raise ValueError("Không có hidden_states. Đảm bảo output_hidden_states=True trong config ViT.")
        if not hasattr(outputs, 'attentions') or outputs.attentions is None:
            raise ValueError("Không có attentions. Đảm bảo output_attentions=True trong forward pass.")
        
        last_layer_output = outputs.hidden_states[-1]  # Shape: [1, 197, 768]
        last_attention = outputs.attentions[-1]  # Shape: [1, num_heads, 197, 197]
        print(f"Last layer output shape: {last_layer_output.shape}")
        print(f"Last attention shape: {last_attention.shape}")
        
        target_logit = raw_logits[:, 0]  # Shape: [1]
        print(f"Target logit (pre-sigmoid): {target_logit}")
        
        cls_output = last_layer_output[:, 0, :]  # Shape: [1, 768]
        cls_output.retain_grad()
        
        model.zero_grad()
        target_logit.backward(retain_graph=True)
        
        if cls_output.grad is None:
            print("Warning: Không có gradient cho CLS output. Kiểm tra phương án thay thế.")
            if last_layer_output.grad is None:
                print("Warning: Không có gradient cho last layer output. Sử dụng trọng số attention làm fallback.")
                attention = last_attention.mean(dim=1).detach()  # Shape: [1, 197, 197]
                heatmap = attention[:, 0, 1:]  # CLS-to-patches: [1, 196]
                heatmap = torch.relu(heatmap)
            else:
                cls_grad = last_layer_output.grad[:, 0, :].detach()  # Shape: [1, 768]
                print(f"CLS gradient (từ last layer) shape: {cls_grad.shape}, min/max: {cls_grad.min()}, {cls_grad.max()}")
                attention = last_attention.mean(dim=1).detach()
                heatmap = attention[:, 0, 1:] * cls_grad.sum(dim=-1, keepdim=True)
                heatmap = torch.relu(heatmap)
        else:
            cls_grad = cls_output.grad.detach()  # Shape: [1, 768]
            print(f"CLS gradient shape: {cls_grad.shape}, min/max: {cls_grad.min()}, {cls_grad.max()}")
            attention = last_attention.mean(dim=1).detach()
            heatmap = attention[:, 0, 1:] * cls_grad.sum(dim=-1, keepdim=True)
            heatmap = torch.relu(heatmap)
        
        print(f"Raw heatmap shape: {heatmap.shape}, Raw heatmap min/max: {heatmap.min()}, {heatmap.max()}")
        
        num_patches = heatmap.size(1)
        patch_size = int(num_patches ** 0.5)  # ví dụ: 14 cho 196 patch (224/16)
        heatmap = heatmap.view(1, patch_size, patch_size)  # Shape: [1, 14, 14]
        
        heatmap_max = torch.max(heatmap)
        if heatmap_max > 0:
            heatmap = heatmap / (heatmap_max + 1e-8)
        else:
            print("Warning: Heatmap toàn số 0. Trả về heatmap zero.")
            return np.zeros((patch_size, patch_size))
        
        heatmap = heatmap.squeeze().detach().cpu().numpy()  # Shape: [14, 14]
        print(f"Final heatmap shape: {heatmap.shape}, Final heatmap min/max: {heatmap.min()}, {heatmap.max()}")
        
        return heatmap
    
    except Exception as e:
        print(f"Error computing Grad-CAM: {e}")
        patch_size = 14
        print(f"Trả về heatmap zero có shape [{patch_size}, {patch_size}] do lỗi.")
        return np.zeros((patch_size, patch_size))
    
import io
import base64
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def visualize_gradcam_combined_base64(image, heatmap, alpha=0.6):
    """
    Tạo ảnh tổng hợp gồm: original, heatmap màu, superimposed. Trả về dạng base64 PNG.
    
    Parameters:
    - image: Tensor ảnh đầu vào [3, H, W]
    - heatmap: Mảng numpy [H, W]
    - alpha: Độ trong suốt của heatmap khi chồng lấn
    
    Returns:
    - base64 string của ảnh tổng hợp (original + heatmap + superimposed)
    """
    try:
        # Chuyển tensor ảnh về numpy và chuẩn hóa
        image_np = image.cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]
        image_np = (image_np * np.array([0.229, 0.224, 0.225]) + 
                    np.array([0.485, 0.456, 0.406])).clip(0, 1)

        # Resize heatmap
        heatmap_resized = np.uint8(255 * heatmap)
        heatmap_resized = Image.fromarray(heatmap_resized).resize(
            (image_np.shape[1], image_np.shape[0]), resample=Image.BILINEAR
        )
        heatmap_arr = np.array(heatmap_resized) / 255.0

        # Tạo heatmap màu
        heatmap_color = plt.cm.jet(heatmap_arr)[:, :, :3]

        # Tạo ảnh chồng lấn
        superimposed_img = heatmap_color * alpha + image_np * (1 - alpha)
        superimposed_img = (superimposed_img * 255).astype(np.uint8)

        # Ảnh gốc và heatmap cũng chuyển sang uint8
        original_img = (image_np * 255).astype(np.uint8)
        heatmap_color_img = (heatmap_color * 255).astype(np.uint8)

        # Ghép 3 ảnh nằm ngang
        combined_img = np.concatenate([original_img, heatmap_color_img, superimposed_img], axis=1)
        combined_pil = Image.fromarray(combined_img)

        # Encode sang base64
        buf = io.BytesIO()
        combined_pil.save(buf, format="PNG")
        base64_img = base64.b64encode(buf.getvalue()).decode('utf-8')

        return base64_img

    except Exception as e:
        print(f"Error generating combined Grad-CAM base64 image: {e}")
        return None

# Hàm dự đoán và vẽ Grad-CAM
def predict_and_visualize_gradcam(image, model, device='cuda', class_names=['NORMAL', 'PNEUMONIA']):
    """
    Dự đoán nhãn và vẽ Grad-CAM cho một ảnh X-quang bất kỳ sử dụng mô hình ViT đã huấn luyện.
    
    Parameters:
    - image_path (str): Đường dẫn đến ảnh đầu vào (jpg, png, v.v.).
    - model_path (str): Đường dẫn đến file checkpoint của mô hình đã huấn luyện.
    - device (str): Thiết bị tính toán ('cuda' hoặc 'cpu').
    - class_names (list): Danh sách tên lớp (mặc định: ['NORMAL', 'PNEUMONIA']).
    
    Returns:
    - dict: Kết quả dự đoán gồm nhãn, xác suất, và thông báo.
    """
    try:
        # Khởi tạo thiết bị
        device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load mô hình
        # model = ViTBinaryClassifier()
        # checkpoint = torch.load(model_path, map_location=device)
        # model.load_state_dict(checkpoint)
        # model.to(device)
        # model.eval()
        # print(f"Loaded model from {model_path}")
        
        # Định nghĩa biến đổi ảnh (giống trong XRayDataset)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Load và xử lý ảnh
        # image = Image.open(image_path).convert('RGB')  # Chuyển sang RGB nếu là grayscale
        image_tensor = transform(image)  # Shape: [3, 224, 224]
        image_tensor_input = image_tensor.unsqueeze(0).to(device)  # Shape: [1, 3, 224, 224]
        
        # Dự đoán
        with torch.no_grad():
            sigmoid_outputs, _ = model(image_tensor_input)
            probability = sigmoid_outputs.item()  # Xác suất PNEUMONIA
            predicted = 1 if probability > 0.5 else 0
            predicted_label = class_names[predicted]
        
        # Kết quả dự đoán
        result = {
            'label': predicted_label,
            'probability': probability,
            'message': f"Predicted: {predicted_label} with probability {probability:.4f}"
        }
        print(result['message'])
        
        # Tính và vẽ Grad-CAM
        heatmap = get_gradcam_heatmap(model, image_tensor, class_idx=predicted, device=device)
        base64_img = visualize_gradcam_combined_base64(image_tensor, heatmap)
        # visualize_gradcam(image_tensor, heatmap, title=f"Grad-CAM for {predicted_label}")

        result = {
            'label': predicted_label,
            'probability': probability,
            'message': f"Predicted: {predicted_label} with probability {probability:.4f}",
            'gradcam_image': base64_img
        }
        
        return result
    
    except Exception as e:
        print(f"Error processing image or Grad-CAM: {e}")
        return {
            'label': None,
            'probability': None,
            'message': f"Error: {str(e)}"
        }

# Mã chạy pipeline (ví dụ):

# ... (setup_dataloaders, XRayDataset, train_model, evaluate_model giữ nguyên)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# list_path = ["D:/CV/model/gr1.jpg"]
# # Dự đoán và vẽ Grad-CAM cho một ảnh
# for path in list_path:
#     image_path = path  # Thay bằng đường dẫn thực tế
#     result = predict_and_visualize_gradcam(image_path, model_path='D:/CV/model/pneumonia_cnn_resnet_model.pth', device=device)
#     print(result)