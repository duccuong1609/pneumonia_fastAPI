from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import os
import torch
from PIL import Image
import io
from app.model.grad_cam import predict_and_visualize_gradcam
from app.model.predict import predict
from fastapi.middleware.cors import CORSMiddleware
from app.model.grad_cam import ViTBinaryClassifier
from app.model.predict import CNNBinaryClassifier
from huggingface_hub import hf_hub_download

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Cho phép tất cả domain
    allow_credentials=True,    # Nếu không cần cookie có thể đặt False
    allow_methods=["*"],       # Cho phép tất cả method: GET, POST, PUT, DELETE...
    allow_headers=["*"],       # Cho phép tất cả headers
)

load_dotenv()

# Bỏ image_path cũ, chỉ giữ model path thôi
cache_dir = "/tmp/hf_models"
model_path = hf_hub_download(repo_id=os.getenv("REPO_IT_PATH"), filename=os.getenv("MODEL_PATH"), cache_dir=cache_dir)
model_vit_path = hf_hub_download(repo_id=os.getenv("REPO_IT_PATH"), filename=os.getenv("MODEL_VIT"), cache_dir=cache_dir)

print(model_path)

device = "cuda" if torch.cuda.is_available() else "cpu"

model_vit = ViTBinaryClassifier()
checkpoint = torch.load(model_vit_path, map_location=device)
model_vit.load_state_dict(checkpoint)
model_vit.to(device)
model_vit.eval()
print(f"Loaded model from {model_vit_path}")

model = CNNBinaryClassifier(model_type='resnet18', pretrained=False).to(device)

@app.on_event("startup")
def warmup_model():
    import numpy as np
    # Tạo ảnh giả để warm up
    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
    predict(
            image=dummy_img,
            model_path=model_path,
            model=model,
            device=device
        )
    predict_and_visualize_gradcam(
            image=dummy_img,
            model=model_vit,
            device=device
        )
    print("Model warmup done!")

@app.get('/')
def home():
    return {'msg': 'Welcome in my api'}

@app.post("/predict")
async def run_prediction(file: UploadFile = File(...)):
    try:
        # Đọc file ảnh upload về PIL Image
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Gọi hàm predict với img truyền vào thay vì image_path
        # Bạn cần chỉnh hàm predict nhận PIL Image hoặc tensor (nếu chưa, mình có thể giúp)
        result = predict(
            image=img,
            model_path=model_path,
            model=model,
            device=device
        )
        return {"result": result}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/predict_gradcam")
async def predict_and_visualize(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        
        result = predict_and_visualize_gradcam(
            image=img,
            model=model_vit,
            device=device
        )
        return {"result": result}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})