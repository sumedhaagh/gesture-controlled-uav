import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision import transforms
from PIL import Image, ImageOps
import os
import cv2
import numpy as np

MODEL_PATH = r"C:\Users\Jinx\Desktop\PROJECT(MAIN)\final_best_vit (1).pth"
IMG_SIZE = 224

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

MEAN = [0.474015, 0.538135, 0.587003]
STD  = [0.201344, 0.201758, 0.194989]

CLASS_NAMES = ['Roll_up', 'Throttle_up']
NUM_CLASSES = len(CLASS_NAMES)

print("Classes:", CLASS_NAMES)


class ResizeWithAspectRatio:
    def __init__(self, size):
        self.size = size
    #condtructor of transformer object/runs once its created 
    def __call__(self, img):
        #w - width h - height
        w, h = img.size
        scale = min(self.size / w, self.size / h)
        nw, nh = int(w * scale), int(h * scale)
        img = img.resize((nw, nh), Image.BICUBIC)
        pad_w = self.size - nw
        pad_h = self.size - nh
        return ImageOps.expand(
            img,
            (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2),
            fill=0
        )
#this transform is critical for vit as it will be consistestent with space
#perform patch embeddings properly 
#will give irregular resizing to better train



transform = transforms.Compose([
    ResizeWithAspectRatio(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# -------- MODEL --------
weights = ViT_B_16_Weights.IMAGENET1K_V1
model = vit_b_16(weights=weights)
model.heads.head = nn.Linear(model.heads.head.in_features, NUM_CLASSES)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint)

model.to(DEVICE)
model.eval()


# -------- SINGLE IMAGE --------
def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(img)
        probs = torch.softmax(logits, dim=1)

    conf, pred = torch.max(probs, 1)
    return CLASS_NAMES[pred.item()], conf.item()


# -------- FOLDER --------
def predict_folder(folder_path):
    for file in os.listdir(folder_path):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            path = os.path.join(folder_path, file)
            cls, conf = predict_image(path)
            print(f"{file:30s} → {cls:15s} ({conf:.4f})")


# -------- WEBCAM --------
#this will diaply video feedback
#this will continously capture webcam frames and convert to models fornat and also run infernce
def webcam_inference(camera_id=0):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("No webcam")
        return

    print("Press q to quit")

    while True:
        ret, frame = cap.read() #ret means if frame was read successfully 
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)#opencv accpets a bgr but pil will have rgb and this is pil image object
        img_t = transform(img).unsqueeze(0).to(DEVICE)#this is to resize normalize and make tensors

        with torch.no_grad(): #this will disable model learning and enable real prediction
            logits = model(img_t)#forward pass patch embedding etc
            probs = torch.softmax(logits, dim=1) #1 and 0 propbablity of correct gesture
            conf, pred = torch.max(probs, 1)

        label = CLASS_NAMES[pred.item()]
        confidence = conf.item()

        cv2.putText(frame, f"{label} ({confidence:.2f})",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.1, (0, 255, 0), 2)

        cv2.imshow("ViT Prediction", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    webcam_inference()
