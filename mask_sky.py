import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

# --------------------------- 配置 ---------------------------
model_path = ""

input_folder = ""

output_folder = os.path.join(os.path.dirname(input_folder), "masks_sky")
os.makedirs(output_folder, exist_ok=True)

# --------------------------- 设备 ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------- 加载模型 ---------------------------
feature_extractor = SegformerFeatureExtractor.from_pretrained(model_path)

model = SegformerForSemanticSegmentation.from_pretrained(model_path)
model.to(device)
model.eval()

# --------------------------- 读取图片 ---------------------------
image_files = sorted([
    f for f in os.listdir(input_folder)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
])

# --------------------------- 批处理 ---------------------------
for img_name in tqdm(image_files, desc="Processing images"):

    img_path = os.path.join(input_folder, img_name)
    image = Image.open(img_path).convert("RGB")

    orig_size = image.size  # (W, H)

    # ------------------- preprocess -------------------
    inputs = feature_extractor(
        images=image,
        return_tensors="pt"
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    # ------------------- forward -------------------
    with torch.no_grad():
        outputs = model(**inputs)

    # ------------------- CPU上处理 logits（关键） -------------------
    logits = outputs.logits.cpu()

    logits_up = F.interpolate(
        logits,
        size=orig_size[::-1],
        mode="bilinear",
        align_corners=False
    )

    # ------------------- soft probability（关键提升质量） -------------------
    prob = torch.softmax(logits_up, dim=1)[0]

    # ------------------- vegetation = grass + tree -------------------
    vegetation = prob[4] + prob[9]

    # 阈值可以调（0.2~0.5之间效果不同）
    vegetation_mask = (vegetation > 0.3).numpy().astype(np.uint8) * 255

    # ------------------- save -------------------
    mask_name = os.path.splitext(img_name)[0] + ".png"
    mask_path = os.path.join(output_folder, mask_name)

    Image.fromarray(vegetation_mask).save(mask_path)

print(f"完成！vegetation mask 已保存到: {output_folder}")