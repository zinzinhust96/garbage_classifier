import os
import time

import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from quantize_utils import static_quantize_model
import hyperparams as hparams
from model import load_model

MODEL_PATH = "/home/namdng/garbage_classifier/models/resnet50_tuned_lr_1e-3_bs_64_sche-f0.2-p6/quantized_ckpt_57_0.9597.pth"
BENCHMARK = True

torch.set_num_threads(1)

### Device
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"

### Data transforms
data_transforms = {
    'test': transforms.Compose([
        transforms.Resize((hparams.IMAGE_SIZE, hparams.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

### Load model
class_names = ["cardboard_paper", "glass", "metal", "others", "plastic"]
model_conv = load_model(hparams.BACKBONE, hparams.NUM_IMMEDIATE_FEATURES, len(class_names), hparams.DROPOUT_RATE)
if "quantize" in MODEL_PATH:
    device = "cpu"
    model_conv = static_quantize_model(model_conv, dataloader=None, backend="fbgemm")

print("Loading model: ", MODEL_PATH)
model_conv.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
model_conv = model_conv.to(device)

### Inference
def visualize_model_predictions(model,img_path):
    model.eval()
    since = time.time()

    img = Image.open(img_path)
    img = data_transforms['test'](img)
    img = img.unsqueeze(0)
    img = img.to(device)

    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)
    
    return class_names[preds[0]], time.time() - since


TEST_DIR = 'data_split/test'
test_images_paths = []
for class_name in os.listdir(TEST_DIR):
    class_dir = os.path.join(TEST_DIR, class_name)
    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        test_images_paths.append(img_path)
test_images_paths = test_images_paths[:200]
print(f"Total test images: {len(test_images_paths)}")

if BENCHMARK:
    time_array = []
    for img_path in tqdm(test_images_paths):
        pred, time_elapsed = visualize_model_predictions(model_conv, img_path)
        time_array.append(time_elapsed)

    print(f"Average inference time: {sum(time_array) / len(time_array)}")
else:
    # pred, time_elapsed = visualize_model_predictions(model_conv, os.path.join(TEST_DIR, test_images_name[0]))
    pred, time_elapsed = visualize_model_predictions(model_conv, "data_split/test/others/trash518_data1.jpg")
    print(f"Predicted class: {pred}")
    print(f"Inference time: {time_elapsed}")

# Latency for GPU: 0.014438848256210272
