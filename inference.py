import os
import time

import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

MODEL_PATH = "/home/namdng/garbage_classfier/models/init/ckpt_26_0.9547.pth"
IMAGE_SIZE = 394
BENCHMARK = True

### Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

### Data transforms
data_transforms = {
    'test': transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

### Load model
class_names = ["cardboard_paper", "glass", "metal", "others", "plastic"]
model_conv = torchvision.models.efficientnet_v2_s()
num_ftrs = model_conv.classifier[1].in_features
model_conv.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(num_ftrs, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, len(class_names))
)
model_conv = model_conv.to(device)
model_conv.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
model_conv.eval()

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
