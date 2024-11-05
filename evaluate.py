import os
import time
import random

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm

from quantize_utils import quantize_model

# Setting seed for all random initializations
SEED = 2
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

MODEL_PATH = "/home/namdng/garbage_classifier/models/lr_1e-3_bs_64_sche-f0.2-p6/ckpt_63_0.9641_.pth"
DATA_DIR = 'data_split'
DATA_SPLITS = ['calibration', 'val', 'test']

BATCH_SIZE = 32
IMAGE_SIZE = 394
QUANTIZE_MODEL = True

### Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

### Data transforms
data_transforms = {
    x: transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]) for x in DATA_SPLITS
}

### Data loaders
image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x])
                  for x in DATA_SPLITS}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
                                             shuffle=True, num_workers=8)
              for x in DATA_SPLITS}
dataset_sizes = {x: len(image_datasets[x]) for x in DATA_SPLITS}
class_names = image_datasets[DATA_SPLITS[0]].classes

### Load model
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
print("Load model from", MODEL_PATH)
model_conv.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
criterion = nn.CrossEntropyLoss()

### Evaluate
def evaluate_model(model, criterion):
    model.eval()
    since = time.time()
    for phase in DATA_SPLITS:
        model_conv.eval()
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in tqdm(dataloaders[phase]):
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        final_loss = running_loss / dataset_sizes[phase]
        final_acc = running_corrects.double() / dataset_sizes[phase]

        print(f"{phase} Loss: {final_loss:.4f} Acc: {final_acc:.4f}")

    time_elapsed = time.time() - since
    print(f"Eval complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")

if QUANTIZE_MODEL:
    device = "cpu"
    model_conv = model_conv.to(device)
    quantized_model = quantize_model(model_conv, dataloaders['calibration'], backend="fbgemm")
    evaluate_model(quantized_model, criterion)

    # enable this to save quantized model
    # model_dir_name, model_base_name = os.path.dirname(MODEL_PATH), os.path.basename(MODEL_PATH)
    # model_name = model_base_name.replace('.pth', '')
    # torch.jit.save(torch.jit.script(quantized_model), f"{model_dir_name}/{model_name}_quantized.pt")
else:
    evaluate_model(model_conv, criterion)