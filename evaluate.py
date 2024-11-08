import os
import time
import random

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm

from quantize_utils import static_quantize_model
import hyperparams as hparams
from model import load_model

# Setting seed for all random initializations
SEED = 2
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

MODEL_PATH = "/home/namdng/garbage_classifier/models/resnet50_tuned_lr_1e-3_bs_64_sche-f0.2-p6/quantized_ckpt_57_0.9597.pth"
DATA_DIR = 'data_split'
DATA_SPLITS = ['calibration', 'val', 'test']

BATCH_SIZE = 64

### Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

### Data transforms
data_transforms = {
    x: transforms.Compose([
        transforms.Resize((hparams.IMAGE_SIZE, hparams.IMAGE_SIZE)),
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
print("Loading model: ", MODEL_PATH)
model_conv = load_model(hparams.BACKBONE, hparams.NUM_IMMEDIATE_FEATURES, len(class_names), hparams.DROPOUT_RATE)
if "quantize" in MODEL_PATH:
    device = "cpu"
    model_conv = static_quantize_model(model_conv, dataloader=None, backend="fbgemm")

model_conv.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
model_conv = model_conv.to(device)
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

evaluate_model(model_conv, criterion)