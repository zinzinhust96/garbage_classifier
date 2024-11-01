import os
import time

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from tqdm import tqdm

MODEL_PATH = "/home/namdng/garbage_classfier/models/init/ckpt_10_0.9331.pth"
DATA_DIR = 'data_split'
DATA_SPLITS = ['val', 'test']

BATCH_SIZE = 32
IMAGE_SIZE = 394

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
model_conv.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
model_conv.eval()
criterion = nn.CrossEntropyLoss()

### Evaluate
def evaluate_model(model, criterion):
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