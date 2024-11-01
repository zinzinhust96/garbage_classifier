import os
import time
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torchvision
from torchvision import datasets, transforms

###
EXPERIMENT_NAME = 'garbage_classifier'
MODEL_DIR = 'models/'
DATA_DIR = 'data_split'

### hyperparameters
BATCH_SIZE = 128
IMAGE_SIZE = 394
LEARNING_RATE = 0.001
NUM_EPOCHS = 100

### Data transforms
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop((IMAGE_SIZE, IMAGE_SIZE), scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

### Data loaders
image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
                                             shuffle=True, num_workers=8)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

### Init models
model_conv = torchvision.models.efficientnet_v2_s(weights='IMAGENET1K_V1')

# freeze all layers
for param in model_conv.parameters():
    param.requires_grad = False

num_ftrs = model_conv.classifier[1].in_features
model_conv.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(num_ftrs, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, len(class_names))
)
model_conv = model_conv.to(device)

### Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer_conv = optim.Adam(model_conv.parameters(), lr=LEARNING_RATE)
exp_lr_scheduler = None


def keep_k_best_checkpoints(model_dir, checkpoint_save_total_limit):
    # Delete old checkpoints
    if checkpoint_save_total_limit is not None and checkpoint_save_total_limit > 0:
        old_checkpoints = []
        for subdir in os.listdir(model_dir):
            if subdir.endswith(".pth"):
                subdir_step, subdir_score = subdir.replace(".pth", "").split("_")[1:]
                old_checkpoints.append({
                    'step': int(subdir_step),
                    'score': float(subdir_score),
                    'path': os.path.join(model_dir, subdir),
                })

        if len(old_checkpoints) > checkpoint_save_total_limit:
            old_checkpoints = sorted(old_checkpoints, key=lambda x: x['score'])
            print(f"Deleting old checkpoints: {old_checkpoints[0]['path']}")
            shutil.rmtree(old_checkpoints[0]['path'])


def train_model(model, criterion, optimizer, scheduler, model_path, num_epochs=25):
    os.makedirs(model_path, exist_ok=True)

    for epoch in range(num_epochs):
        since = time.time()
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                if scheduler:
                    scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Epoch {epoch}/{num_epochs - 1}. Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val':
                ckpt_path = os.path.join(model_path, f"ckpt_{epoch}_{epoch_acc:.4f}.pth")
                print(f'Saving best model to {ckpt_path}')
                torch.save(model.state_dict(), ckpt_path)

                # Keep only the best k checkpoints
                keep_k_best_checkpoints(model_path, 5)

        time_elapsed = time.time() - since
        print(f'Epoch complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

RUN_NAME = 'init'
model_conv = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler,
                         model_path=os.path.join(MODEL_DIR, RUN_NAME), num_epochs=NUM_EPOCHS)