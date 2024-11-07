import os
import time
import random

import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import torchvision
from torchvision import datasets, transforms
from dotenv import load_dotenv

from early_stop import EarlyStopper

load_dotenv()

# Setting seed for all random initializations
SEED = 2
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

###
EXPERIMENT_NAME = "GC"
MODEL_DIR = 'models/'
DATA_DIR = 'data_split'

### hyperparameters
BACKBONE = 'resnet50'
IMAGE_SIZE = 224 if BACKBONE == 'resnet50' else 394
NUM_EPOCHS = 20
NUM_IMMEDIATE_FEATURES = 128
DROPOUT_RATE = 0.2

### Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

### init mlflow tracking
mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
mlflow.set_experiment(EXPERIMENT_NAME)

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

### Datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes


def keep_k_best_checkpoints_by_score(model_dir, checkpoint_save_total_limit):
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

        # remove old checkpoints based on score
        if len(old_checkpoints) > checkpoint_save_total_limit:
            old_checkpoints = sorted(old_checkpoints, key=lambda x: x['score'])
            print(f"Deleting old checkpoints: {old_checkpoints[0]['path']}")
            os.remove(old_checkpoints[0]['path'])


def train_model(model, dataloaders, criterion, optimizer, scheduler, model_path, num_epochs=25, early_stopper=None):
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

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            current_lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
            print(f'{phase} Epoch {epoch}/{num_epochs - 1}. Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} LR: {current_lr}')
            mlflow.log_metric(f'{phase}_loss', epoch_loss, step=epoch)
            mlflow.log_metric(f'{phase}_acc', epoch_acc, step=epoch)
            if phase == 'train':
                mlflow.log_metric(f'running_lr', current_lr, step=epoch)

            if phase == 'val' and scheduler:
                print(f'[SCHEDULER] step with loss: {epoch_loss:.4f}. Best loss {scheduler.best}. Is better: {scheduler.is_better(epoch_loss, scheduler.best)}')
                scheduler.step(epoch_loss)
                print(f'[SCHEDULER] Num bad epochs: {scheduler.num_bad_epochs}')

            # deep copy the model
            if phase == 'val':
                ckpt_path = os.path.join(model_path, f"ckpt_{epoch}_{epoch_acc:.4f}.pth")
                print(f'Saving best model to {ckpt_path}')
                torch.save(model.state_dict(), ckpt_path)

                # Keep only the best k checkpoints based on validation score
                keep_k_best_checkpoints_by_score(model_path, 3)

                # early stopping based on validation loss
                if early_stopper and early_stopper.early_stop(epoch_loss):
                    print(f'Early stopping at epoch {epoch}')
                    return model

        time_elapsed = time.time() - since
        print(f'Epoch complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    return model


def run_an_experiment(learning_rate, batch_size, run_name):
    os.makedirs(os.path.join(MODEL_DIR, run_name), exist_ok=True)
    print(f"\n\n ++++++++++ Training with exp name: {run_name}")

    ### Dataloader for the current batch size
    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                        shuffle=True, num_workers=8)
        for x in ['train', 'val']
    }

    ### Init models
    def setup_classifier_head(num_ftrs, len_class_names):
        return nn.Sequential(
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(num_ftrs, NUM_IMMEDIATE_FEATURES),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(NUM_IMMEDIATE_FEATURES, len_class_names)
        )

    if BACKBONE == "efficientnet_v2s":
        model_conv = torchvision.models.efficientnet_v2_s(weights='IMAGENET1K_V1')
        for param in model_conv.parameters():
            param.requires_grad = False
        num_ftrs = model_conv.classifier[1].in_features
        model_conv.classifier = setup_classifier_head(num_ftrs, len(class_names))
    elif BACKBONE == "resnet50":
        model_conv = torchvision.models.resnet50(weights='IMAGENET1K_V1')
        for param in model_conv.parameters():
            param.requires_grad = False
        num_ftrs = model_conv.fc.in_features
        model_conv.fc = setup_classifier_head(num_ftrs, len(class_names))
        
    model_conv = model_conv.to(device)

    ### Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_conv = optim.Adam(model_conv.parameters(), lr=learning_rate)

    ### Learning rate scheduler based on validation loss
    # exp_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_conv, mode='min', factor=0.2, patience=6)
    exp_lr_scheduler = None

    ### early stopping based on validation loss
    # early_stopper = EarlyStopper(patience=10)
    early_stopper = None

    ### start to train ###
    ### mlflow
    mlflow.start_run(run_name=run_name)

    ### log hyperparameters
    hyperparameters = {
        'batch_size': batch_size,
        'image_size': IMAGE_SIZE,
        'learning_rate': learning_rate,
        'num_epochs': NUM_EPOCHS,
        'num_immediate_features': NUM_IMMEDIATE_FEATURES,
        'dropout_rate': DROPOUT_RATE
    }
    mlflow.log_params(hyperparameters)

    ### Train
    model_conv = train_model(model_conv, dataloaders, criterion, optimizer_conv, exp_lr_scheduler,
                            model_path=os.path.join(MODEL_DIR, run_name), 
                            num_epochs=NUM_EPOCHS, early_stopper=early_stopper)

    ### end mlflow
    mlflow.end_run()


### Hyperparameter tuning
learning_rates = [3e-3, 1e-3, 3e-4]
batch_sizes = [16, 32, 64, 128]

for lr in learning_rates:
    for bs in batch_sizes:
        run_name = f"resnet50_param_tuning_lr_{lr}_bs_{bs}"
        run_an_experiment(lr, bs, run_name)

### Normal run
# LEARNING_RATE = 1e-3
# BATCH_SIZE = 32
# RUN_NAME = "resnet50_lr_1e-3_bs_32_imsize224"
# run_an_experiment(LEARNING_RATE, BATCH_SIZE, RUN_NAME)