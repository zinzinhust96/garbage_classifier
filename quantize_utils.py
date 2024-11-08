import os
import time
import copy
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from tqdm import tqdm
from torch.ao.quantization import QuantStub, DeQuantStub

from model import load_model
import hyperparams as hparams

class QuantizedEfficientNet(nn.Module):
    def __init__(self, original_model):
        super(QuantizedEfficientNet, self).__init__()
        # Copy the original EfficientNet components
        self.features = original_model.features
        self.avgpool = original_model.avgpool
        self.classifier = original_model.classifier

        # Add QuantStub and DeQuantStub
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        # Modify the features module to include quant and dequant stubs
        self.features = nn.Sequential(
            self.quant,  # Insert QuantStub at the beginning
            *self.features,
            self.dequant  # Insert DeQuantStub at the end
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def static_quantize_model(model_conv, dataloader, backend="fbgemm"):
    """ Quantize a model using PyTorch's post-training static quantization.

    Args:
        model_conv: The model to be quantized.
        dataloader (torch.utils.data.DataLoader): The dataloader for calibration.
        backend (str): Pytorch backend for quantization. Can be one of fbgemm' for server, 'qnnpack' for mobile. Default is 'fbgemm'.
    """
    print(str(type(model_conv)))
    if "EfficientNet" in str(model_conv):
        quantized_model = QuantizedEfficientNet(model_conv)
    elif "ResNet" in str(model_conv):
        quantized_model = copy.deepcopy(model_conv)
        # quantized_model.fuse_model()
        # quantized_model = QuantizedResNet(model_conv)

    quantized_model = quantized_model.to("cpu")
        
    ### Post-Training Static Quantization ###
    quantized_model.qconfig = torch.ao.quantization.get_default_qconfig(backend)
    # Set the backend on which the quantized kernels need to be run
    torch.backends.quantized.engine=backend

    torch.ao.quantization.prepare(quantized_model, inplace=True)

    # Calibration
    def calibrate(model, sample_loader):
        model.eval()
        with torch.no_grad():
            for inputs, _ in tqdm(sample_loader):
                inputs = inputs.to("cpu")
                model(inputs)

    # Run calibration
    if dataloader:
        print("[Quantization] Calibrating model...")
        calibrate(quantized_model, dataloader)

    # Convert to quantized model
    torch.ao.quantization.convert(quantized_model, inplace=True)
    ### End of Post-Training Static Quantization ###

    return quantized_model

def dynamic_quantize_model(model_conv):
    raise NotImplementedError("Dynamic quantization is not supported yet.")

if __name__ == "__main__":
    MODEL_PATH = "/home/namdng/garbage_classifier/models/resnet50_tuned_lr_1e-3_bs_64_sche-f0.2-p6/ckpt_57_0.9597.pth"
    
    ### Load model
    print("Loading model: ", MODEL_PATH)
    model_conv = load_model(hparams.BACKBONE, hparams.NUM_IMMEDIATE_FEATURES, 5, hparams.DROPOUT_RATE)
    model_conv.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))

    ### Quantize model
    DATA_DIR = 'data_split/val'
    data_transform = transforms.Compose([
        transforms.Resize((hparams.IMAGE_SIZE, hparams.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_dataset = datasets.ImageFolder(DATA_DIR, data_transform)
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=1)
    quantized_model = static_quantize_model(model_conv, dataloader, backend="fbgemm")
    # print(quantized_model)

    # test to see if quantized model can be run
    start_time = time.time()
    quantized_model.eval()
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to("cpu")
            # print(inputs.shape)
            outputs = quantized_model(inputs)
            # print(outputs)
    print("Inference time quantized: ", time.time() - start_time)

    ### Save quantized model
    model_dir = os.path.dirname(MODEL_PATH)
    quantized_model_path = os.path.join(model_dir, "quantized_" + os.path.basename(MODEL_PATH))
    torch.save(quantized_model.state_dict(), quantized_model_path)

