import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm

from torch.ao.quantization import QuantStub, DeQuantStub

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
    quantized_model = QuantizedEfficientNet(model_conv)
        
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
    print("[Quantization] Calibrating model...")
    calibrate(quantized_model, dataloader)

    # Convert to quantized model
    torch.ao.quantization.convert(quantized_model, inplace=True)
    ### End of Post-Training Static Quantization ###

    return quantized_model

def dynamic_quantize_model(model_conv):
    raise NotImplementedError("Dynamic quantization is not supported yet.")