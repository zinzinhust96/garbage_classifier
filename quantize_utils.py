import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm

### Define quantized model
class QuantizedModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model_features = model.features
        self.model_avgpool = model.avgpool
        self.model_classifier = model.classifier
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        
    def forward(self, x):
        x = self.quant(x)
        x = self.model_features(x)
        x = self.dequant(x)
        x = self.model_avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model_classifier(x)
        return x
    
def quantize_model(model_conv, dataloader, backend="fbgemm"):
    """ Quantize a model using PyTorch's post-training static quantization.

    Args:
        model_conv: The model to be quantized.
        dataloader (torch.utils.data.DataLoader): The dataloader for calibration.
        backend (str): Pytorch backend for quantization. Can be one of fbgemm' for server, 'qnnpack' for mobile. Default is 'fbgemm'.
    """
    quantized_model = QuantizedModel(model_conv)

    ### Post-Training Static Quantization ###
    model_conv.qconfig = torch.quantization.get_default_qconfig(backend)
    # Set the backend on which the quantized kernels need to be run
    torch.backends.quantized.engine=backend

    torch.quantization.prepare(quantized_model, inplace=True)

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
    torch.quantization.convert(quantized_model, inplace=True)
    ### End of Post-Training Static Quantization ###

    return quantized_model

if __name__ == "__main__":
    MODEL_PATH = "/home/namdng/garbage_classifier/models/lr_1e-3_bs_64_sche-f0.2-p6/ckpt_63_0.9641_.pth"
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
