import torch
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