import os
import time

import torch
import torch.nn as nn
import torchvision
import onnxruntime as rt
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from quantize_utils import static_quantize_model
import hyperparams as hparams
from model import load_model

# set num threads to 1 (NOTE: disable this when on Raspberry Pi)
sess_opt = rt.SessionOptions()
sess_opt.intra_op_num_threads = 1

MODEL_PATH = "/home/namdng/garbage_classifier/models/resnet50_tuned_lr_1e-3_bs_64_sche-f0.2-p6/gc_torchscript.onnx"
BENCHMARK = True

### Device
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"

### Data transforms
data_transforms = {
    'test': transforms.Compose([
        transforms.Resize((hparams.IMAGE_SIZE, hparams.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

### Load model
class_names = ["cardboard_paper", "glass", "metal", "others", "plastic"]
ort_session = rt.InferenceSession(MODEL_PATH, sess_opt, providers=["CPUExecutionProvider"])


### Inference
def visualize_model_predictions(ort_session,img_path):
    since = time.time()

    img = Image.open(img_path)
    img = data_transforms['test'](img)
    img = img.unsqueeze(0)
    img = img.to(device)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img)}
    ort_outs = ort_session.run(None, ort_inputs)
    outputs = torch.tensor(ort_outs[0])
    _, preds = torch.max(outputs, 1)
    
    return class_names[preds[0]], time.time() - since


TEST_DIR = 'data_split/test'
test_images_paths = []
for class_name in os.listdir(TEST_DIR):
    class_dir = os.path.join(TEST_DIR, class_name)
    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        test_images_paths.append(img_path)

test_images_paths = test_images_paths[:200]
print(f"Total test images: {len(test_images_paths)}")

if BENCHMARK:
    time_array = []
    for img_path in tqdm(test_images_paths):
        pred, time_elapsed = visualize_model_predictions(ort_session, img_path)
        time_array.append(time_elapsed)

    print(f"Average inference time: {sum(time_array) / len(time_array)}")
else:
    # pred, time_elapsed = visualize_model_predictions(model_conv, os.path.join(TEST_DIR, test_images_name[0]))
    pred, time_elapsed = visualize_model_predictions(ort_session, "data_split/test/others/battery_27_data2.jpg")
    print(f"Predicted class: {pred}")
    print(f"Inference time: {time_elapsed}")

# Latency for CPU (i5-12500, 1 CPU core):
# efficientnet_v2s:
#   0.2265708318320654 (torch model)
#   0.17627952814102174 (onnx opset 10)
#   0.1762681257724762 (onnx opset 20)

# resnet50:
#   0.08464848399162292 (torch model)
#   0.043368810415267946 (quantized torch model)
#   0.0657833194732666 (onnx opset 10)
