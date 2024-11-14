import os
import time

import cv2
import torch
import onnxruntime as rt
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

import hyperparams as hparams

# utils
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# run this on Raspberry Pi
sess_opt = rt.SessionOptions()
sess_opt.intra_op_num_threads = 4

INPUT_SAVE_PATH = "./input"
os.makedirs(INPUT_SAVE_PATH, exist_ok=True)

MODEL_PATH = "/home/namdng/Documents/linhtinh/garbage_classifier/models/resnet50_tuned_lr_1e-3_bs_64_sche-f0.2-p6/gc_torchscript.onnx"
class_names = ["cardboard_paper", "glass", "metal", "others", "plastic"]
ort_session = rt.InferenceSession(MODEL_PATH, sess_opt, providers=["CPUExecutionProvider"])
device = "cpu"

### Camera
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
FRAME_WIDTH = 320
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
FRAME_HEIGHT = 240
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 30)

### Data transforms
data_transforms = {
    'test': transforms.Compose([
        transforms.Resize((hparams.IMAGE_SIZE, hparams.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

count = 0

# Display video feed and capture on Space key press
with torch.no_grad():
    while True:
        # read frame
        ret, image = cap.read()
        if not ret:
            print("Failed to read frame.")
            break

        # draw a rectangle to crop the image
        feed_frame = image.copy()
        cv2.rectangle(feed_frame, ((FRAME_WIDTH-FRAME_HEIGHT)//2, 0), ((FRAME_WIDTH-FRAME_HEIGHT)//2 + FRAME_HEIGHT, FRAME_HEIGHT), (0, 255, 0), 1)

        # display the video feed
        cv2.imshow("Camera Feed", feed_frame)

        # check for key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # Space key to capture and process the image
            # convert opencv output from BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            input_image = Image.fromarray(image_rgb)

            # crop the image
            input_image = input_image.crop(((FRAME_WIDTH-FRAME_HEIGHT)//2, 0, (FRAME_WIDTH-FRAME_HEIGHT)//2 + FRAME_HEIGHT, FRAME_HEIGHT))

            # TEMP: save the image
            input_image.save(os.path.join(INPUT_SAVE_PATH, f'capture_{count}.jpg'))
            count += 1

            # preprocess
            input_tensor = data_transforms['test'](input_image)
            
            # create a mini-batch as expected by the model
            input_batch = input_tensor.unsqueeze(0).to(device)

            # run model
            ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input_batch)}
            ort_outs = ort_session.run(None, ort_inputs)
            outputs = torch.tensor(ort_outs[0])
            _, preds = torch.max(outputs, 1)

            print(f"Prediction: {class_names[preds[0]]}")

        elif key == ord('q'):  # Press 'q' to exit
            print("Exiting...")
            break

# release resources
cap.release()
cv2.destroyAllWindows()



