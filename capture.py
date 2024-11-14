import os
import time

import cv2
import torch
import onnxruntime as rt
from PIL import Image
from torchvision import transforms

import hyperparams as hparams

MODEL_PATH = "/home/namdng/Documents/linhtinh/garbage_classifier/models/resnet50_tuned_lr_1e-3_bs_64_sche-f0.2-p6/gc_torchscript.onnx"
class_names = ["cardboard_paper", "glass", "metal", "others", "plastic"]
device = "cpu"

# Set up the ONNX session with options
sess_opt = rt.SessionOptions()
sess_opt.intra_op_num_threads = 4
ort_session = rt.InferenceSession(MODEL_PATH, sess_opt, providers=["CPUExecutionProvider"])

### Data transforms
data_transforms = transforms.Compose([
    transforms.Resize((hparams.IMAGE_SIZE, hparams.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Function to preprocess the image and run inference
def classify_image(ort_session, img):
    img = data_transforms(img)
    img = img.unsqueeze(0)  # Add batch dimension
    img = img.to(device)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img)}
    ort_outs = ort_session.run(None, ort_inputs)
    outputs = torch.tensor(ort_outs[0])
    _, preds = torch.max(outputs, 1)
    return class_names[preds[0]]

### Camera
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
FRAME_WIDTH = 640
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
FRAME_HEIGHT = 480
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

print("Press 'space' to capture an image, 'q' to quit.")

while True:
    # Read the video frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # draw a rectangle to crop the image
    feed_frame = frame.copy()
    cv2.rectangle(feed_frame, ((FRAME_WIDTH-FRAME_HEIGHT)//2, 0), ((FRAME_WIDTH-FRAME_HEIGHT)//2 + FRAME_HEIGHT, FRAME_HEIGHT), (0, 255, 0), 1)

    # Display the frame
    cv2.imshow("Video Feed", feed_frame)

    # Wait for key press
    key = cv2.waitKey(1) & 0xFF

    # Capture the frame when spacebar is pressed
    if key == ord(' '):
        # crop the frame
        frame = frame[:, (FRAME_WIDTH-FRAME_HEIGHT)//2:(FRAME_WIDTH-FRAME_HEIGHT)//2 + FRAME_HEIGHT]

        # Save the captured frame as an image file
        captured_image_path = "capture.jpg"
        cv2.imwrite(captured_image_path, frame)

        # Load the image and classify it
        img = Image.open(captured_image_path)
        start_time = time.time()
        pred_class = classify_image(ort_session, img)
        inference_time = time.time() - start_time

        # Display classification result
        print(f"Predicted class: {pred_class}")
        print(f"Inference time: {inference_time:.4f} seconds")

        # Show the captured frame with prediction
        cv2.putText(frame, f"{pred_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow("Captured Image", frame)

    # Exit if 'q' is pressed
    elif key == ord('q'):
        break

# release resources
cap.release()
cv2.destroyAllWindows()



