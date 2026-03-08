import torch
from PIL import Image
import numpy as np
from pathlib import Path

# Assuming YOLOv5 is installed and importable in the same directory
from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import non_max_suppression, scale_boxes
from utils.plots import Annotator, colors

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = DetectMultiBackend('path_to_yolo_model', device=device)
model.eval()

#directory to store bounding box annotations
bbox_dir = 'path_to_the_bounding_box_annotations'

#directory to store annotated images
annotated_dir = 'path_of_directory_to_store_labeled_images'
Path(bbox_dir).mkdir(parents=True, exist_ok=True)
Path(annotated_dir).mkdir(parents=True, exist_ok=True)

#image directory for which you want to create BB and annotated images
image_dir = 'path_of_image_directory'

def process_and_save_images(image_dir):
    dataset = LoadImages(image_dir, img_size=640)
    for path, img, im0s, vid_cap, s in dataset:
        img = torch.from_numpy(img).to(device).float()
        img /= 255.0  # normalize images
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        # Inference
        pred = model(img, augment=False, visualize=False)
        pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)

        for i, det in enumerate(pred):  # detections per image
            p, im0 = Path(path), im0s.copy()

            annotator = Annotator(im0, line_width=3, example=str(colors))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results to file
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = f'{model.names[c]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(c, True))
                    with open(f'{bbox_dir}/{p.stem}.txt', 'a') as f:
                        f.write(f'{model.names[c]} {(*xyxy, conf.item())}\n')  # save to file

            # Save annotated image
            annotated_path = str(Path(annotated_dir) / p.name)
            annotator.save(annotated_path)

# Run the processing on the dehazed images directory
process_and_save_images(image_dir)