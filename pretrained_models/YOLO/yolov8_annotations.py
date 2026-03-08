import torch
from ultralytics import YOLO
from PIL import Image
import numpy as np
from pathlib import Path
import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the YOLOv8 model
model = YOLO('path_to_yolo_model')

bbox_dir = 'path_to_the_bounding_box_annotations'
annotated_dir = 'path_of_directory_to_store_labeled_images'
Path(bbox_dir).mkdir(parents=True, exist_ok=True)
Path(annotated_dir).mkdir(parents=True, exist_ok=True)

image_dir = 'path_of_image_directory'

def process_and_save_images(image_dir):
    for img_path in Path(image_dir).glob('*'):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            # Perform inference
            results = model(img_path, conf=0.25)

            # Process detections
            for result in results:
                boxes = result.boxes
                img = result.orig_img

                # Create an annotated image
                annotated_img = img.copy()
                for box in boxes:
                    # Get box coordinates, confidence, and class
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf.item()
                    cls = int(box.cls.item())
                    
                    # Draw box on image
                    cv2.rectangle(annotated_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    
                    # Add label
                    label = f'{model.names[cls]} {conf:.2f}'
                    cv2.putText(annotated_img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    # Save detection to file in the format you specified
                    with open(f'{bbox_dir}/{img_path.stem}.txt', 'a') as f:
                        f.write(f"{model.names[cls]} (tensor({x1:.0f}, device='cuda:0'), tensor({y1:.0f}, device='cuda:0'), tensor({x2:.0f}, device='cuda:0'), tensor({y2:.0f}, device='cuda:0'), {conf})\n")

                # Save annotated image
                cv2.imwrite(str(Path(annotated_dir) / img_path.name), annotated_img)

# Run the processing on the image directory
process_and_save_images(image_dir)