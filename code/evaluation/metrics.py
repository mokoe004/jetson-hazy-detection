import os
import numpy as np
from collections import defaultdict

# Function to parse a single annotation file
def parse_annotation_file(file_path):
    detections = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split()
            class_label = parts[0]
            # Clean and try converting the confidence string to float
            try:
                confidence = parts[-1].strip(')').strip('tensor(').strip(',').strip('device=\'cuda:0\'')
                confidence = float(confidence)
                if np.isnan(confidence):  # Replace nan with 0.0 or another default value
                    confidence = 0.0  # Adjust this default value as needed
            except ValueError:
                confidence = 0.0  # Assign a default value in case of conversion failure
            detections.append((class_label, confidence))
    return detections




# Function to calculate average precision for one class
def calculate_average_precision(confidences, threshold=0.5):
    sorted_confidences = sorted(confidences, reverse=True)
    if not sorted_confidences:
        return 0.0  # Return 0.0 AP if there are no confidences to evaluate

    tp = np.array([1 if conf >= threshold else 0 for conf in sorted_confidences])
    fp = np.array([1 if conf < threshold else 0 for conf in sorted_confidences])
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    if tp_cumsum[-1] == 0:  # No true positives
        return 0.0

    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    recalls = tp_cumsum / tp_cumsum[-1]
    # Integration over the recall to get AP
    return np.trapz(precisions, recalls)


# Main function to calculate mAP across all files in a directory
def calculate_map(directory):
    class_detections = defaultdict(list)
    
    # Read each file and parse detections
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):  # Adjust the extension based on your files
            file_path = os.path.join(directory, filename)
            detections = parse_annotation_file(file_path)
            for class_label, confidence in detections:
                class_detections[class_label].append(confidence)

    # Calculate AP for each class and then the mean AP
    aps = []
    for cls, confidences in class_detections.items():
        ap = calculate_average_precision(confidences)
        aps.append(ap)
        print(f"Class: {cls}, AP: {ap}")
    
    mean_ap = np.nanmean(aps) if aps else 0
    print(f"Mean Average Precision (mAP): {mean_ap}")
    return mean_ap

# Directory containing your annotation files
directory = 'pat_to_annotation_directory'
calculate_map(directory)
