#!/usr/bin/env python3
"""
Test script to run YOLO model on an answer sheet image and display the results.
"""
from ultralytics import YOLO
import json
import sys

# Load the model
model_path = "model_yolo26n/best.pt"
image_path = "/Users/donhuvy/Downloads/20250218_122310982_iOS.jpg"

print(f"Loading model from: {model_path}")
model = YOLO(model_path)

# Print model info
print(f"\nModel class names: {model.names}")
print(f"Number of classes: {len(model.names)}")

# Run inference
print(f"\nRunning inference on: {image_path}")
results = model(image_path, conf=0.25, verbose=True)

# Parse results
print("\n" + "="*80)
print("DETECTION RESULTS")
print("="*80)

for r in results:
    boxes = r.boxes
    print(f"\nTotal detections: {len(boxes)}")
    
    if len(boxes) == 0:
        print("No detections found!")
        continue
    
    for i, box in enumerate(boxes):
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()
        
        print(f"\n  Detection #{i+1}:")
        print(f"    Class: {cls_name} (id={cls_id})")
        print(f"    Confidence: {conf:.4f}")
        print(f"    Box [x1,y1,x2,y2]: [{xyxy[0]:.1f}, {xyxy[1]:.1f}, {xyxy[2]:.1f}, {xyxy[3]:.1f}]")

# Save annotated image
output_path = "/Users/donhuvy/Downloads/20250218_122310982_iOS_detected.jpg"
for r in results:
    annotated = r.plot()
    import cv2
    cv2.imwrite(output_path, annotated)
    print(f"\nAnnotated image saved to: {output_path}")

print("\nDone!")
