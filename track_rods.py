import cv2
from ultralytics import YOLO
import numpy as np
import os

def calculate_iou(box1, box2):
    # box: [x1, y1, x2, y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def ensemble_predict(frame, models, conf=0.5, iou_threshold=0.4):
    all_boxes = []
    all_scores = []
    
    for model in models:
        results = model.predict(frame, conf=conf, iou=iou_threshold, agnostic_nms=True, verbose=False)
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()
            all_boxes.append(boxes)
            all_scores.append(scores)
    
    if not all_boxes:
        return []
    
    # Combine all predictions
    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    
    # Apply Non-Maximum Suppression (NMS) using OpenCV
    # NMSBoxes expects boxes as [x, y, w, h]
    cv_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box
        cv_boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
    
    indices = cv2.dnn.NMSBoxes(cv_boxes, scores, conf, iou_threshold)
    
    final_boxes = []
    if len(indices) > 0:
        for i in indices.flatten():
            final_boxes.append(boxes[i].astype(int))
            
    return final_boxes

def process_source(source_path, models):
    # Check if source is image or video
    ext = os.path.splitext(source_path)[1].lower()
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']

    if ext in image_extensions:
        # Process Image
        print(f"Processing image: {source_path} using ensemble models")
        frame = cv2.imread(source_path)
        if frame is None:
            print(f"Error: Could not read image {source_path}")
            return

        # Use ensemble predict
        boxes = ensemble_predict(frame, models, conf=0.5)
        count = len(boxes)
        
        for box in boxes:
            # Draw bounding box (Green color BGR: 0, 255, 0)
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)

        # Display Total Count
        cv2.putText(
            frame,
            f"Total Rods: {count}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3,
        )
        
        cv2.imshow("Steel Rod Detection (Ensemble)", frame)
        print(f"Finished. Total rods detected: {count}")
        print("Press any key in the image window to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif ext in video_extensions:
        # Process Video with Manual Tracking and Speed Optimization
        cap = cv2.VideoCapture(source_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {source_path}")
            return

        print(f"Processing video: {source_path} using ensemble models... Press 'q' to stop.")

        # Speed Optimization Settings
        frame_skip = 3  # Process every 3rd frame
        target_height = 480 # Resize height for inference to speed up
        
        # Tracking state
        next_id = 0
        active_tracks = {} # id -> last_box (in original scale)
        unique_rod_ids = set()
        frame_count = 0
        current_boxes = [] # Store boxes for interpolation during skips
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            frame_count += 1

            # Only run detection every 'frame_skip' frames
            if frame_count % frame_skip == 1 or frame_skip <= 1:
                # Resize frame for faster inference
                scale = target_height / h
                target_width = int(w * scale)
                resized_frame = cv2.resize(frame, (target_width, target_height))

                # Run ensemble detection on resized frame
                scale_inv = 1.0 / scale
                detected_boxes = ensemble_predict(resized_frame, models, conf=0.5)
                
                # Scale boxes back to original size
                current_boxes = []
                for box in detected_boxes:
                    current_boxes.append((box * scale_inv).astype(int))

                current_frame_tracks = {}
                used_box_indices = set()

                # Match with existing tracks using IOU
                for track_id, last_box in active_tracks.items():
                    best_iou = 0
                    best_box_idx = -1
                    
                    for i, box in enumerate(current_boxes):
                        if i in used_box_indices:
                            continue
                        iou = calculate_iou(last_box, box)
                        if iou > best_iou:
                            best_iou = iou
                            best_box_idx = i
                    
                    if best_iou > 0.2: # Lower threshold due to frame gaps
                        current_frame_tracks[track_id] = current_boxes[best_box_idx]
                        used_box_indices.add(best_box_idx)
                        unique_rod_ids.add(track_id)

                # Assign new IDs to unmatched boxes
                for i, box in enumerate(current_boxes):
                    if i not in used_box_indices:
                        current_frame_tracks[next_id] = box
                        unique_rod_ids.add(next_id)
                        next_id += 1

                active_tracks = current_frame_tracks

            # Draw results (either from this frame or cached from previous)
            for track_id, box in active_tracks.items():
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)

            # Display Total Count on frame
            total_unique_count = len(unique_rod_ids)
            cv2.putText(
                frame,
                f"Total Unique Rods: {total_unique_count}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                3,
            )

            # Show the frame
            cv2.imshow("Steel Rod Detection and Tracking (Optimized)", frame)

            # Break on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print(f"Finished. Total unique rods detected: {len(unique_rod_ids)}")
    else:
        print(f"Error: Unsupported file format '{ext}'. Please use an image or video file.")

def main():
    # Load the trained models
    print("Loading models...")
    model1 = YOLO("best.pt")
    model2 = YOLO("best01.pt")
    models = [model1, model2]

    # Target source
    source_path = "Round-Steel-Rod.jpg" 
    
    if os.path.exists(source_path):
        process_source(source_path, models)
    else:
        print(f"Error: File '{source_path}' not found.")

if __name__ == "__main__":
    main()
