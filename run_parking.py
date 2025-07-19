import cv2
import json
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple, Any
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedParkingManagement:
    """
    Enhanced parking management system with proper occupancy detection and visualization.
    Handles loading of parking slot definitions, vehicle detection, occupancy logic, and visualization overlays.
    """
    def __init__(self, model_path: str, json_path: str, conf_threshold: float = 0.3):
        """
        Initialize the parking management system.
        Args:
            model_path (str): Path to the YOLO model.
            json_path (str): Path to the parking slot JSON file.
            conf_threshold (float): Confidence threshold for YOLO detections.
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold if conf_threshold else 0.1
        self.parking_slots = self.load_parking_slots(json_path)
        self.occupied_slots = set()
        self.available_slots = set()
        logger.info(f"Initialized with {len(self.parking_slots)} parking slots")
        self.colors = {
            'occupied': (0, 0, 255),
            'available': (0, 255, 0),
            'text': (255, 255, 255),
            'background': (0, 0, 0),
            'car_label': (255, 255, 0)
        }

    def load_parking_slots(self, json_path: str) -> List[Dict]:
        """
        Load parking slot coordinates from a JSON file.
        Args:
            json_path (str): Path to the JSON file.
        Returns:
            List[Dict]: List of slot definitions with id and coordinates.
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            slots = []
            if isinstance(data, list):
                for i, slot in enumerate(data):
                    if isinstance(slot, dict):
                        normalized_slot = {
                            "id": slot.get('id', i),
                            "coordinates": slot.get('points') or slot.get('coordinates') or slot.get('polygon')
                        }
                        if normalized_slot["coordinates"]:
                            slots.append(normalized_slot)
                    else:
                        logger.warning(f"Invalid slot format at index {i}: {slot}")
            elif isinstance(data, dict):
                if 'slots' in data or 'parking_slots' in data:
                    slot_data = data.get('slots', data.get('parking_slots', []))
                    for i, slot in enumerate(slot_data):
                        normalized_slot = {
                            "id": slot.get('id', i),
                            "coordinates": slot.get('points') or slot.get('coordinates') or slot.get('polygon')
                        }
                        if normalized_slot["coordinates"]:
                            slots.append(normalized_slot)
                elif 'points' in data:
                    slots = [{"id": 0, "coordinates": data["points"]}]
                else:
                    raise ValueError("Invalid JSON structure - no recognizable slot data found")
            else:
                raise ValueError("Invalid JSON structure - expected list or dict")
            logger.info(f"Loaded {len(slots)} parking slots")
            if len(slots) == 0:
                logger.error("No valid parking slots found in JSON file!")
            else:
                logger.info(f"First slot example: {slots[0]}")
            return slots
        except Exception as e:
            logger.error(f"Error loading parking slots: {e}")
            return []

    @staticmethod
    def polygon_to_bbox(polygon: List[List[int]]) -> Tuple[int, int, int, int]:
        """
        Convert a polygon to a bounding box (min x, min y, max x, max y).
        """
        if not polygon:
            return 0, 0, 0, 0
        x_coords = [p[0] for p in polygon]
        y_coords = [p[1] for p in polygon]
        return min(x_coords), min(y_coords), max(x_coords), max(y_coords)

    @staticmethod
    def calculate_overlap(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """
        Calculate the intersection-over-union (IoU) of two bounding boxes.
        """
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        intersection = x_overlap * y_overlap
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0

    @staticmethod
    def is_point_in_polygon(point: Tuple[int, int], polygon: List[List[int]]) -> bool:
        """
        Ray casting algorithm for testing if a point is inside a polygon.
        """
        x, y = point
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y + 1e-9) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def detect_occupancy(self, frame: np.ndarray) -> Tuple[int, int, list, list]:
        """
        Detect vehicles and determine slot occupancy for a given frame.
        Returns:
            Tuple: (occupied_count, available_count, all_vehicles, matched_vehicles)
        """
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        vehicles = []
        matched_vehicles = []
        # Calculate average slot size for filtering large boxes
        slot_widths = []
        slot_heights = []
        for slot in self.parking_slots:
            coords = slot.get('coordinates') or slot.get('polygon') or slot.get('points')
            if coords:
                x_coords = [p[0] for p in coords]
                y_coords = [p[1] for p in coords]
                slot_widths.append(max(x_coords) - min(x_coords))
                slot_heights.append(max(y_coords) - min(y_coords))
        avg_slot_width = np.mean(slot_widths) if slot_widths else 0
        avg_slot_height = np.mean(slot_heights) if slot_heights else 0
        max_box_width = avg_slot_width * 2.2
        max_box_height = avg_slot_height * 2.2
        if results[0].boxes is not None:
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                if class_id != 0:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                box_width = x2 - x1
                box_height = y2 - y1
                if box_width > max_box_width or box_height > max_box_height:
                    continue
                vehicles.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': float(box.conf[0]),
                    'class_id': class_id
                })
        if hasattr(self, 'frame_count'):
            self.frame_count += 1
        else:
            self.frame_count = 1
        if self.frame_count % 30 == 0:
            logger.info(f"Detected {len(vehicles)} objects in frame {self.frame_count}")
        self.occupied_slots.clear()
        self.available_slots.clear()
        for i, slot in enumerate(self.parking_slots):
            slot_id = slot.get('id', i)
            coordinates = slot.get('coordinates') or slot.get('polygon') or slot.get('points')
            if not coordinates:
                continue
            is_occupied = False
            slot_bbox = self.polygon_to_bbox(coordinates)
            for vehicle in vehicles:
                vehicle_bbox = vehicle['bbox']
                overlap = self.calculate_overlap(vehicle_bbox, slot_bbox)
                center = (
                    (vehicle_bbox[0] + vehicle_bbox[2]) // 2,
                    (vehicle_bbox[1] + vehicle_bbox[3]) // 2
                )
                if overlap > 0.10 or self.is_point_in_polygon(center, coordinates):
                    is_occupied = True
                    matched_vehicles.append(vehicle)
                    break
            if is_occupied:
                self.occupied_slots.add(slot_id)
            else:
                self.available_slots.add(slot_id)
        if self.frame_count % 30 == 0:
            logger.info(f"Frame {self.frame_count} - Occupied: {len(self.occupied_slots)}, Available: {len(self.available_slots)}")
        return len(self.occupied_slots), len(self.available_slots), vehicles, matched_vehicles

    def draw_detected_cars(self, frame: np.ndarray, vehicles: list) -> np.ndarray:
        """
        Draw bounding boxes and labels for detected cars.
        """
        for vehicle in vehicles:
            x1, y1, x2, y2 = vehicle['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.colors['car_label'], 1)
            text = "car"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            text_x = x1 + (x2 - x1) // 2 - text_size[0] // 2
            text_y = max(y1 - 5, 10)
            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['car_label'], 1, cv2.LINE_AA)
        return frame

    def draw_parking_slots(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw outlines and IDs for all parking slots, color-coded by occupancy.
        """
        for i, slot in enumerate(self.parking_slots):
            slot_id = slot.get('id', i)
            coordinates = slot.get('coordinates') or slot.get('polygon') or slot.get('points')
            if not coordinates:
                continue
            points = np.array(coordinates, dtype=np.int32)
            color = self.colors['occupied'] if slot_id in self.occupied_slots else self.colors['available']
            cv2.polylines(frame, [points], True, color, 2)
            center_x = sum(p[0] for p in coordinates) // len(coordinates)
            center_y = sum(p[1] for p in coordinates) // len(coordinates)
            cv2.putText(frame, str(slot_id), (center_x - 10, center_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 2)
        return frame

    def draw_statistics(self, frame: np.ndarray, occupied_count: int, available_count: int) -> np.ndarray:
        """
        Draw statistics overlay (total, occupied, available, occupancy rate) at the top of the frame.
        """
        total_slots = occupied_count + available_count
        stats_height = 100
        stats_bg = np.zeros((stats_height, frame.shape[1], 3), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(stats_bg, f"Total Slots: {total_slots}", (20, 30), font, 0.8, self.colors['text'], 2)
        cv2.putText(stats_bg, f"Occupied: {occupied_count}", (20, 60), font, 0.8, self.colors['occupied'], 2)
        cv2.putText(stats_bg, f"Available: {available_count}", (250, 60), font, 0.8, self.colors['available'], 2)
        occupancy_rate = (occupied_count / total_slots * 100) if total_slots > 0 else 0
        cv2.putText(stats_bg, f"Occupancy Rate: {occupancy_rate:.1f}%", (450, 60), font, 0.8, self.colors['text'], 2)
        cv2.rectangle(stats_bg, (frame.shape[1] - 200, 20), (frame.shape[1] - 180, 40), self.colors['occupied'], -1)
        cv2.putText(stats_bg, "Occupied", (frame.shape[1] - 170, 35), font, 0.5, self.colors['text'], 1)
        cv2.rectangle(stats_bg, (frame.shape[1] - 200, 50), (frame.shape[1] - 180, 70), self.colors['available'], -1)
        cv2.putText(stats_bg, "Available", (frame.shape[1] - 170, 65), font, 0.5, self.colors['text'], 1)
        return np.vstack([stats_bg, frame])

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single video frame: detect cars, update occupancy, and draw overlays.
        """
        occupied_count, available_count, vehicles, matched_vehicles = self.detect_occupancy(frame)
        frame_with_cars = self.draw_detected_cars(frame, matched_vehicles)
        annotated_frame = self.draw_parking_slots(frame_with_cars)
        final_frame = self.draw_statistics(annotated_frame, occupied_count, available_count)
        return final_frame

def main():
    """
    Main function to process the parking video and output annotated results.
    """
    video_path = "parking_video1.mp4"
    json_path = "bounding_boxes.json"
    model_path = "parkbest.pt"
    output_dir = "outputs"
    output_path = os.path.join(output_dir, "parking_output4.mp4")
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video file: {video_path}")
        return
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
    try:
        parking_manager = EnhancedParkingManagement(
            model_path=model_path,
            json_path=json_path,
            conf_threshold=0.1
        )
    except Exception as e:
        logger.error(f"Failed to initialize parking management: {e}")
        cap.release()
        return
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_height = height + 100
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, out_height))
    logger.info("Starting video processing...")
    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            processed_frame = parking_manager.process_frame(frame)
            writer.write(processed_frame)
            frame_count += 1
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                logger.info(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    except Exception as e:
        logger.error(f"Error during processing: {e}")
    finally:
        cap.release()
        writer.release()
        logger.info(f"Processing complete! Output saved to: {output_path}")

if __name__ == "__main__":
    main()
# Minor change for commit demo
