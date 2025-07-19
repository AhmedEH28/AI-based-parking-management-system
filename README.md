# AI-based Parking Management System

This project is an advanced, AI-powered parking management system that uses a custom-trained YOLO model to detect cars in parking slots from a top-down video feed. It provides real-time occupancy statistics and visualizes both detected vehicles and parking slot statuses on the video output.

## Features
- Detects cars in parking slots using a custom YOLO model
- Visualizes occupied and available slots with color-coded overlays
- Draws bounding boxes only for valid car detections
- Outputs annotated video with real-time statistics (total, occupied, available, occupancy rate)
- Highly configurable and easy to adapt to new parking layouts

## Requirements
- Python 3.8+
- See `requirements.txt` for dependencies

## Installation
1. Clone this repository:
   ```bash
   git clone <your-repo-url>
   cd <your-repo-directory>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Place your parking video and parking slot JSON file in the project directory.
2. Update the paths in `run_parking.py` if needed:
   - `video_path`: Path to your parking video
   - `json_path`: Path to your parking slot JSON file
   - `model_path`: Path to your trained YOLO model (e.g., `best.pt`)
   - `output_path`: Output video file name
3. Run the script:
   ```bash
   python run_parking.py
   ```
4. The output video with overlays will be saved to the specified output path.

## Parking Slot JSON Format
The JSON file should contain a list of slot definitions, each with a list of `points` (polygon coordinates):
```json
[
  {
    "points": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
  },
  ...
]
```

## Customization
- Adjust the confidence threshold or overlap threshold in `run_parking.py` for your scenario.
- The system is designed for top-down views and can be adapted for other perspectives with retraining.

## License
MIT License

## Acknowledgements
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- OpenCV
- NumPy 