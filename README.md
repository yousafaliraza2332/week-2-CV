# Human Detection Using YOLOv8

This application performs person detection in video files using a pretrained YOLOv8 model. It identifies humans, marks their locations with rectangular boxes, and displays a confidence value for each detection. The final video is saved for later viewing.

## 1. Requirements

Install the project dependencies by running:

```bash
pip install -r requirements.txt
```

## 2. Execution

Start the detection process with this command:

```bash
python detect_people.py --input input.mp4 --output output.mp4 --conf 0.4
```

## 3. Configuration

The program supports the following command-line arguments:

- `--input`: Path to the video that needs to be analyzed.
- `--output`: Destination for the resulting video.
- `--model`: Path or name of the YOLO model weights; defaults to `yolov8n.pt`.
- `--conf`: Minimum detection confidence, set to `0.4` by default.

## 4. Expected Output

After processing, the program produces an annotated video file. Green rectangles indicate detected people, and confidence scores appear alongside the boxes. The model filters out other object types so that only people are included in the detection results.