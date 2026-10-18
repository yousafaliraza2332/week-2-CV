
import argparse
import cv2
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to input video")
    p.add_argument("--output", default="output.mp4", help="Path to output video")
    p.add_argument("--model", default="yolov8n.pt", help="YOLO model weights")
    p.add_argument("--conf", type=float, default=0.4, help="Confidence threshold")
    args = p.parse_args()

    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.input)

    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.input}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    out = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    frames = 0

    while True:
        success, frame = cap.read()

        if not success:
            break

        result = model.predict(
            frame,
            classes=[0],
            conf=args.conf,
            verbose=False
        )[0]

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])

            label = f"Person {confidence:.2f}"

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        out.write(frame)
        frames += 1

    cap.release()
    out.release()

    print(f"Processing complete: {frames} frames saved to {args.output}")


if __name__ == "__main__":
    main()
