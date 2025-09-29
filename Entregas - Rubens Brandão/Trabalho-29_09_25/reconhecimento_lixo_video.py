
from ultralytics import YOLO
import cv2

model = YOLO("model/best.pt")

video_path = "video/video3.mp4"

cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter(
    "output_detected_simple.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

max_display_width = 800
max_display_height = 600

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(
        source=frame,
        conf=0.25,
        iou=0.45,
        imgsz=640,
        verbose=False
    )

    annotated_frame = results[0].plot()

    h, w = annotated_frame.shape[:2]
    scale = min(max_display_width / w, max_display_height / h, 1.0)
    display_w, display_h = int(w * scale), int(h * scale)
    resized_frame = cv2.resize(annotated_frame, (display_w, display_h))

    cv2.imshow("Detecção YOLO", resized_frame)
    out.write(annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ Processamento finalizado!")
