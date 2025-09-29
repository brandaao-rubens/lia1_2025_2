
from ultralytics import YOLO
import cv2
import os
from collections import Counter
import matplotlib.pyplot as plt

model = YOLO("model/best.pt")

image_folder = "image"
output_folder = "output_images"
os.makedirs(output_folder, exist_ok=True)

max_display_width = 800
max_display_height = 600

class_counter = Counter()

for file_name in os.listdir(image_folder):
    if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(image_folder, file_name)

        results = model.predict(
            source=image_path,
            conf=0.25,
            iou=0.45,
            imgsz=640,
            verbose=False
        )

        if results[0].boxes is not None:
            labels = results[0].boxes.cls.tolist()
            for cls in labels:
                class_name = model.names[int(cls)]
                class_counter[class_name] += 1

        annotated_image = results[0].plot()

        h, w = annotated_image.shape[:2]
        scale = min(max_display_width / w, max_display_height / h, 1.0)
        display_w, display_h = int(w * scale), int(h * scale)
        resized_image = cv2.resize(annotated_image, (display_w, display_h))

        cv2.imshow("Detecção", resized_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        output_path = os.path.join(output_folder, file_name)
        cv2.imwrite(output_path, annotated_image)

        print(f"Imagem processada e salva em: {output_path}")

if class_counter:
    plt.figure(figsize=(8,5))
    plt.bar(class_counter.keys(), class_counter.values(), color='skyblue')
    plt.xlabel("Classes")
    plt.ylabel("Frequência")
    plt.title("Frequência das classes detectadas")
    plt.xticks(rotation=45)
    plt.show()
else:
    print("Nenhuma classe detectada nas imagens.")

print("✅ Processamento finalizado!")
