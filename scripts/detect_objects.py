import os
from ultralytics import YOLO


def detect_objects(image_folder, output_folder, model_path='yolov8s.pt'):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load YOLOv8 model
    model = YOLO(model_path)

    for img_file in os.listdir(image_folder):
        img_path = os.path.join(image_folder, img_file)

        # Perform detection
        results = model(img_path)

        # Save results
        results.save(save_dir=output_folder)


if __name__ == "__main__":
    image_folder = "data/images/"
    output_folder = "data/detections/"
    model_path = "yolov8s.pt"  # Path to your YOLOv8 model
    detect_objects(image_folder, output_folder, model_path)
