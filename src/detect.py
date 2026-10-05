from pathlib import Path

import cv2
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"
ASSETS_DIR = PROJECT_ROOT / "assets"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "car_detection"

VIDEO_PATH = ASSETS_DIR / "sample_video.mp4"

image_files = list(ASSETS_DIR.glob("sample_image.*"))

if not image_files:
    raise FileNotFoundError("Sample image not found in assets folder.")

IMAGE_PATH = image_files[0]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def draw_car_count(image, count):
    text = f"Car Count: {count}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1

    (text_width, text_height), _ = cv2.getTextSize(
        text, font, font_scale, thickness
    )

    margin = 12
    padding_x = 8
    padding_y = 6

    x = image.shape[1] - text_width - margin
    y = image.shape[0] - margin

    x1 = x - padding_x
    y1 = y - text_height - padding_y
    x2 = x + text_width + padding_x
    y2 = y + padding_y

    overlay = image.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)

    cv2.addWeighted(overlay, 0.65, image, 0.35, 0, image)

    cv2.putText(
        image,
        text,
        (x, y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )

    return image


def process_image(model):
    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")

    result = model.predict(
        source=image,
        classes=[3],
        verbose=False
    )[0]

    car_count = len(result.boxes)

    output_image = result.plot()
    output_image = draw_car_count(output_image, car_count)

    output_path = OUTPUT_DIR / "detected_sample_image.jpg"
    cv2.imwrite(str(output_path), output_image)

    print(f"Image Car Count: {car_count}")
    print(f"Image saved to: {output_path}")


def process_video(model):
    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {VIDEO_PATH}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    output_path = OUTPUT_DIR / "detected_sample_video.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height)
    )

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        result = model.predict(
            source=frame,
            classes=[3],
            verbose=False
        )[0]

        car_count = len(result.boxes)

        output_frame = result.plot()
        output_frame = draw_car_count(output_frame, car_count)

        writer.write(output_frame)

    cap.release()
    writer.release()

    print(f"Video saved to: {output_path}")


def main():
    model = YOLO(MODEL_PATH)

    process_image(model)
    process_video(model)

    print("Detection completed.")
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()