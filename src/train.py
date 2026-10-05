from pathlib import Path
import shutil

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_NAME = "yolov8n.pt"
DATASET_CONFIG = "VisDrone.yaml"

EPOCHS = 20
IMAGE_SIZE = 640


def main():
    model = YOLO(MODEL_NAME)

    model.train(
        data=DATASET_CONFIG,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        project=PROJECT_ROOT / "runs",
        name="visdrone_yolov8n"
    )

    output_model = PROJECT_ROOT / "models" / "best.pt"
    output_model.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy(
        model.trainer.best,
        output_model
    )

    print(f"Best model saved to: {output_model}")


if __name__ == "__main__":
    main()