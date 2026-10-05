from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"
VIDEO_PATH = PROJECT_ROOT / "assets" / "sample_video.mp4"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "car_tracking"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "tracked_sample_video.mp4"


CAR_CLASS_ID = 3
CONF_THRESHOLD = 0.25

SAMPLE_EVERY = 3

CLUSTER_EPS = 8
MIN_CLUSTER_SAMPLES = 3
MIN_CLUSTER_CONFIDENCE = 0.30

SHORT_CLUSTER_MAX_SAMPLES = 5
STRONG_CLUSTER_MIN_SAMPLES = 20
SHORT_MERGE_DISTANCE = 25.0

ASSIGN_DISTANCE = 30.0


def transform_point(transform, x, y):
    point = np.array([x, y, 1.0], dtype=np.float32)
    result = transform @ point

    if abs(result[2]) > 1e-8:
        result /= result[2]

    return float(result[0]), float(result[1])


def estimate_camera_transform(previous_gray, current_gray):
    points = cv2.goodFeaturesToTrack(
        previous_gray,
        maxCorners=500,
        qualityLevel=0.01,
        minDistance=8,
        blockSize=7
    )

    if points is None or len(points) < 10:
        return np.eye(3, dtype=np.float32)

    next_points, status, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        points,
        None
    )

    if next_points is None or status is None:
        return np.eye(3, dtype=np.float32)

    status = status.reshape(-1).astype(bool)

    old_points = points.reshape(-1, 2)[status]
    new_points = next_points.reshape(-1, 2)[status]

    if len(old_points) < 10:
        return np.eye(3, dtype=np.float32)

    affine, _ = cv2.estimateAffinePartial2D(
        old_points,
        new_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0
    )

    if affine is None:
        return np.eye(3, dtype=np.float32)

    transform = np.eye(3, dtype=np.float32)
    transform[:2] = affine

    return transform


def collect_detections(model):
    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {VIDEO_PATH}")

    all_detections = []
    sampled_points = []
    sampled_info = []

    previous_gray = None
    global_transform = np.eye(3, dtype=np.float32)
    frame_number = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if previous_gray is not None:
            motion = estimate_camera_transform(previous_gray, gray)

            try:
                inverse_motion = np.linalg.inv(motion)
                global_transform = global_transform @ inverse_motion
            except np.linalg.LinAlgError:
                pass

        previous_gray = gray

        result = model.predict(
            source=frame,
            classes=[CAR_CLASS_ID],
            conf=CONF_THRESHOLD,
            verbose=False
        )[0]

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()

            for box, confidence in zip(boxes, confidences):
                x1, y1, x2, y2 = box

                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                global_x, global_y = transform_point(
                    global_transform,
                    center_x,
                    center_y
                )

                detection = [
                    frame_number,
                    float(confidence),
                    x1,
                    y1,
                    x2,
                    y2,
                    global_x,
                    global_y
                ]

                all_detections.append(detection)

                if frame_number % SAMPLE_EVERY == 0:
                    sampled_points.append([global_x, global_y])
                    sampled_info.append([frame_number, float(confidence)])

        frame_number += 1

    cap.release()

    return (
        np.array(all_detections, dtype=np.float32),
        np.array(sampled_points, dtype=np.float32),
        np.array(sampled_info, dtype=np.float32)
    )


def cluster_cars(sampled_points, sampled_info):
    labels = DBSCAN(
        eps=CLUSTER_EPS,
        min_samples=MIN_CLUSTER_SAMPLES
    ).fit_predict(sampled_points)

    clusters = {}

    for cluster_id in sorted(set(labels) - {-1}):
        mask = labels == cluster_id

        points = sampled_points[mask]
        info = sampled_info[mask]

        average_confidence = info[:, 1].mean()

        if average_confidence < MIN_CLUSTER_CONFIDENCE:
            continue

        clusters[cluster_id] = {
            "count": len(points),
            "centroid": points.mean(axis=0),
            "first_frame": int(info[:, 0].min()),
            "last_frame": int(info[:, 0].max()),
            "confidence": float(average_confidence)
        }

    merge_map = {
        cluster_id: cluster_id
        for cluster_id in clusters
    }

    strong_clusters = [
        cluster_id
        for cluster_id, data in clusters.items()
        if data["count"] >= STRONG_CLUSTER_MIN_SAMPLES
    ]

    merged_count = 0

    for cluster_id, data in clusters.items():
        if data["count"] > SHORT_CLUSTER_MAX_SAMPLES:
            continue

        best_target = None
        best_distance = float("inf")

        for target_id in strong_clusters:
            if target_id == cluster_id:
                continue

            distance = np.linalg.norm(
                data["centroid"] - clusters[target_id]["centroid"]
            )

            if distance < best_distance and distance <= SHORT_MERGE_DISTANCE:
                best_distance = distance
                best_target = target_id

        if best_target is not None:
            merge_map[cluster_id] = best_target
            merged_count += 1

    grouped_points = {}
    grouped_frames = {}

    for cluster_id, data in clusters.items():
        final_id = merge_map[cluster_id]

        grouped_points.setdefault(final_id, []).append(
            (data["centroid"], data["count"])
        )

        grouped_frames.setdefault(final_id, []).append(
            data["first_frame"]
        )

    final_clusters = {}

    for cluster_id, values in grouped_points.items():
        total_weight = sum(weight for _, weight in values)

        centroid = sum(
            point * weight
            for point, weight in values
        ) / total_weight

        final_clusters[cluster_id] = {
            "centroid": centroid,
            "first_frame": min(grouped_frames[cluster_id])
        }

    ordered_clusters = sorted(
        final_clusters,
        key=lambda cluster_id: final_clusters[cluster_id]["first_frame"]
    )

    display_ids = {
        cluster_id: index + 1
        for index, cluster_id in enumerate(ordered_clusters)
    }

    print(f"Initial valid clusters: {len(clusters)}")
    print(f"Duplicate clusters merged: {merged_count}")
    print(f"Final unique clusters: {len(final_clusters)}")

    return final_clusters, display_ids


def assign_detections(all_detections, final_clusters, display_ids):
    cluster_ids = list(final_clusters.keys())

    centroids = np.array([
        final_clusters[cluster_id]["centroid"]
        for cluster_id in cluster_ids
    ])

    frame_detections = {}

    for detection in all_detections:
        frame_number = int(detection[0])
        confidence = float(detection[1])

        x1 = int(round(detection[2]))
        y1 = int(round(detection[3]))
        x2 = int(round(detection[4]))
        y2 = int(round(detection[5]))

        global_point = np.array([detection[6], detection[7]])

        distances = np.linalg.norm(
            centroids - global_point,
            axis=1
        )

        closest_index = int(np.argmin(distances))

        if distances[closest_index] > ASSIGN_DISTANCE:
            continue

        cluster_id = cluster_ids[closest_index]
        car_id = display_ids[cluster_id]

        if frame_number not in frame_detections:
            frame_detections[frame_number] = {}

        old_detection = frame_detections[frame_number].get(car_id)

        current_detection = (
            car_id,
            confidence,
            x1,
            y1,
            x2,
            y2
        )

        if old_detection is None or confidence > old_detection[1]:
            frame_detections[frame_number][car_id] = current_detection

    return frame_detections


def draw_box(frame, car_id, confidence, x1, y1, x2, y2):
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    text = f"Car {car_id} | {confidence:.2f}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.42
    thickness = 1

    (text_width, text_height), _ = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness
    )

    label_y1 = max(0, y1 - text_height - 8)

    cv2.rectangle(
        frame,
        (x1, label_y1),
        (x1 + text_width + 6, y1),
        (0, 255, 0),
        -1
    )

    cv2.putText(
        frame,
        text,
        (x1 + 3, y1 - 4),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA
    )


def draw_counts(frame, frame_count, unique_count):
    lines = [
        f"Cars in Frame: {frame_count}",
        f"Car Count: {unique_count}"
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1

    sizes = [
        cv2.getTextSize(text, font, font_scale, thickness)[0]
        for text in lines
    ]

    text_width = max(size[0] for size in sizes)
    text_height = max(size[1] for size in sizes)

    margin = 12
    padding_x = 8
    padding_y = 6
    line_gap = 6

    box_width = text_width + 2 * padding_x
    box_height = 2 * text_height + line_gap + 2 * padding_y

    x2 = frame.shape[1] - margin
    y2 = frame.shape[0] - margin

    x1 = x2 - box_width
    y1 = y2 - box_height

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (x1, y1),
        (x2, y2),
        (0, 0, 0),
        -1
    )

    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    first_y = y1 + padding_y + text_height

    cv2.putText(
        frame,
        lines[0],
        (x1 + padding_x, first_y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        lines[1],
        (x1 + padding_x, first_y + text_height + line_gap),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )


def create_output_video(frame_detections):
    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(OUTPUT_PATH),
        fourcc,
        fps,
        (width, height)
    )

    seen_ids = set()
    frame_number = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        detections = frame_detections.get(frame_number, {})

        for detection in detections.values():
            car_id, confidence, x1, y1, x2, y2 = detection

            seen_ids.add(car_id)

            draw_box(
                frame,
                car_id,
                confidence,
                x1,
                y1,
                x2,
                y2
            )

        draw_counts(
            frame,
            len(detections),
            len(seen_ids)
        )

        writer.write(frame)
        frame_number += 1

    cap.release()
    writer.release()

    return len(seen_ids)


def main():
    print("Loading YOLO model...")
    model = YOLO(MODEL_PATH)

    print("Detecting cars and estimating camera motion...")
    all_detections, sampled_points, sampled_info = collect_detections(model)

    if len(sampled_points) == 0:
        print("No cars detected.")
        return

    print(f"Detections collected: {len(all_detections)}")
    print("Clustering vehicles...")

    final_clusters, display_ids = cluster_cars(
        sampled_points,
        sampled_info
    )

    if not final_clusters:
        print("No valid vehicle clusters found.")
        return

    print("Assigning vehicle IDs...")

    frame_detections = assign_detections(
        all_detections,
        final_clusters,
        display_ids
    )

    print("Creating output video...")

    total_cars = create_output_video(frame_detections)

    print()
    print("Finished.")
    print(f"Car Count: {total_cars}")
    print(f"Video saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
