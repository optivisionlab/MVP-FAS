import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from ultralytics import YOLO


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Face detection and alignment")
    parser.add_argument("--mode", choices=["yolo", "insightface"], required=True)
    parser.add_argument("--input-dir", default=None, help="Folder ảnh khi không dùng CSV")
    parser.add_argument("--input-dir-image", default=None, help="Base folder nối với relative path trong CSV")
    parser.add_argument("--csv-file", default=None, help="CSV chứa danh sách ảnh")
    parser.add_argument("--csv-column", default=None, help="Tên cột chứa path ảnh")
    parser.add_argument("--object-column", default="object", help="Tên cột dùng để tạo output folder")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--yolo-model", default=None)
    parser.add_argument("--insightface-model", default="buffalo_l")
    parser.add_argument("--ctx-id", type=int, default=0)
    parser.add_argument("--det-size", type=int, default=640)
    parser.add_argument("--det-thresh", type=float, default=0.5)
    parser.add_argument("--face-size", type=int, default=112)
    return parser.parse_args()


def validate_args(args):
    if not args.csv_file and not args.input_dir:
        raise ValueError("Provide either --csv-file or --input-dir")

    if args.csv_file:
        if not args.csv_column:
            raise ValueError("--csv-column is required when using --csv-file")
        if not Path(args.csv_file).is_file():
            raise FileNotFoundError(f"CSV file not found: {args.csv_file}")

    if args.input_dir and not Path(args.input_dir).is_dir():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    if args.input_dir_image and not Path(args.input_dir_image).is_dir():
        raise FileNotFoundError(f"Input image base directory not found: {args.input_dir_image}")

    if args.mode == "yolo" and not args.yolo_model:
        raise ValueError("--yolo-model is required when --mode yolo")


def load_detector(args):
    if args.mode == "yolo":
        return YOLO(args.yolo_model)

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if args.ctx_id >= 0 else ["CPUExecutionProvider"]
    app = FaceAnalysis(name=args.insightface_model, allowed_modules=["detection"], providers=providers)
    app.prepare(ctx_id=args.ctx_id, det_size=(args.det_size, args.det_size), det_thresh=args.det_thresh)
    return app


def detect_with_yolo(model, img, args):
    result = model.predict(img, imgsz=args.det_size, conf=args.det_thresh, verbose=False)[0]

    if result.boxes is None or len(result.boxes) == 0 or result.keypoints is None:
        return None

    boxes = result.boxes.xyxy.cpu().numpy()
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    idx = int(np.argmax(areas))

    landmarks = result.keypoints.xy[idx].cpu().numpy()

    if landmarks.shape[0] < 5:
        return None

    return landmarks[:5].astype(np.float32)


def detect_with_insightface(app, img):
    faces = app.get(img)

    if not faces:
        return None

    face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))

    if face.kps is None:
        return None

    return face.kps.astype(np.float32)


def detect_landmarks(detector, img, args):
    if args.mode == "yolo":
        return detect_with_yolo(detector, img, args)

    return detect_with_insightface(detector, img)


def get_images_from_csv(args):
    items = []

    with open(args.csv_file, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames or args.csv_column not in reader.fieldnames:
            raise ValueError(f"Column '{args.csv_column}' not found. Available columns: {reader.fieldnames}")

        if args.object_column not in reader.fieldnames:
            raise ValueError(f"Column '{args.object_column}' not found. Available columns: {reader.fieldnames}")

        for row_idx, row in enumerate(reader, start=2):
            image_value = row.get(args.csv_column)
            object_value = row.get(args.object_column)

            if not image_value or not image_value.strip():
                print(f"[SKIP] CSV row {row_idx}: empty image path")
                continue

            if not object_value or not object_value.strip():
                print(f"[SKIP] CSV row {row_idx}: empty object")
                continue

            csv_path = Path(image_value.strip())

            if args.input_dir_image and not csv_path.is_absolute():
                image_path = Path(args.input_dir_image) / csv_path
            else:
                image_path = csv_path

            items.append({
                "image_path": image_path,
                "object": object_value.strip(),
            })

    return items


def get_images_from_folder(args):
    input_dir = Path(args.input_dir)

    return [
        {
            "image_path": path,
            "object": None,
        }
        for path in sorted(input_dir.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def get_image_items(args):
    if args.csv_file:
        return get_images_from_csv(args)

    return get_images_from_folder(args)


def process_image(image_path, object_name, output_dir, detector, args):
    if not image_path.is_file():
        print(f"[NOT FOUND ERROR] {image_path}")
        return False

    img = cv2.imread(str(image_path))

    if img is None:
        print(f"[READ ERROR] {image_path}")
        return False

    landmarks = detect_landmarks(detector, img, args)

    if landmarks is None:
        print(f"[NO FACE ERROR] {image_path}")
        return False

    try:
        aligned = face_align.norm_crop(img, landmark=landmarks, image_size=args.face_size)
    except Exception as e:
        print(f"[ALIGN ERROR] {image_path}: {e}")
        return False

    if object_name:
        object_output_dir = output_dir / object_name
    else:
        object_output_dir = output_dir

    object_output_dir.mkdir(parents=True, exist_ok=True)

    output_path = object_output_dir / f"{image_path.stem}.png"

    if not cv2.imwrite(str(output_path), aligned):
        print(f"[SAVE ERROR] {output_path}")
        return False

    print(f"[OK] {image_path} -> {output_path}")
    return True


def main():
    args = parse_args()
    validate_args(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = load_detector(args)
    items = get_image_items(args)

    print(f"Mode       : {args.mode}")
    print(f"Source     : {'CSV' if args.csv_file else 'Folder'}")
    print(f"Total      : {len(items)}")
    print(f"Output dir : {output_dir}")

    success = 0
    failed = 0

    for idx, item in enumerate(items, start=1):
        image_path = item["image_path"]
        object_name = item["object"]

        print(f"[{idx}/{len(items)}] {image_path} | object={object_name}")

        if process_image(image_path, object_name, output_dir, detector, args):
            success += 1
        else:
            failed += 1

    print(f"Done: total={len(items)}, success={success}, failed={failed}")


if __name__ == "__main__":
    main()