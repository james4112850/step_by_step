import os
from typing import Optional

import cv2
from ultralytics import YOLO

from utils import (
    basename_no_ext,
    extract_numeric_token,
    list_images,
    read_image_bgr,
    write_image,
    ensure_dir,
)


def main(input_dir: Optional[str] = None, output_dir: Optional[str] = None, weights_path: Optional[str] = None) -> None:
    project_root = os.path.abspath(os.path.dirname(__file__))
    if input_dir is None:
        input_dir = os.path.join(project_root, "c_preprocess2")
    if output_dir is None:
        output_dir = os.path.join(project_root, "d_crop_plate")
    if weights_path is None:
        weights_path = os.path.join(os.path.dirname(__file__), "plate.pt")

    # 嚴格檢查輸入資料夾來源
    expected_input = os.path.join(project_root, "c_preprocess2")
    if os.path.normcase(os.path.abspath(input_dir)) != os.path.normcase(os.path.abspath(expected_input)):
        raise ValueError(f"input_dir 必須為 {expected_input}，目前為 {input_dir}")

    ensure_dir(output_dir)

    model = YOLO(weights_path)

    image_paths = list_images(input_dir)
    if not image_paths:
        print(f"No images found in {input_dir}")
        return

    for path in image_paths:
        try:
            image = read_image_bgr(path)
            results = model.predict(source=image, conf=0.8, save=False, verbose=False)
            token = extract_numeric_token(basename_no_ext(path))
            if not results:
                print(f"No plate results: {path}")
                continue

            plate_idx = 0
            for res in results:
                if res.boxes is None:
                    continue
                for box in res.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    if (x2 - x1) <= 60:
                        continue
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(image.shape[1], x2)
                    y2 = min(image.shape[0], y2)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    plate_idx += 1
                    cropped = image[y1:y2, x1:x2]
                    resized = cv2.resize(cropped, (416, 416), interpolation=cv2.INTER_CUBIC)
                    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                    out_name = f"{token}_d(crop_plate)_{plate_idx}.png"
                    out_path = os.path.join(output_dir, out_name)
                    write_image(out_path, gray)
                    print(f"Wrote: {out_path}")
            if plate_idx == 0:
                print(f"No plate detected: {path}")
        except Exception as e:
            print(f"Failed on {path}: {e}")


if __name__ == "__main__":
    main()


