import csv
import os
from typing import List, Optional, Tuple

import numpy as np
from ultralytics import YOLO

from utils import (
    basename_no_ext,
    extract_numeric_token,
    list_images,
    read_image_bgr,
    write_image,
    ensure_dir,
    to_bgr,
)


def dedupe_and_sort_chars(char_entries: List[List[float]]) -> List[List[float]]:
    # char entry: [x1, y1, x2, y2, conf, class_name]
    deduped: List[List[float]] = []
    for entry in char_entries:
        x1, y1, x2, y2, conf, cls_name = entry
        replaced = False
        cx = x1 + x2
        for i, cur in enumerate(deduped):
            ccx = cur[0] + cur[2]
            if abs(ccx - cx) <= 5:
                if conf > cur[4]:
                    deduped[i] = entry
                replaced = True
                break
        if not replaced:
            deduped.append(entry)
    deduped.sort(key=lambda x: x[0])
    return deduped


def apply_taiwan_plate_rules(chars: List[List[float]]) -> List[List[float]]:
    # find dash index
    dash_idx = -1
    for i, c in enumerate(chars):
        if c[5] == '-':
            dash_idx = i
            break
    if dash_idx == -1:
        return chars

    # while dash > 4: pop left
    while dash_idx > 4 and chars:
        chars.pop(0)
        dash_idx -= 1

    # while right side > 5: pop right
    while (len(chars) - dash_idx) > 5 and chars:
        chars.pop(-1)

    # if 5 right and dash == 4: pop first
    if (len(chars) - dash_idx) == 5 and dash_idx == 4 and chars:
        chars.pop(0)
        dash_idx -= 1

    # if 4 right and dash == 4: pop last
    if (len(chars) - dash_idx) == 4 and dash_idx == 4 and chars:
        chars.pop(-1)

    # potential 3+4 error: if len right 5 and dash==3 and first is '1': pop first
    if (len(chars) - dash_idx) == 5 and dash_idx == 3 and chars and chars[0][5] == '1':
        chars.pop(0)
        dash_idx -= 1

    return chars


def recognize_plate_text(model: YOLO, plate_bgr: np.ndarray) -> str:
    result = model.predict(source=plate_bgr, save=False, verbose=False)
    plate_id = ''
    for res in result:
        if res.boxes is None:
            continue
        entries: List[List[float]] = []
        for box in res.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0].item()
            cls_id = int(box.cls[0].item())
            class_name = model.names[cls_id]
            entries.append([x1.item(), y1.item(), x2.item(), y2.item(), conf, class_name])

        chars = dedupe_and_sort_chars(entries)
        chars = apply_taiwan_plate_rules(chars)
        plate_id = ''.join([c[5] for c in chars])
    return plate_id


def main(input_dir: Optional[str] = None, output_dir: Optional[str] = None, weights_path: Optional[str] = None) -> None:
    project_root = os.path.abspath(os.path.dirname(__file__))
    if input_dir is None:
        input_dir = os.path.join(project_root, "d_crop_plate")
    if output_dir is None:
        output_dir = os.path.join(project_root, "e_characters")
    if weights_path is None:
        weights_path = os.path.join(os.path.dirname(__file__), "characters.pt")

    # 嚴格檢查輸入資料夾來源
    expected_input = os.path.join(project_root, "d_crop_plate")
    if os.path.normcase(os.path.abspath(input_dir)) != os.path.normcase(os.path.abspath(expected_input)):
        raise ValueError(f"input_dir 必須為 {expected_input}，目前為 {input_dir}")

    ensure_dir(output_dir)
    results_csv = os.path.join(output_dir, "results.csv")

    char_model = YOLO(weights_path)
    image_paths = list_images(input_dir)
    if not image_paths:
        print(f"No images found in {input_dir}")
        return

    with open(results_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["image", "plate_text"]) 

        for path in image_paths:
            try:
                # read as BGR regardless extension; these were saved gray but conversion handles it
                bgr = read_image_bgr(path)
                bgr = to_bgr(bgr)
                text = recognize_plate_text(char_model, bgr)

                token = extract_numeric_token(basename_no_ext(path))
                # keep any index suffix coming from plate step
                base = basename_no_ext(path)
                suffix = ""
                parts = base.split("_d(crop_plate)")
                if len(parts) == 2 and parts[1]:
                    suffix = parts[1]

                txt_name = f"{token}_e(characters){suffix}.txt"
                txt_path = os.path.join(output_dir, txt_name)
                ensure_dir(os.path.dirname(txt_path))
                with open(txt_path, 'w', encoding='utf-8') as tf:
                    tf.write(text)
                writer.writerow([os.path.basename(path), text])
                print(f"Wrote: {txt_path} -> {text}")
            except Exception as e:
                print(f"Failed on {path}: {e}")


if __name__ == "__main__":
    main()


