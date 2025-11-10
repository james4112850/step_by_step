import os
from typing import Optional

from utils import (
    adjust_hsv_lightness_by_percentile,
    basename_no_ext,
    extract_numeric_token,
    list_images,
    read_image_bgr,
    write_image,
    ensure_dir,
)


def main(input_dir: Optional[str] = None, output_dir: Optional[str] = None) -> None:
    project_root = os.path.abspath(os.path.dirname(__file__))
    if input_dir is None:
        input_dir = os.path.join(project_root, "raw_images")
    if output_dir is None:
        output_dir = os.path.join(project_root, "a_preprocess1")
    # 嚴格檢查輸入資料夾來源
    expected_input = os.path.join(project_root, "raw_images")
    if os.path.normcase(os.path.abspath(input_dir)) != os.path.normcase(os.path.abspath(expected_input)):
        raise ValueError(f"input_dir 必須為 {expected_input}，目前為 {input_dir}")
    ensure_dir(output_dir)

    image_paths = list_images(input_dir)
    if not image_paths:
        print(f"No images found in {input_dir}")
        return

    for path in image_paths:
        try:
            image = read_image_bgr(path)
            processed = adjust_hsv_lightness_by_percentile(image, low=3, high=97)
            token = extract_numeric_token(basename_no_ext(path))
            out_name = f"{token}_a(preprocess1).jpg"
            out_path = os.path.join(output_dir, out_name)
            write_image(out_path, processed)
            print(f"Wrote: {out_path}")
        except Exception as e:
            print(f"Failed on {path}: {e}")


if __name__ == "__main__":
    main()


