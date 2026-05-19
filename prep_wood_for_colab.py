"""Resize wood images and bundle them into a small zip for Colab upload.

Why: raw wood/ is ~395 MB. Training resizes everything to 256x256 internally
(pad_to_square), so anything above 512 px on the longest side is wasted.
Resizing to 512 px JPEG @ q90 typically drops the bundle to ~30-50 MB, which
uploads to Colab in seconds via the file picker.

Run:
    python prep_wood_for_colab.py

Produces:
    wood_data.zip   (drop this into Colab via files.upload())
"""

import os
import zipfile

import cv2


SRC_DIR = os.path.join("data", "wood")
OUT_ZIP = "wood_data.zip"
MAX_SIDE = 512
JPEG_QUALITY = 90
CLASSES = ("clean", "smudged", "cracked")
VALID_EXTS = (".jpg", ".jpeg", ".png")
# Code files bundled into the same zip so Colab only needs one upload.
CODE_FILES = ("layers.py", "model.py", "dataset.py", "train.py", "evaluate.py")


def resize_keep_aspect(img, max_side):
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return img
    scale = max_side / longest
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def main():
    if not os.path.isdir(SRC_DIR):
        raise SystemExit(f"Source folder not found: {SRC_DIR}")

    total_in = 0
    total_out = 0
    with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        # 1. Bundle code files at the top of the zip.
        for code_file in CODE_FILES:
            if not os.path.isfile(code_file):
                print(f"  skipped (missing code): {code_file}")
                continue
            zf.write(code_file, arcname=code_file)
            print(f"  bundled: {code_file}")

        # 2. Bundle resized wood images.
        for cls in CLASSES:
            cls_dir = os.path.join(SRC_DIR, cls)
            if not os.path.isdir(cls_dir):
                print(f"  skipped (missing): {cls_dir}")
                continue

            files = sorted(f for f in os.listdir(cls_dir) if f.lower().endswith(VALID_EXTS))
            print(f"  {cls}: {len(files)} images")

            used_names = set()
            for fname in files:
                src_path = os.path.join(cls_dir, fname)
                total_in += os.path.getsize(src_path)

                img = cv2.imread(src_path, cv2.IMREAD_COLOR)
                if img is None:
                    print(f"    skip unreadable: {fname}")
                    continue

                resized = resize_keep_aspect(img, MAX_SIDE)
                # Re-encode as JPEG. .jpg extension keeps loaders happy even
                # if the source was .png.
                ok, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                if not ok:
                    print(f"    skip encode-fail: {fname}")
                    continue

                # Drop extension and append .jpg. If two source files share a
                # stem (e.g. "8.jpg" + "8.jpeg"), append a numeric suffix so
                # neither image gets dropped from the bundle.
                stem = os.path.splitext(fname)[0]
                candidate = stem
                suffix = 1
                while candidate in used_names:
                    candidate = f"{stem}_{suffix}"
                    suffix += 1
                used_names.add(candidate)

                arcname = f"data/wood/{cls}/{candidate}.jpg"
                zf.writestr(arcname, buf.tobytes())
                total_out += len(buf)

    print(f"\nWrote {OUT_ZIP}")
    print(f"Source size : {total_in / 1024 / 1024:7.1f} MB")
    print(f"Bundled size: {total_out / 1024 / 1024:7.1f} MB ({total_out/total_in*100:.1f}% of original)")
    print(f"Final zip   : {os.path.getsize(OUT_ZIP) / 1024 / 1024:7.1f} MB")


if __name__ == "__main__":
    main()
