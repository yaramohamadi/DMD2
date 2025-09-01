import os
from PIL import Image

def resize_in_place(folder, size=(256, 256)):
    for fname in os.listdir(folder):
        fpath = os.path.join(folder, fname)

        if not os.path.isfile(fpath):
            continue

        try:
            with Image.open(fpath) as img:
                img = img.convert("RGB")
                img_resized = img.resize(size, Image.BILINEAR)

                # always save as PNG
                base, _ = os.path.splitext(fname)
                new_path = os.path.join(folder, f"{base}.png")
                img_resized.save(new_path, format="PNG")

            # delete old file if it wasn't the new PNG
            if not fname.lower().endswith(".png") or fpath != new_path:
                os.remove(fpath)

            print(f"Resized and saved: {new_path}")

        except Exception as e:
            print(f"Skipping {fname}: {e}")

if __name__ == "__main__":
    folder = "0_mystuff_face/datasets/sunglasses/0"
    resize_in_place(folder, size=(256, 256))
