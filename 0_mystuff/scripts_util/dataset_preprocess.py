import json, glob, os
from pathlib import Path
from PIL import Image, ImageOps

data_path="0_mystuff/datasets/sunglasses"
clss="0"

folder = Path(os.path.join(data_path,clss))
for f in folder.glob("*.jp*g"):  # matches .jpg and .jpeg
    try:
        with Image.open(f) as im:
            im = ImageOps.exif_transpose(im)
            if im.mode not in ("RGB", "L"):
                im = im.convert("RGB")
            out = f.with_suffix(".png")
            im.save(out, format="PNG", optimize=True, compress_level=6)

        # delete the original only if conversion succeeded
        if out.exists():
            f.unlink()
            print(f"Converted & deleted: {f.name}")
    except Exception as e:
        print(f"Skipping {f.name}: {e}")

files=sorted(glob.glob(os.path.join(data_path,clss,"*.png")))
print(files)
labels={"labels":[[f"{clss}/{os.path.basename(p)}", 0] for p in files]}
json.dump(labels, open(os.path.join(data_path,"dataset.json"),"w"))

# python main/data/create_imagenet_lmdb.py --data_path 0_mystuff/datasets/pokemon --lmdb_path 0_mystuff/checkpoint_path/pokemon_lmdb