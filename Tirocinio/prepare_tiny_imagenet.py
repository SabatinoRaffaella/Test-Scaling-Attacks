import os
import shutil

root = "data/tiny-imagenet-200"
val_dir = os.path.join(root, "val")
images_dir = os.path.join(val_dir, "images")
annot_file = os.path.join(val_dir, "val_annotations.txt")

# destination: replace val/ with val/wnid subfolders
with open(annot_file, "r") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) < 2:
            continue
        img, wnid = parts[0], parts[1]

        src = os.path.join(images_dir, img)
        dst_dir = os.path.join(val_dir, wnid)
        os.makedirs(dst_dir, exist_ok=True)

        dst = os.path.join(dst_dir, img)
        if not os.path.exists(dst):
            shutil.copy(src, dst)  # change to shutil.move if you want to move instead
