import os
import glob
import numpy as np
from PIL import Image, ImageOps

# Augment existing dataset by rotations and reflections
# output_dir needs to be a copy of input_dir
def augment_images(input_dir, output_dir):
    for i, file in enumerate(glob.glob(input_dir + "/*/*.png")):
        split_path = file.split(os.sep)
        subdir = split_path[-2]
        filename = os.path.splitext(split_path[-1])[0]

        if not os.path.exists(os.path.join(output_dir, subdir)):
            os.mkdir(os.path.join(output_dir, subdir))

        with Image.open(file, 'r') as image:
            # Rotations
            for angle in [0, 90, 180, 270]:
                image_rot = image.rotate(angle)
                # image_rot.save(os.path.join(output_dir, subdir, f"{filename}_{angle}.png"))

                # # Reflections
                image_lr = ImageOps.mirror(image_rot)
                image_tb = ImageOps.flip(image_rot)
                image_lr.save(os.path.join(output_dir, subdir, f"{filename}_{angle}_lr.png"))
                image_tb.save(os.path.join(output_dir, subdir, f"{filename}_{angle}_tb.png"))

        if (i > 0 and i % 10 == 0):
            print(f"Augmented {i} images.")

def main():
    augment_images("./gouwens-data/training_images_t_type/", "./gouwens-data/training_images_t_type_augmented")

if __name__ == "__main__":
    main()