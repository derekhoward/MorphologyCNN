import os
import glob
import numpy as np
from PIL import Image
from pathlib import Path
#import pdb

# Preprocess morphology images to uniform, square input images
def preprocess_images(input_dir, output_dir, downsize=8, transfer_learning=False):
    # Get max image size
    max_width, max_height = 0, 0
    for file in glob.glob(input_dir + "/*.png"):
        with Image.open(file, 'r') as image:
            max_width = max(max_width, image.size[0])
            max_height = max(max_height, image.size[1])
    max_dim = max(max_width, max_height)
    #pdb.set_trace()
    resize = 224 if transfer_learning else max_dim // downsize
    print(resize)

    # Resize images to dimension (max_dim, max_dim)
    for i, file in enumerate(glob.glob(input_dir + "/*.png")):
        output_file = output_dir + os.path.basename(file)
        if not os.path.isfile(output_file):
            with Image.open(file, 'r') as image:
                width, height = image.size
                background = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
                offset_width = (max_dim - width) // 2
                offset_height = (max_dim - height) // 2
                background.paste(image, (offset_width, offset_height))
                background = background.resize((resize, resize))
                background.save(output_dir + os.path.basename(file))
        if (i > 0 and i % 50 == 0):
            print(f"Preprocessed {i} images.")

def main():
    # Pass this argument as True for 224x224 (compatible with pre-trained models for transfer learning)
    transfer_learning = True

    # For other magnitudes of downscaling, uncomment and pass the following:
    # downsize = 8
    # transfer_learning = False

    # input_dir: directory containing unscaled images (original dimensions in 720x720)
    # output_dir: directory where downscaled images will be stored
    
    print('Processing Gouwens dataset')
    gouwens_output_p = Path("./gouwens-data/preprocessed_images/")
    gouwens_output_p.mkdir(exist_ok=True)
    preprocess_images("./gouwens-data/images/", "./gouwens-data/preprocessed_images/", transfer_learning=transfer_learning)
    print('Processing Scala dataset')
    scala_output_p = Path("./scala-data/preprocessed_images/")
    scala_output_p.mkdir(exist_ok=True)
    preprocess_images("./scala-data/inhibitory/images/", "./scala-data/preprocessed_images/", transfer_learning=transfer_learning)
    print('Processing combined dataset')
    combined_output_p = Path("./combined-data/preprocessed_images/")
    combined_output_p.mkdir(exist_ok=True)
    preprocess_images("./combined-data/images/", "./combined-data/preprocessed_images/", transfer_learning=transfer_learning)

if __name__ == "__main__":
    main()
