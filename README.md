# MorphologyCNN

## Install packages

Install neuron_morphology package: [https://neuron-morphology.readthedocs.io/en/latest/install.html](https://neuron-morphology.readthedocs.io/en/latest/install.html)


## Download data

### Gouwens morphology

Download SWC files from [ftp://download.brainlib.org:8811/biccn/zeng/pseq/morph/200526/](ftp://download.brainlib.org:8811/biccn/zeng/pseq/morph/200526/) into `gouwens-data/data/`

**Extracted features**: Download extracted features (`SpecimenMetadata.csv`) from https://knowledge.brain-map.org/data/1HEYEW7GMUKWIQW37BO/specimens

### Scala morphology

Download SWC files from [https://download.brainimagelibrary.org/3a/88/3a88a7687ab66069/inhibitory/](https://download.brainimagelibrary.org/3a/88/3a88a7687ab66069/inhibitory/) into `scala-data/inhibitory/`

## View SWC files

3D morphology viewer: https://neuroinformatics.nl/HBP/morphology-viewer/

## Analysis steps
1. Run collect_training_images.ipynb to generate .png images from the Gouwens and Scala .swc files. This code will also output some metadata required for copying/moving the files into the correct folders for splitting training/validation/test of the model.

2. Run preprocess.py to downscale images to appropriate sizes

3. the only dataset subfolders you should run for first set of tests is create_t_type_subfolders.py
- if you run other scripts for creating data subfolders, you end up with 21 subtypes in the t-type directory?
- model training code `vanilla_cnn_t_type.py` requires 20 t-type classes for training to work.

4. Run `vanilla_cnn_t_type.py`
- saves model to `model_path = './gouwens-data/models/vanilla_cnn_gouwens_t_type_modified_b2_lr1e-4_e4_rs7_256_vderek'`
- should be able to reload either that model or the previously trained one on github for inference in next step

5. Run `inference_t_type.py`
- test out both versions of model tried.
- how do scores compare on Scala dataset?


