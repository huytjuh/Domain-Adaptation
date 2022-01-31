# Domain-Adaptation

![](https://i.ytimg.com/vi/8L11aMN5KY8/maxresdefault.jpg)

Extension to YOLOv3 person-detection model by incorporating domain adaptation on 3D synthetic images, allowing for synthetic-to-real translation tasks using Cycle-consistent Generative Adversarial Networks (CycleGAN) for one-to-one mapping framework.

A PyTorch implementation of (Bayesian) CycleGAN inspired by [Zhu et al. (2017)]() and [You et al. (2020)]().

***Version: 1.1 (2021)***

---

## Introduction

## Prerequisites

* python 3.8
* tensorflow 2.4
* Keras 2.1.4
* NVIDIA GPU + CUDA CuDNN
* Blender 2.92 (custom-built)

## Getting Started

### Installation Blender 2.92 Custom-Built
* Enable rendering of viewer nodes in background mode to properly update the pixels within Blender background.
* Open `source/blender/compositor/operations/COM_ViewerOperation.h` and change lines:
```
bool isOutputOperation(bool /*rendering*/) const { 
if (G.background) return false; return isActiveViewerOutput();
```
into:
```
bool isOutputOperation(bool /*rendering*/) const {return isActiveViewerOutput(); }
```
* Open `source/blender/compositor/operations/COM_PreviewOperation.h` and change line:
```
bool isOutputOperation(bool /*rendering*/) const { return !G.background; }
```
into:
```
bool isOutputOperation(bool /*rendering*/) const { return true; }
```

### Create Synthetic Images in Blender + Annotations
* Render 3D person images.
```
#!./scripts/run_blender.sh
"Blender Custom/blender.exe" --background --python "Data/Blender.py" -- 1
```
* Annotation files are saved in the respective `.txt` file with the same name and has the following format:
```
image_file_path min_x,min_y,max_x,max_y,class_id min_x,min_y,max_x,max_y,class_id ...
```

### Run YOLOv3 Blender synthetic model
* Run Trained Blender Synthetic Model.
```
#!./scripts/run_yolov3.sh
python3 scripts/yolo_video.py --image
python3 scripts/evaluation.py
```
* The bounding box predictions are saved in folder `output`.
* Performance scores and evaluation metrics are saved in `Evaluation` (Default is `overlap_threshold=0.5`).

## Test Results & Performances

![](https://github.com/huytjuh/YOLOv3-Blender/blob/main/example/example_graph.png)

## Custom Datasets for YOLOv3 Blender Training

### YOLOv3 Blender Training 
* Select & combine annotation files into a single `.txt` file as input for YOLOv3 training. Edit `Annotations/cfg.txt` accordingly.
```
!./scripts/run_annotations.sh
python3 Annotations/Annotation_synthetic2.py
```
* Specify the following three folders in your `Main/Model_<name>` folder required to train YOLOv3 model:
  * `Model_<name>/Model`: `synthetic_classes.txt` (class_id file) and `yolo_anchors.txt` (default anchors).
  * `Model_<name>/Train`: `DarkNet53.h5` (default .h5 weight) and `Model_Annotations.txt` (final annotation `.txt` file).
  * `Model_<name>/linux_logs`: Saves a `train.txt` logfile and includes training process and errors if there are any.
* Specify learning parameters and number of epochs in `train.py`. Defaults are:
  * Initial Stage (Freeze first 50 layers): `Adam(lr=1e-2)`, `Batch_size=8`, `Epochs=10`
  * Main Process (Unfreeze all layers): `Adam(lr=1e-3)`, `Batch_size=8`, `Epochs=100`
* Recompile anchor boxes using `kmeans.py` script (OPTIONAL)
* Configure settings and initialize paths in `Model_<name>/cfg.txt`
* Train YOLOv3 model.
```
!./scripts/run_train.sh
python3 train.py >Main/Model_Synth_Lab/linux_logs/train.log
```

### Benchmark & Evaluate All YOLOv3 Trained Models
* Obtain Precision-Recall (PR) curve and highest F1-scores by iterating through all `Main/Model_<name>/Evaluation` folders and calculate & combine all performance scores.
```
!./scripts/run_scores.sh
python3 scores_all.py
python3 Visualizations/create_graphs.py
python3 Results_IMGLabels/scores_IMGLabels.py
```
* Case-by-case AP-score Evaluation using `Main/scores_IMGLabels.py` (OPTIONAL)
  * Resulting case-by-case evaluation score can be found in `Main/Evaluation_IMGlabels-case.xlsx` with each tab corresponding to a feature kept fixed.

## Extracting RGB Images from Google OpenImages Database v6
Google’s OpenImages Database v6 dataset is used to collect negative non-person samples by extracting pre-annotated images that includes all kinds of objects and environments but without containing instances of persons.
* Non-person images are filtered and downloaded.
```
!./scripts/run_openimages.sh
python3 OpenImages.py > OpenImages/openimages.log
```
* Configure settings and initialize paths in `OpenImages/cfg.txt`.
* Annotation files are saved in the respective `.txt` file with the same name and has the following format:
```
image_file_path min_x,min_y,max_x,max_y,class_id min_x,min_y,max_x,max_y,class_id ...
```

> Source: https://storage.googleapis.com/openimages/web/download.html

## Acknowledgements

Code is inspired by [qqwweee/keras-yolo3](https://github.com/qqwweee/keras-yolo3).
