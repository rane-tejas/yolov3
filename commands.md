## Commands to run and evaluate YOLO v3-tiny models for vessel segmentation

***

### Table of Contents:
1. [Requirements](#requirements)
2. [Pre-Training](#repository-overview)
3. [Training the YOLO v3-tiny](#requirements)
4. [Evaluation](#building-the-repository)
5. [Ellipse Fitting](#running-the-robot)

***

### Requirements

Create a conda environment using the `conda_req.txt` file.

```bash
>>> conda create --name <env_name> --file conda_req.txt
>>> conda activate <env_name>
```

If using pip instead of conda, install the requirements using the `pip_req.txt` file.

```bash
>>> pip install -r pip_req.txt
```

***

### Pre-Training

Create the `dataset` directory and add the `name_of_dataset` dataset to this directory. The dataset should follow the following folder structure and name convention.

~~~{.bash}
dataset
└── name_of_dataset
    ├── train
    │   ├── images      # Images of vessels (256*256*3, .png images)
    │   └── labels_bb   # Bounding box labels (txt)
    ├── valid
    │   ├── images      # Images of vessels (256*256*3, .png images)
    │   └── labels_bb   # Bounding box labels (txt)
    └── test
        ├── images      # Images of vessels (256*256*3, .png images)
        └── labels_bb   # Bounding box labels (txt)
~~~

In the `data` directory, create a YAML file coresponing your dataset, with the following contents.

```yaml
path: dataset/mixed_pig_fukuda_dataset1     # dataset root dir

train: train/images                         # train images (relative to 'path')
val: valid/images                           # val images (relative to 'path') 128 images
test: test/images                           # test images (optional)

# Classes
nc: 2                                       # number of classes
names: ['artery', 'vein']                   # name of classes
```
For vessel segmentation in ultrasound images, we have already provided the `data/vessel_multiclass.yaml` file.

Create a `runs` directory to store the training results. The `runs` directory should have `train` and `val` subdirectories to store the training and validation results respectively.

***

### Training the YOLO v3-tiny

To train the YOLO v3-tiny model, run the `train.py` script. The `train.py` script takes multiple arguments, but here are a few important ones.

~~~{.bash}
>>> python train.py -h
usage: train.py [-h] [--weights WEIGHTS] [--data DATA][--epochs EPOCHS]
                [--batch-size BATCH_SIZE] [--imgsz IMGSZ] [--name NAME]
                [--single-cls] [--augment]
optional arguments:
  -h, --help            show this help message and exit
  --weights WEIGHTS     initial weights path
  --data DATA           dataset.yaml path
  --epochs EPOCHS       number of epochs
  --batch-size BATCH_SIZE
                        total batch size for all GPUs, -1 for autobatch
  --imgsz IMGSZ, --img IMGSZ, --img-size IMGSZ
                        train, val image size (pixels)
  --single-cls          train multi-class data as single-class
  --name NAME           save to project/name
  --augment             data augmentation needed
~~~

Here are a couple of examples on how to train the YOLO v3-tiny model. The first one shows single class training, and the second one shows multi-class training. The third one demonstrates how these can be run with with data augmentation.

~~~{.bash}
>>> python train.py --data vessel_multiclass.yaml --weights yolov3-tiny.pt --img 256 --epoch 300 --batch-size 16 --name "yolo_singleclass" --single-cls
~~~
~~~{.bash}
>>> python train.py --data vessel_multiclass.yaml --weights yolov3-tiny.pt --img 256 --epoch 300 --batch-size 16 --name "yolo_multiclass"
~~~
~~~{.bash}
>>> python train.py --data vessel_multiclass.yaml --weights yolov3-tiny.pt --img 256 --epoch 300 --batch-size 16 --name "yolo_multiclass_aug" --augment
~~~

***

### Evaluation

To evaluate the YOLO v3-tiny trained model, run the `infer.py` script. This script is hardcoded to perform inference on the `Pig_A`, `Pig_B`, `Pig_C` and `Pig_D` datasets. Download these datasets (.zip files) from the TRACIR shared Google Drive under the `datasets` folder. The `infer.py` script takes multiple arguments, but here are a few important ones.

~~~{.bash}
>>> python infer.py -h
usage: infer.py [-h] [--data DATA] [--weights WEIGHTS [WEIGHTS ...]]
                [--batch-size BATCH_SIZE] [--imgsz IMGSZ] [--augment]
                [--name NAME] [--single-cls]

optional arguments:
  -h, --help            show this help message and exit
  --data DATA           dataset.yaml path
  --weights WEIGHTS [WEIGHTS ...]
                        model.pt path(s)
  --batch-size BATCH_SIZE
                        batch size
  --imgsz IMGSZ, --img IMGSZ, --img-size IMGSZ
                        inference size (pixels)
  --augment             augmented inference
  --name NAME           save to project/name
  --single-cls          treat as single-class dataset
~~~

Here are a couple of examples on how to run the evaluation script. The first one shows single class testing, and the second one shows multi-class testing. The third one demonstrates how these can be run with with data augmentation.

~~~{.bash}
>>> python infer.py --data vessel_multiclass.yaml --weights runs/train/yolo_singleclass/weights/best.pt --img 256 --name "test" --single-cls
~~~
~~~{.bash}
>>> python infer.py --data vessel_multiclass.yaml --weights runs/train/yolo_multiclass/weights/best.pt --img 256 --name "test"
~~~
~~~{.bash}
>>> python infer.py --data vessel_multiclass.yaml --weights runs/train/yolo_singleclass_aug/weights/best.pt --img 256 --name "test" --single-cls --augment
~~~

***

### Ellipse Fitting

After YOLO predicts the bounding boxes, we fit an ellipse to each bounding box using the Spokes Ellipse algorithm, to perform "segmentation" with YOLO. The ellipse fitting is already taken care of in the `infer.py` script.

Ellipse fitting happens in 2 steps, `binary_search` and `fit_ellipse`. The `binary_search` function is the "Spokes" part of the algorithm. It assumes that the center of the bounding box is the center of the ellipse, and creates 8 spokes. Then the alogrithm travels along these spokes in the image to find the point where the intensity of the image changes (from dark to light, the boundary of the vessel), thus performing a binary search. The `binary_search` function returns the 8 points (list of x and y) that are the endpoints of the spokes.

The `fit_ellipse` function is the "Ellipse" part of the algorithm. It takes the 8 end points of the spokes and fits an ellipse on it using the least squares method. The optimization code implementation is taken from this [link](http://juddzone.com/ALGORITHMS/least_squares_ellipse.html). The `fit_ellipse` function returns the center of the ellipse, the major and minor axes, and the angle of the ellipse.

***