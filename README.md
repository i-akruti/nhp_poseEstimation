# nhp_poseEstimation
Pose estimation for Non-Human Primates

This code is branched from [lightweight human pose estimation](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch) and changed for non-human primate estimation.

## Requirements
* Ubuntu 20.04
* Python 3.6
* PyTorch >= 1.0 (not tested on CUDA/GPU)

## Installation
Install the requirements by running the 
```python
pip install -r requirements.txt
```

## Data Download and Conversion
* Download the OpenMonkeyChallenge train, validation and test dataset from [OpenMonkeyChallenge Website](https://competitions.codalab.org/competitions/34342)
* The dataset contains annotations which are not in COCO format. So to change the annotation to COCO format run `monkeydataset_2_coco.ipynb` which is present in `scripts` folder. You need to provide the paths to the images, annotaions.json and path to store converted annotation JSON.


 ## Train
 1. Download the pre-trained [model(37000 iterations)](https://download.01.org/opencv/openvino_training_extensions/models/human_pose_estimation/checkpoint_iter_370000.pth) on COCO(human) dataset for initializing weights. Now either initilize the weights using this model to train again,
 **OR**
 use the `Exp3_checkpoints/checkpoint_iter_560.pth` which has been trained by initializing weights from COCO light-weight openpose estimation for 370000 iterations, on OpenMonkeyChallenge train dataset for 560 iterations. The results may not be desirable on the primates dataset. For better results, train for more iterations.
 2. Convert the train annotations ,which have been changed to COCO dataset, to internal format by running the
 `prepare_train_labels.py` in scripts after adding the labels (converted annotation labels) and the output name. It generates the `prepared_train_annotation.pkl` file.
 ```python
python3 prepare_train_labels.py
```
 3. For fast validation, make subset of the validation dataset. Add the path to the validation labels, output name and the number of images for validation dataset. Then run
 ```python
 python3 make_val_subset.py
 ```
 4. To start the training, change the following arguments in the 'train.py':
 
*  prepared_train_labels -- add the path to the `prepared_train_annotation.pkl` file generated earlier
*  train_images_folder -- add path to the training images folder
*  num_refinement_stages -- add the number of refinement stages. The number of refinement stages can be increased but the trade of between accuracy and inference time is observed to be negligible. 
*  base_lr -- add the initial learning rate
*  batch_size -- add the batch size
*  batches_per_iter -- add the number of batchs to accumulate gradient from
*  checkpoint_path --add the path to the pre-trained weights chosen in step 1
*  experiment_name -- add the name of the output folder to create for checkpoints
*  val_labels --path to json with keypoints cal labels
*  val_images_folder --path to OpenMonkeyChallenge validation images folder
*  finetune_monkey --set True to finetune the complete architecture for the OpenMonkeyChallenge dataset
*  num_heatmaps -- should correspond to number of keypoints to detect + 1 for background
*  num_pafs -- should correspond to number of connections between the keypoints, i.e., keypoints pairs for grouping

Run the 'train.py';

```python
python3 train.py
```

 ## Inference
 
 For inference, change the arguments in the `demo.py`, add the path to the checkpoint and add a list of path of images path.
```python
python3 demo.py
```
 
