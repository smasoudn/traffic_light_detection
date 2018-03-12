# [WIP] traffic_light_detection
Here,  I will show how to you can develope a deep model to detect *traffic lights* and *their status* for self-driving cars. To do so, I proposed two approaches:
1) using **TensorflowAPI** which is a quick way to develop your own object detection/classification modules
2) developing a deep neural network **from scratch** using tensorflow

## Traffic light detection using TensorflowAPI
**Note** You need to have Tensorflow version > 1.4.0

1) Follow [these steps](https://github.com/tensorflow/models/blob/3f78f4cfd21c786c62bf321c07830071027ebb5e/research/object_detection/g3doc/installation.md) to install TensorflowAPI properly
2)  Download Udacity traffic light data set (simulator and real data) from [here](https://drive.google.com/file/d/0B-Eiyn-CUQtxdUZWMkFfQzdObUE/view) (Thanks to [Anthony Sarkis](https://medium.com/@anthony_sarkis) for making this data set available).
3) Create your own workspace with **data**, **training** and **pretrained_models** folders:
```
mkdir my_workspace
cd my_workspace
mkdir data
mkdir training
mkdir pretrained_models
```
4) Extract the downloaded data set into the **data/** folder.
5) Convert the Udacity data set to TFRecord format by running the following script:
```
   python create_udacity_sim_tf_record.py  --input_yaml=data/sim_training_data/sim_data_annotations.yaml  --output_path=data/sim_training_data/sim_data.record
   python create_udacity_sim_tf_record.py  --input_yaml=data/real_training_data/real_data_annotations.yaml  --output_path=data/real_training_data/real_data.record
```
Now we have the record data compatible with our Tensorflow API.

6) Download the pretrained model you want to use for this task from [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) and extract it into the **pretrained_models/** folder.
7) Download the *config* file corresponding to the model you just downloaded from [here](https://github.com/tensorflow/models/tree/3f78f4cfd21c786c62bf321c07830071027ebb5e/research/object_detection/samples/configs) and customized it. **Note:** Here, I provided two config models: *ssd_inception_v2_coco* and *faster_rcnn_resnet101_coco*.
8) Prepare a label mapping file (`*.pbtxt`) and put it into **training/** folder. **Note:** `label_map.pbtxt` already has been provided in this repo.

Excellent, now we have everything to do the actual trainig :)
9) Now, copy the folders **data/**, **pretrained_models**, **training** into your **[tensorflow models]/research/object_detection/** folder (where you installed *TensorflowAPI*).
10) Run the training command from *[tensorflow models]/research/*:
```
     python3 train.py  --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_inception_v2_coco.config
```

Wonderful! Now you are training your model. To see how losses are being changed, you may run tensorboard on another terminal and  from *[tensorflow models]/research/* by typing:
```
    tensorboard --logdir=training/
```
and open the link appears in the terminal in your browser.



## Traffic light detection - developping  a deep net from scratch

