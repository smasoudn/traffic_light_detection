# [WIP] traffic_light_detection
Here,  I will show how to you can develope a deep model to detect *traffic lights* and *their status* for self-driving cars. To do so, I proposed two approaches:
1) using **TensorflowAPI** which is a quick way to develop your own object detection/classification modules
2) developing a deep neural network **from scratch** using tensorflow

## Traffic light detection using TensorflowAPI
- Converting Udacity data-set to TFRecord format:
```
   python create_udacity_sim_tf_record.py  --input_yaml=/home/user/annotation.yaml --output_path=/home/user/udacity.record
```

## Traffic light detection - developping  a deep net from scratch

