# Image Recognition Using TensorFlow Machine Learning Demo

### Simple Demonstration to Train a Machine Learning Model to recognize Apples from Bananas

### Installation (Tested on Ubuntu 18.04 - 8GB Memory - 4 core cpu VM):

```
git clone https://github.com/chrisjd20/img_rec_tf_ml_demo.git
cd img_rec_tf_ml_demo
sudo apt install python3 python3-pip -y
sudo python3 -m pip install --upgrade pip
sudo python3 -m pip install --upgrade setuptools
sudo python3 -m pip install --upgrade tensorflow==1.15
sudo python3 -m pip install tensorflow_hub              #this one may or may not be needed in order to run
```

### Training a TensowFlow ML Model Based on Images in `./training_images/` Folder ( tested on PNG files only ):

```
python3 retrain.py --image_dir ./training_images/
```
<sub><sup>`retrain.py` is a slightly modified version of https://raw.githubusercontent.com/tensorflow/hub/master/examples/image_retraining/retrain.py</sup></sub>

This will create two files we will be using at:

1. `/tmp/retrain_tmp/output_graph.pb`     - Trained Machine Learning Model
2. `/tmp/retrain_tmp/output_labels.txt`   - Labels for Images

### Predicting Images in the `unknown_images` folder based on our trained Model:
```
chmod 755 predict_images_using_trained_model.py
./predict_images_using_trained_model.py
```

### Remove Temp Cache Files Between Different Retrains

```
rm -rf /tmp/retrain_tmp /tmp/tfhub_modules
```
