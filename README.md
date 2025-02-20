# Code
* This is the code for DCF-YOLO. The specific innovative points are as follows: the code for the CAS module is in cas.py, the DySample module is in dysample.py, and the Focaler-IoU loss function is included in loss.py; the model configuration file is dcf-yolo.yaml.
* If you want to use our code, you must have the following preparation under the PyTorch framework: see requirement.txt for details
* Code Guidance: Download the dataset in the **Dataset** **link**, put the  images  into "firedatset/images" ,replace the common.py in the DCF-YOLO/models folder with the common.py in the DCF-YOLO folder, and replace the loss.py in the DCF-YOLO/utils folder with the loss.py in the DCF-YOLO folder,and then run the train.py to successfully train our model.

# DOI: 10.5281/zenodo.14880534

# Dataset
The complete dataset link is https://pan.baidu.com/s/1NW0S6MjWYbSVjRb1RW1iHA.
Extraction code: 0215

# Citation
@article{
  title={Enhanced YOLOv7-tiny for Small-Scale Fire Detection via Multi-Scale Channel Spatial Attention and Dynamic Upsampling},
  publisher={The Visual Computer}
}

# Requirements
```python  
pip install -r requirements.txt  
```





