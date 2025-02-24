# Code
* This is the code for DCF-YOLO. The specific innovative points are as follows: the code for the CAS module is in cas.py, the DySample module is in dysample.py, and the Focaler-IoU loss function is included in loss.py; the model configuration file is dcf-yolo.yaml.
* If you want to use our code, you must have the following preparation under the PyTorch framework: see requirement.txt for details
* Code Guidance: Download the dataset in the **Dataset** **link**, put the  images  into "firedatset/images" ,replace the common.py in the DCF-YOLO/models folder with the common.py in the DCF-YOLO folder, and replace the loss.py in the DCF-YOLO/utils folder with the loss.py in the DCF-YOLO folder,and then run the train.py to successfully train our model.

# DOI: 10.5281/zenodo.14880534

# Dataset

* You can obtain the complete dataset using the extraction code at the provided link, which has been divided into train, validation, and test sets as described in the paper. We have also provided both versions with labels and without labels in the link.

# Citation
@article{
  title={**Enhanced YOLOv7-tiny for Small-Scale Fire Detection via Multi-Scale Channel Spatial Attention and Dynamic Upsampling**},
}

# Requirements
```python  
pip install -r requirements.txt  
```

# Reference
*	Lin, X., Sun, S. Z., Huang, W. et al.: Eapt: Efficient attention pyramid transformer for image processing.IEEE Trans. Multimedia. (2023). https://doi.org/10.1109/tmm.2021.3120873.
*	Shen, B., Li, L., Hu, X. et al.: Point cloud upsampling generative adversarial network based on residual multi-scale off-set attention.Virtual Reality & Intelligent Hardware. (2023). https://doi.org/10.1016/j.vrih.2022.08.016.





