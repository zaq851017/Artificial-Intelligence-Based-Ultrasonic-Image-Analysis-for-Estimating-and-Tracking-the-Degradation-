# Artificial Intelligence-Based Ultrasonic Image Analysis for Estimating and Tracking the Degradation of Injection Laryngoplasty
# Usage
To start working on this work, you should clone this repository into your local machine by using the following command.

### Dataset and Example Input Ultrasound Video
In the starter code of this repository, we have provided a shell script for downloading and extracting the dataset for this assignment. For Linux users, simply use the following command.
    
    bash ./get_dataset.sh
The shell script will automatically download the dataset and store the data in a folder called `Medcial_data` and the example ultrasound throat video in a folder called `input_video`. Note that this command by default only works on Linux. If you are using other operating systems, you should download the dataset from [this link](https://www.dropbox.com/s/0eywff6askh1wq3/Medical_data.zip?dl=0) and example input ultrasound video [this link](https://www.dropbox.com/s/83yo5tivkojvcom/input_video.zip?dl=0) and unzip the compressed file manually.

### Packages
For a list of packages you are allowed to import in this assignment, please refer to the requirments.txt for more details.

You can run the following command to install all the packages listed in the requirements.txt:

    pip3 install -r requirements.txt

### Training

    bash ./train.sh
The shell script will automatically setup hyperparmeters for model.
> ⚠️ ***Parameters Explanation*** ⚠️  
> *  `whch_model`: the all model illustrated in all_model.py
> + `FCN`, `UNET`, `UNET++`, `PSPNET`, `LINKNET`, `DEEPLABV3+`, `VNET`, `TCSNet`
> * `continuous`: `0` for single-image segmentation model and `1` for temporal-image segmentation
> * `continue_num`: whcih frame index as input such as `[-3, -2, -1, 0, 1, 2, 3]`
> * `gamma`: for Temporal Loss Weight Function expotent
### Testing

    bash ./test.sh
This shell script will download examples of ultrasound video and corresponding HA height in input_video folder. The code will convert ultrasound video in input_video folder to image sequence in output_frame folder. Finally, the code will use these image sequences to predict the HA masks and estimate the HA volume.

> ⚠️ ***Folder Explanation*** ⚠️  
>  * `input_video` : the input of ultrasound video and corresponding HA height
> * `output_frame` : the original image sequence frames
> * `output_prediction` : the prediction masks and estimated HA volume.

### Results
![image](https://github.com/zaq851017/Artificial-Intelligence-Based-Ultrasonic-Image-Analysis-for-Estimating-and-Tracking-the-Degradation-/blob/main/img3.png)
![image](https://github.com/zaq851017/Artificial-Intelligence-Based-Ultrasonic-Image-Analysis-for-Estimating-and-Tracking-the-Degradation-/blob/main/img1.png)
![image](https://github.com/zaq851017/Artificial-Intelligence-Based-Ultrasonic-Image-Analysis-for-Estimating-and-Tracking-the-Degradation-/blob/main/img2.png)

