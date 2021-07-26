# Artificial Intelligence-Based Ultrasonic Image Analysis for Estimating and Tracking the Degradation of Injection Laryngoplasty
# Usage
To start working on this work, you should clone this repository into your local machine by using the following command.

### Dataset and Example Input Ultrasound Video
In the starter code of this repository, we have provided a shell script for downloading and extracting the dataset for this assignment. For Linux users, simply use the following command.
    
    bash ./get_dataset.sh
The shell script will automatically download the dataset and store the data in a folder called `Medcial_data` and the example ultrasound throat video in a folder called `input_video`. Note that this command by default only works on Linux. If you are using other operating systems, you should download the dataset from [this link](https://drive.google.com/file/d/12Ewhd5MMRF5I9NT0E6FVoFTz3h4TQt1L/view?usp=sharing) and example input ultrasound video [this link](https://drive.google.com/file/d/1ccKv8zfjZwU10m491T4b89IKJh_rg6FH/view?usp=sharing) and unzip the compressed file manually.

### Training

    bash ./train.sh
The shell script will automatically setup hyperparmeters for model.
> ⚠️ ***Parameters Explanation*** ⚠️  
> *  `whch_model`: the all model illustrated in all_model.py
> * `continuous`: 0 for single-image segmentation model and 1 for temporal-image segmentation
> * `continue_num`: whcih frame index as input such as [-3, -2, -1, 0, 1, 2, 3]
> * `gamma`: $r$ for Temporal Loss weight $w(d) = (1 - \frac{|d|}{2 \times \rho })^r$
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

