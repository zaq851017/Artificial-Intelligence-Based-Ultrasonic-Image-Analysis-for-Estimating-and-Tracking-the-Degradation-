# Artificial Intelligence-Based Ultrasonic Image Analysis for Estimating and Tracking the Degradation of Injection Laryngoplasty
# Usage
To start working on this assignment, you should clone this repository into your local machine by using the following command.

### Dataset
In the starter code of this repository, we have provided a shell script for downloading and extracting the dataset for this assignment. For Linux users, simply use the following command.
    
    bash ./get_dataset.sh
The shell script will automatically download the dataset and store the data in a folder called `Medcial_data`. Note that this command by default only works on Linux. If you are using other operating systems, you should download the dataset from [this link](https://drive.google.com/file/d/1c4nEjrUISeSl7LEf9VUmkpKwvdx3fnuj/view?usp=sharing) and unzip the compressed file manually.

### Training

    bash train.sh
The shell script will automatically setup hyperparmeters for model.
### Testing

    bash test.sh
This shell script will download examples of ultrasound video and corresponding HA height in input_video folder. The code will convert ultrasound video in input_video folder to image sequence in output_frame folder. Finally, the code will use these image sequences to predict the HA masks and estimate the HA volume.

> ⚠️ ***Folder Explanation*** ⚠️  
* input_video : the input of ultrasound video and corresponding HA height
* output_frame : the original image sequence output
* output_prediction : the prediction resluts and estimated HA volume.
