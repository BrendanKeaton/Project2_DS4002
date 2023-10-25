# Neural Network Algorithm to Guess Famous Locations
## SRC:
### Installing and Building your code
PD_EDA Requires Jupyter Notebook and Python in order to run. Additionally, change the .csv file paths for your computer.

Image Neural Network.py requires TensorFlow, PIL, and NumPy installed in Python to run. 

Final_Landmarks.py requires Jupyter Notebook and Python in order to run. Additionally, change the .csv file paths for your computer.

### Usage of your Code
PD_EDA: This notebook aims to get value counts for each landmark. We pre-selected 40 landmarks that we wanted to train our model with. We later found out that even though some of these are famous landmarks, there are not a lot of images for some, or they are under multiple names. Finding each and every label for our desired landmarks is a lot of work that we might not be able to do within the given time. After looking at our desired landmarks, we looked at the most common landmark IDs and ordered them by count.

Image Neural Network.py: This script is used to download the images, pre-process the data, and then run a basic Neural Network model on the images. 

Final_Landmarks: This notebook is functionally the same as PD_EDA. The difference is that this notebook shows the counts for the 35 landmarks we decided to use after exploratory data analysis. 

## DATA:
https://docs.google.com/spreadsheets/d/1C3bwJgl-Js3vmslL6Kx5R_t-O-dHueyx3v5wEyaLBZk/edit?usp=sharing
### Data Dictionaries:
Previously we used the following:
| NAME        | CONTENTS    |
| :---        |    :----:   |
| Landmark_ids.csv       | Key-Value dictionary to transfer from landmark IDs to landmark names for visualization purposes                        |
| index_image_to_landmark.csv      | 16 character image id to landmark id                                                                         |
| index_label_to_category.csv      | Landmark id (from above) to URL of location information                                                      |
| select_landmarks.csv             | all landmarks within the above dictionary, along with image IDs and landmark IDs   |
| top1000.csv                      | Top 1000 landmarks in terms of number of images-- used when selecting the 35 landmarks for the model         |
| image_dataset.csv                     | Our manually created dataset to create a basic version of the Neural Network that is in scope with the size of this project.         |

Once we realized how out of scope that data set was though, we collected our own 30 images of 8 locations to use that simply had a manually created image ID, along with the landmark name. 

### Link to Original Data Set Github:
### https://github.com/cvdfoundation/google-landmark
We used a Python script in order to merge two of the data sets regarding image IDs and their landmarks. With this, we then found a count of all landmarks and sorted from greatest to least. We noticed, however, that many of the locations with the most images were simply military bases in the United States and around the world. Thus, we chose 35 hand-picked locations to use in the model, to try and make the project both interesting, while still having enough data to make the model work correctly. We purposefully chose different counts of images though, to see how the model reacts to having different amounts of training data (with a general range of about ~30 to ~1000 images for training). 

## FIGURES:
### Figure 1: Counts of the 30 most common landmarks for original dataset:
![image](https://github.com/BrendanKeaton/Project2_DS4002/assets/100185367/9eade716-88e3-44dc-b81b-a1a3753cab00)

## REFERENCES:
"Google Landmarks Dataset v2 - A Large-Scale Benchmark for Instance-Level Recognition and Retrieval"
T. Weyand*, A. Araujo*, B. Cao, J. Sim
Proc. CVPR'20

"Optimization of Rank Losses for Image Retrieval"
E. Ramzi, N. Audebert, C. Rambour, A. Araujo, X. Bitot, N. Thome
In submission to: IEEE Transactions on Pattern Analysis and Machine Intelligence.
