# Neural Network Algorithm to Guess Famous Locations

## SRC:
### Installing and Building your code
### Usage of your Code

## DATA:
### Data Dictionaries:
| NAME        | CONTENTS    |
| :---        |    :----:   |
| Landmark_ids.csv       | Key-Value dictionary to transfer from landmark IDs to landmark names for visualization purposes                        |
| index_image_to_landmark.csv      | 16 character image id to landmark id                                                                         |
| index_label_to_category.csv      | Landmark id (from above) to URL of location information                                                      |
| select_landmarks.csv             | ***NEEDS TO BE UPDATED*** all landmarks within the above dictionary, along with image IDs and landmark IDs   |
| top1000.csv                      | Top 1000 landmarks in terms of number of images-- used when selecting the 35 landmarks for the model         |

### Link to Original Data Set Github:
### https://github.com/cvdfoundation/google-landmark
We used a Python script in order to merge two of the data sets regarding image IDs and their landmarks. With this, we then found a count of all landmarks and sorted from greatest to least. We noticed, however, that many of the locations with the most images were simply military bases in the United States and around the world. Thus, we chose 35 hand-picked locations to use in the model, to try and make the project both interesting, while still having enough data to make the model work correctly. We purposefully chose different counts of images though, to see how the model reacts to having different amounts of training data (with a general range of about ~30 to ~1000 images for training). 

## 
