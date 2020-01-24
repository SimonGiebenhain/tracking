# Multiple Object Tracking with Point Patterns

### This work was done as my Bachelor Project.
 
## Pocket Guide to this repository

### Kalman Filter Approach
Code regarding the Kalman Filter approach is split up into multiple folders. The folder names are self explanatory and these files are partially commented.
The most notable folders are **MultipleObjectTracking** and **SingleObjectTracking**. In the following I explain the most important files shortly.

**/MultipleObjectTracking:**
This folder holds all function regarding the management of multiple tracks. The most important functions herein are:
 - *ownMOT.m:* This function does the complete MOT. All relevant functions are accessed in some way from this file.
 - *createNewTracks.m:* This function does some pattern matching to determine whether new tracks can be initialized.
 - *birdsMOTstreamlined.m:* This script does MOT in a user friendly way. You just have to specify the file with the raw, unlabelled detections and a folder which defines the patterns of birds.
 
**/SingleObjectTracking:**
This folder holds all function regarding the tracking of a single object. The most important functions herein are:
 - *predictKalman.m:* The predict step of the Kalman Filter 
 - *correctKalman.m:* The correct step. However this makes use of *match_patterns.m* in order to find a matching between the detections and markers.
 - *setupKalman.m:* Constructs a new Kalman Filter "objcet"
 - *getMeasurementFunction.m:* Constructs the measurement function (mapping from state space to measurement space) for a given pattern.
 
**/Patterns:**
This folder holds functions regardin the patterns and matching of patterns. The most important functions herein are:
 - *match_patterns.m:* Implements a pattern matching algorithm both relying on the prediction of the KF and rotation and position invariant information from the pattern. 
 - *umeyama.m:* Method to calculate the rotation between two point clouds with known correspondences.
 

 
The Code regarding the deep learning approach can be found in the **ModernMethods** folder.

## Results

### SOT in a noise free setting
See the neural network results on the left and the Kalman Filter results on the right
![](GIFS/NN_noise_free.gif)  ![](GIFS/KF_noise_free.gif)

### SOT in a noisy setting
![](GIFS/NN_noisy.gif)  ![](GIFS/KF_noisy.gif)

### MOT results
![](GIFS/closeup_MOT.gif)  ![](GIFS/MOT.gif)

