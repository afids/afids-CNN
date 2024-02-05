# afids-CNN
Leveraging the recent release of the anatomical fiducial framework for developing an open software infrastructure to solve the landmark regression problem on 3D MRI images

## Preparation 
1- install poetry and configure cache directory
2- poetry install and shell to activate environment 

## Processing imaging data for training can be found in the following repo (
1 - rigid registraion to MNI template 
2 - conforming image to 1mm iso res
3 - intensity normalization (i.e., WM to 110) followed by minmax norm

## Processing landmark data (AFIDs)
1 - extract points from landmark file (.fcsv is supported)
2 - extact a landmark Euclidean distance map (could be considered probability map; each voxel communicates the distance to a AFID of interest) 

## Machine learning 
1 - a standard 3D Unet


