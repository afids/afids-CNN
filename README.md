# afids-CNN
Leveraging the recent release of the anatomical fiducial framework for developing an open software infrastructure to solve the landmark regression problem on 3D MRI images

## Processing imaging data for training 
1 - skull stripping
2 - conforming image 
3 - intensity normalization (i.e., WM to 110)

## Processing landmark data (AFIDs)
1 - extract points from landmark file (.fcsv is supported)
2 - extact a landmark Euclidean distance map (could be considered probability map; each voxel communicates the distance to a AFID of interest) 

## Machine learning 
1 - a standard 3D Unet


