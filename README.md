# afids-NN
Utilizing the anatomical fiducals framework to identify other salient brain regions and automatic localization of anatomical fiducials using neural networks


# Processing data for training 

Convert3D

## Anatomical landmark data (AFIDs)

Convert3D:
1) .fcsv -> threshold image -> landmark distance map (could be considered probability map) 
2) distance map used for training 

## Structural T1w imaging 

Convert3D: 
1) brainmask.nii -> 3D patches sampled at x voxels 
2) matching of distance maps and anatomical imaging patches is crucial for proper training 


