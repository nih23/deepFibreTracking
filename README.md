#  deepFibreTracking - README

## TODO
This documentation is far from complete. It is missing 

1. the details of what each script does
2. the removal of redundancies in the code, caused by the low modularity
3. further details on the use of the code
4. the results so far 


## General Information
This project is the codebase of the BeLL, which - according to the current status - is executable. The horovod required for Taurus is currently commented out to keep the code executable on the workstation. The changes are minimal, however, as the code only requires the number of the current node, which is now replaced by 0.

## Utilization
The code is able to first generate suitable training data, prepare it and put it into a suitable format and finally train a number of networks on it and review the results. At the moment, however, the modularization of the code is low. Compared to the first executable version, the whole thing is now programmed object-oriented, but it is not a finished pipeline for tractography, because some settings in the code are hardcoded. The function of the individual components of the pipeline is explained in the following, however, a general approch for training and evaluating a model could be

1. **Generate** Training Data with  ```python 00_generateTrainingData.py -nx 3 -ny 3 -nz 3 --noStreamlines 10000 csd ```
2. **Generate** Padded Training Data with ```python 00_generate_paddedData.py -input [DATA_FILE]```
3. **Train** a random model with ```python 01_randomModel_training.py```, which is getting saved to local folder `models/`
4. Manually **paste** it into the `03_ismmr2015_fa.py` routine and **start tracking** with  `python 03_ismrm2015_fa.py --faMask -threshold 0.15`
### 1. 00_generateTrainingData.py (Nico's Code)
Generates nerve tracts based on given tractogram data. It was executed in the pipeline with the following parameters:

> ```python 00_generateTrainingData.py -nx 3 -ny 3 -nz 3 --noStreamlines 10000 csd [--rotateData]```

The general syntax is ```python 00_generateTrainingData.py [options] <tensorModel>```

All usable options are listed here:

| option                                       | description                                                                                                                   | value                 | default value | optional            |
| -------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- | --------------------- | ------------- | ------------------- |
| **tensorModel**                              | the tensor model in use                                                                                                       | [dti, csd]            |               | :x:                 |
| **-nx** <value>                              | number of voxels in x direction for each DWI data point                                                                       | integer               | 1             | :heavy_check_mark:  |
| **-ny** <value>                              | number of voxels in y direction for each DWI data point                                                                       | integer               | 1             | :heavy_check_mark:  |
| **-nz** <value>                              | number of voxels in z direction for each DWI data point                                                                       | integer               | 1             | :heavy_check_mark:  | 
| **-b** <value>                               | B-value to use for the DWI-data                                                                                               | integer               | 1000          | :heavy_check_mark:  |
| **-fa** <value>                              | FA-Threshhold to stop tracking                                                                                                | float                 | 0.15          | :heavy_check_mark:  |
| **-sw** <value>                              | step width for tracking                                                                                                       | float                 | 1.0           | :heavy_check_mark:  |
| **-spc** <value>                             | grid Spacing in Pixels/IJK                                                                                                    | float                 | 1.0           | :heavy_check_mark:  |
| **-mil** <value>                             | minimal length of streamline in mm                                                                                            | integer               | 40            | :heavy_check_mark:  |
| **-mal** <value>                             | maximal length of streamline in mm                                                                                            | integer               | 200           | :heavy_check_mark:  |
| **-repr** <value>                            | the target data representation: raw, spherical harmonics, resampled to 100 directions or 16x16 2D resampling (256 directions) | [raw, sh, res100, 2D] | res100        | :heavy_check_mark:  |
| **-sh** <value>                              | order of spherical harmonics is use is specified in `-repr`                                                                   | integer               | 8             | :heavy_check_mark:  |
| **--noStreamlines** <value>                  | choose a subset of n streamlines taken randomly from the generated streamlines                                                | integer               | everything    | :heavy_check_mark:  |
| **--rotateData**                             | rotate the DWI-data with respect to unit tangent, makes prediction easier                                                     | boolean               | false         | :heavy_check_mark:  |
| **--unitTanget**                             | *TODO (unknown)*                                                                                                              | boolean               | false         | :heavy_check_mark:  |
| **--visStreamlines**                         | Visualize the streamlines ahead of data generation in VTK Renderer                                                            | boolean               | false         | :heavy_check_mark:  |
| **--ISMRM2015data**                          | Use the training data of the ISMRM 2015 Dataset instead                                                                       | boolean               | false         | :heavy_check_mark:  |
| **--HCPid** <value>                          | Generate data based on HPC Dataset with given ID                                                                              | integer               | 100307        | :heavy_check_mark:  |
| **--precomputedStreamlines** <value>         | Use precomputed streamlines instead of newly generated. Value should be the file path to VTK file                             | string                | -             | :heavy_check_mark:  |
| **--addRandomDataForEndpointPrediction**     | Generate additional random points out of any streamlines to fix inbalance in endpoint prediction labels                       | boolean               | false         | :heavy_check_mark:  |
| [**-nt**/**--noThreads**] <value>            | Specify number of threads used to generate streamlines. Note that this also increases memory demand.                          | integer               | 4             | :heavy_check_mark:  |

The stored data consists of three different tensors of the following format with the following contents:

| key                          | description                                                                | dimensions        |
| ---------------------------- |----------------------------------------------------------------------------| ----------------- |
| **train_DWI**                | the DWI data for each streamline point, in current configuration 3³ voxels | [N, 3, 3, 3, 100] |
| **train_NextFibreDirection** | the matching directions for each streamline point, same indizes as above   | [N, 3]            |
| **streamlineIndices**        | the start index of the Nth streamline, usable to split streamlines         | [M, 3]        |


### 2. 00_generate_paddedData.py

Converts the data generated by ```python 00_generateTrainingData.py -nx 3 -ny 3 -nz 3 --noStreamlines 10000 csd [--rotateData]``` into a format suitable for training by dividing the data into individual trajectories and padding them.

The syntax is: ```python 00_generate_paddedData.py [options]```.
The options are the following:

| option                | description                                                                                               | value   | default value     | optional           |
| --------------------- | --------------------------------------------------------------------------------------------------------- | ------- | ----------------- | ------------------ |
|**-mal** <value>       | Specifies the max length of a single streamline generated by `00_generateTrainingData.py`                 | integer | 201               | :heavy_check_mark: |
|**-input** <value>     | Sets the input file to use for training. Path depends on `00_generateTrainingData.py`'s output            | string  | `'./train/[...]'` | :heavy_check_mark: |
|**--useCustomPadding** | Indicates wether a padding using the inversed last direction vector instead of zero vector should be used | boolean | False             | :heavy_check_mark: |


The data is then saved in to ```./cache/data_ijk.pt```.
It consists of a `tuple` *(dwi, nextDirection, lengths)* containing the detailed information required for training.

| name          | description                                      | dimensions                    |
| ------------- | ------------------------------------------------ | ----------------------------- |
| dwi           | the DWI data for each point in each streamline   | [N, max length, 2700]         |
| nextDirection | the matching nextDirection in each streamline    | [N, max_length, 3]            |
| lengths       | the original lengths of each streamline in steps | [N]                           |


### 3. 01_trainModel.py

### 4. 01_randomModel_training.py

### 5. 02_unitTest_tracking.py

### 6. 03_ismrm2015.py

### 7. 03_ismrm2015_fa.py