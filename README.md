# NCEP Implementation of FourCastNet model using GDAS data as ICs
We use ai_models_fourcastnetv2 plugin developed by ECMWF to run the model, which is FoureCastNet v2-small (https://arxiv.org/abs/2306.03838).

## Prerequisites
The following packages are needed:
- numpy
- boto3
- xarray
- pygrib
- torch
- ai-models-fourcastnetv2

## Run the model

### Download pre-computed stats

`wget https://get.ecmwf.int/repository/test-data/ai-models/fourcastnetv2/small/weights.tar`

`wget https://get.ecmwf.int/repository/test-data/ai-models/fourcastnetv2/small/global_means.npy`

`wget https://get.ecmwf.int/repository/test-data/ai-models/fourcastnetv2/small/global_stds.npy`

### Get input data (initial conditions) from GDAS product
The script ncep/gdas.py can be used to prepare input data, simply use:

`python gdas.py YYYYMMDDHH <> -s <s3 or nomads>`

file `input_YYYYMMDDHH.npy` will be created and saved in the current firectory.

### Run FourCastNetv2 in inference mode
Run inference using
`python inference.py -w </path/to/weightsandstats> -i </path/to/input/input_YYYYMMDDHH.npy> -o </path/to/output/> -l <forecast-hours>`


