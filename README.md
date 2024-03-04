# NCEP Implementation of FourCastNet model using GDAS data as ICs
We use ECMWF's [ai_models_fourcastnetv2](https://github.com/ecmwf-lab/ai-models-fourcastnetv2/tree/main) plugin to run the model. FoureCastNet v2-small applies Spherical Fourier Neural Operators ([SFNOs](https://arxiv.org/abs/2306.03838)) as neural network architecture.

## Prerequisites
The following packages are needed:
- numpy
- boto3
- xarray
- pygrib
- torch
- ai-models-fourcastnetv2
- iris
- iris_grib
- eccodes
- cf_units

## Run the model

### Download pre-computed stats

```bash
wget https://get.ecmwf.int/repository/test-data/ai-models/fourcastnetv2/small/weights.tar
```

```bash
wget https://get.ecmwf.int/repository/test-data/ai-models/fourcastnetv2/small/global_means.npy
```

```bash
wget https://get.ecmwf.int/repository/test-data/ai-models/fourcastnetv2/small/global_stds.npy
```

### Get input data (initial conditions) from GDAS product
The script ncep/gdas.py can be used to prepare input data, simply use:

```bash
python gdas.py YYYYMMDDHH -s <s3 or nomads> -m <wgrib2 or pygrib> -k <yes or no>
```

file `input_YYYYMMDDHH.npy` will be created and saved in the current firectory.

### Run FourCastNetv2 in inference mode
Run inference using

```bash
python inference.py YYYYMMDDHH -w </path/to/weightsandstats> -i </path/to/input/input_YYYYMMDDHH.npy> -o </path/to/output/> -l <forecast-hours>
```

### Output
The forecast results will be saved in GRIB2 format.

