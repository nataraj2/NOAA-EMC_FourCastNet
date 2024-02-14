## NCEP Implementation of FourCastNet model using GDAS data as ICs
We use ai_models_fourcastnetv2 plugin developed by ECMWF to run the model, which is FoureCastNet v2-small (https://arxiv.org/abs/2306.03838).

### Download pre-computed stats
wget https://get.ecmwf.int/repository/test-data/ai-models/fourcastnetv2/small/weights.tar

wget https://get.ecmwf.int/repository/test-data/ai-models/fourcastnetv2/small/global_means.npy

wget https://get.ecmwf.int/repository/test-data/ai-models/fourcastnetv2/small/global_stds.npy

### Prerequisites
The following packages are needed:
- numpy
- boto3
- xarray
- pygrib
- torch
- ai-models-fourcastnetv2
