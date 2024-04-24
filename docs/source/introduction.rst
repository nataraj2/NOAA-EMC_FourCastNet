Introduction
=======================================================
The FourCastNet Global Forecast System (FourCastNetGFS) is an experimental system set up by the National Centers for Environmental Prediction (NCEP) to
produce medium range global forecasts. The model runs on a 0.25 degree latitude-longitude grid (about 28 km) and 13 pressure levels. The model produces
forecasts 4 times a day at 00Z, 06Z, 12Z and 18Z cycles. Major atmospheric and surface fields including temperature, wind components, geopotential height, 
relative humidity and 2 meter temperature and 10 meter winds are available. The products are 6 hourly forecasts up to 10 days. The data format is GRIB2.

The FourCastNetGFS system is an experimental weather forecast model built upon the pre-trained Nvidia's FourCastNet Machine Learning Weather Prediction
(MLWP) model version 2. The FourCastNet (`Bonev et al. <https://arxiv.org/abs/2306.03838>`_) was developed by Nvidia using Adaptive Fourier Neural Operators. It uses a Fourier 
transform-based token-mixing scheme with the vision transformer architecture. This model is pre-trained with ECMWF ERA5 reanalysis data. 
The FourCastNetGFS takes one model state as initial condition from NCEP 0.25 degree GDAS analysis data and runs FourCastNet with weights 
from the pretrained FourCastNet by Nvidia. Unit conversion to the GDAS data is conducted to match the input data required by FourCastNet 
and to generate forecast products consistent to GFS.

