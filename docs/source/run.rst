Run FourCastNetGFS
=======================================================
We use weights and normalization statistics archived at ECMWF to run FourCastNet in inference mode. These files can be downloaded at::

    wget https://get.ecmwf.int/repository/test-data/ai-models/fourcastnetv2/small/weights.tar
    wget https://get.ecmwf.int/repository/test-data/ai-models/fourcastnetv2/small/global_means.npy
    wget https://get.ecmwf.int/repository/test-data/ai-models/fourcastnetv2/small/global_stds.npy

Run the model in inference mode::

    python inference.py startdate -w </path/to/weightsandstats> -i </path/to/input/input_YYYYMMDDHH.npy> -o </path/to/output/> -l <forecast-hours>

**Arguments**

Requried:

    startdate: string, yyyymmddhh

Optional:

    *-w* or *--weights*: path to downloaded files (weights.tar, global_means.npy, global_stds.npy)

    *-i* or *--input*: path to the input file

    *-o* or *--output*: path to save output files

    *-l* or *--length*: lead time in hours
