Preparing inputs from GDAS product
=======================================================
FourCastNetGFS takes one state of the weather as the initial condition with 73 channel dataset. The order of these 73 channels is as following: 
10u, 10v, 100u, 100v, 2t, sp, msl, pwat, followed by u---, v---, z---, t---, and r--- at pressure levels 50, 100, 150, 200, 250, 300, 400, 500,
600, 700, 850, 925, 1000 hPa.
The input file is in NumPy `npy` format, which can be created with script ncep/gdas.py::

    python gdas.py startdate -s <s3 or nomads> -m <wgrib2 or pygrib> -k <yes or no>

**Arguments**

Requried:

    startdate: string, yyyymmddhh

Optional:

    *-s* or *--source*: s3 or nomads, the sourece to download gdas data (default: s3) 

    *-m* or *--method*: wgrib2 or pygrib, the method to extract required variables (default: wgrib2) 

    *-k* or *--keep*: yes or no, whether to keep downloaded data after processed (default: no) 
