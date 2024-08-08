#!/bin/bash

module use /scratch1/NCEPDEV/nems/role.epic/spack-stack/spack-stack-1.6.0/envs/unified-env-rocky8/install/modulefiles/Core
module load stack-intel
module load awscli-v2

home_dir=/scratch1/NCEPDEV/nems/Linlin.Cui/Tests/FourCastNetv2/realtime_forecast

fcst_date=$(echo $1 | cut -c1-8)
fcst_cycle=$(echo $1 | cut -c9-10)

cd $home_dir
aws s3 sync fcngfs.${fcst_date}/$fcst_cycle s3://noaa-nws-fourcastnetgfs-pds/fcngfs.${fcst_date}/$fcst_cycle

#Delete data locally
#rm -rf fcngfs.${fcst_date}/$(printf "%02d" $fcst_cycle)
rm -rf fcngfs.${fcst_date}

#Unload modules
module purge
