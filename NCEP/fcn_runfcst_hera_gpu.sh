#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=nems
#SBATCH --time=1:00:00 
#SBATCH --job-name=fcngfs
#SBATCH --output=output.txt
#SBATCH --error=error.txt
#SBATCH --partition=fge
#SBATCH --qos=gpuwf

# Activate Conda environment
source /scratch1/NCEPDEV/nems/Linlin.Cui/miniforge3/etc/profile.d/conda.sh
conda activate mlwp

cd /scratch1/NCEPDEV/nems/Linlin.Cui/Tests/FourCastNetv2/realtime_forecast

start_time=$(date +%s)
echo "start runing fourecastnet for: $1"

python3 inference.py $1 -w /scratch1/NCEPDEV/nems/Linlin.Cui/Tests/FourCastNetv2/weights -i input_$1.npy -o ./ -l 240

end_time=$(date +%s)  # Record the end time in seconds since the epoch

# Calculate and print the execution time
execution_time=$((end_time - start_time))
echo "Execution time for fourcastnet: $execution_time seconds"

# Generate grib2 index files
module use /scratch1/NCEPDEV/nems/role.epic/spack-stack/spack-stack-1.6.0/envs/unified-env-rocky8/install/modulefiles/Core
module load wgrib2

echo "Generate grib2 index files: "
fcst_date=$(echo $1 | cut -c1-8)
fcst_cycle=$(echo $1 | cut -c9-10)

for fname in `ls fcngfs.${fcst_date}/$fcst_cycle/fcngfs.*`; do
    wgrib2 -s $fname > $fname.idx
done

#Move input to corresponding folder
mv input_$1.npy fcngfs.${fcst_date}/$fcst_cycle

#Unload modules
module purge
