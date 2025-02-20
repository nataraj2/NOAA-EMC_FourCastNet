#!/bin/bash

echo $1

# Activate Conda environment
source /scratch1/NCEPDEV/nems/Linlin.Cui/miniforge3/etc/profile.d/conda.sh
conda activate graphcast

cd /scratch1/NCEPDEV/nems/Linlin.Cui/Tests/FourCastNetv2/realtime_forecast

start_time=$(date +%s)

python3 gdas.py $1

end_time=$(date +%s)  # Record the end time in seconds since the epoch

# Calculate and print the execution time
execution_time=$((end_time - start_time))
echo "Execution time for gdas.py: $execution_time seconds"
