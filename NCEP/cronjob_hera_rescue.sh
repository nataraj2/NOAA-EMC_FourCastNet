#!/bin/bash --login

echo "Job 1 prepdata is running"
sh /scratch1/NCEPDEV/nems/Linlin.Cui/Tests/FourCastNetv2/realtime_forecast/fcn_prepdata_hera.sh $1
sleep 60  # Simulating some work
echo "Job 1 prepdata completed"

echo "Job 2 runfcst is running"
job2_id=$(sbatch /scratch1/NCEPDEV/nems/Linlin.Cui/Tests/FourCastNetv2/realtime_forecast/fcn_runfcst_hera_cpu.sh $1 | awk '{print $4}')

# Wait for job 2 to complete
while squeue -j $job2_id &>/dev/null; do
    sleep 5  # Adjust the polling interval as needed
done
sleep 5  # Simulating some work
echo "Job 2 runfcst completed"

echo "Job 3 upload data to s3 is running"
sh /scratch1/NCEPDEV/nems/Linlin.Cui/Tests/FourCastNetv2/realtime_forecast/fcn_datadissm_hera.sh $1
sleep 5
echo "Job 3 upload data to s3 completed"
