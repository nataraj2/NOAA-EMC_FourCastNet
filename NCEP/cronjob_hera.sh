#!/bin/bash --login

# Get the UTC hour and calculate the time in the format yyyymmddhh
current_hour=$(date -u +%H)
current_hour=$((10#$current_hour))

if (( $current_hour >= 0 && $current_hour < 6 )); then
    datetime=$(date -u -d 'today 00:00')
elif (( $current_hour >= 6 && $current_hour < 12 )); then
    datetime=$(date -u -d 'today 06:00')
elif (( $current_hour >= 12 && $current_hour < 18 )); then
    datetime=$(date -u -d 'today 12:00')
else
    datetime=$(date -u -d 'today 18:00')
fi

# Calculate time 6 hours before
#curr_datetime=$(date -u -d "$time" +'%Y%m%d%H')
fcst_datetime=$( date -d "$datetime 12 hour ago" "+%Y%m%d%H" )

echo "Job 1 prepdata is running"
sh /scratch1/NCEPDEV/nems/Linlin.Cui/Tests/FourCastNetv2/realtime_forecast/fcn_prepdata_hera.sh $fcst_datetime
sleep 60  # Simulating some work
echo "Job 1 prepdata completed"

echo "Job 2 runfcst is running"
job2_id=$(sbatch /scratch1/NCEPDEV/nems/Linlin.Cui/Tests/FourCastNetv2/realtime_forecast/fcn_runfcst_hera_gpu.sh $fcst_datetime | awk '{print $4}')

# Wait for job 2 to complete
while squeue -j $job2_id &>/dev/null; do
    sleep 5  # Adjust the polling interval as needed
done
sleep 5  # Simulating some work
echo "Job 2 runfcst completed"

echo "Job 3 upload data to s3 is running"
sh /scratch1/NCEPDEV/nems/Linlin.Cui/Tests/FourCastNetv2/realtime_forecast/fcn_datadissm_hera.sh $fcst_datetime
sleep 5
echo "Job 3 uploading data to s3 completed"
