'''
Revision history:
    -20240213: Linlin Cui, adapt graphcast's gdas_utility.py to prepare input files for FourCastNetv2
'''

import os
import sys
from time import time
import argparse
import subprocess
from datetime import datetime, timedelta

import boto3
import xarray as xr
import numpy as np
from botocore.config import Config
from botocore import UNSIGNED
import pygrib

def get_dataarray(grbfile, var_name, level_type, desired_level):

    # Find the matching grib message
    variable_message = grbfile.select(shortName=var_name, typeOfLevel=level_type, level=desired_level)

    if len(variable_message) > 2:
        data = []
      
        for message in variable_message:
            print(message)
            data.append(message.values)
        data = np.array(data)
    else:
        data = variable_message[0].values

    return data.astype(np.float32)


class GFSDataProcessor:
    """ Download GDAS data from either s3 bucket or nomads and extract variables

    """

    def __init__(self, start_datetime, download_source='nomads', output_directory=None, download_directory=None, keep_downloaded_data=True):

        self.start_datetime = start_datetime
        self.download_source = download_source
        self.output_directory = output_directory
        self.keep_downloaded_data = keep_downloaded_data

        if self.output_directory is None:
            self.output_directory = os.getcwd()

        # Specify the local directory where you want to save the files
        if download_directory is None:
            self.local_base_directory = os.path.join(os.getcwd(), 'noaa-gfs-bdp-pds-data')  # Use current directory if not specified
        else:
            self.local_base_directory = os.path.join(download_directory, 'noaa-gfs-bdp-pds-data')

        
        # Define the local directory path where the file will be saved
        self.local_directory = os.path.join(self.local_base_directory, self.start_datetime.strftime("%Y%m%d"), self.start_datetime.strftime("%H"))
        os.makedirs(self.local_directory, exist_ok=True)

        # Define downloaded file name 
        self.out_filename = f'{self.local_directory}/gdas.t{self.start_datetime.strftime("%H")}z.pgrb2.0p25.f000'

        if not os.path.isfile(self.out_filename):
            self.download_data()
    
    def s3bucket(self):
        s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    
        # Specify the S3 bucket name and root directory
        bucket_name = 'noaa-gfs-bdp-pds'
        root_directory = 'gdas'

        # Construct the S3 prefix for the directory
        s3_prefix = f'{root_directory}.{self.start_datetime.strftime("%Y%m%d")}/{self.start_datetime.hour:02d}/atmos'
        obj_key = f'{s3_prefix}/gdas.t{int(self.start_datetime.hour):02d}z.pgrb2.0p25.f000'

        with open(self.out_filename, "wb") as f:
            s3.download_fileobj(bucket_name, obj_key, f)
        print(f"Downloaded {obj_key} to {self.out_filename}")
    
    def nomads(self):

        gdas_url = f'https://nomads.ncep.noaa.gov/cgi-bin/filter_fnl.pl?dir=%2Fgdas.{self.start_datetime.strftime("%Y%m%d")}%2F{self.start_datetime.strftime("%H")}%2Fatmos&file=gdas.t{self.start_datetime.strftime("%H")}z.pgrb2.1p00.f000&var_HGT=on&var_PRMSL=on&var_SPFH=on&var_TMP=on&var_UGRD=on&var_VGRD=on&lev_2_m_above_ground=on&lev_10_m_above_ground=on&lev_1000_mb=on&lev_925_mb=on&lev_850_mb=on&lev_700_mb=on&lev_600_mb=on&lev_500_mb=on&lev_400_mb=on&lev_300_mb=on&lev_250_mb=on&lev_200_mb=on&lev_150_mb=on&lev_100_mb=on&lev_50_mb=on&lev_mean_sea_level=on'

        # Download the file from S3 to the local path
        try:
            # Run the wget command
            subprocess.run(['wget', gdas_url, '-O', self.out_filename], check=True)
            print(f"Download completed: {gdas_url} => {self.out_filename}")
        except subprocess.CalledProcessError as e:
            print(f"Error downloading {gdas_url}: {e}")
        
    def download_data(self):

        if self.download_source == 's3':
            self.s3bucket()
        else:
            self.nomads()

        print("Download completed.")

    def get_data(self, method='wgrib2'):
        if method == "wgrib2":
          self.process_data_with_wgrib2()
        elif method == "pygrib":
          self.process_data_with_pygrib()
        else:
          raise NotImplementedError(f"Method {method} is not supported!")

    def process_data_with_wgrib2(self):

        variables_to_extract = {
            'surface': {
                ':UGRD:': {
                    'levels': [':10 m above ground:'],
                },
                ':VGRD:': {
                    'levels': [':10 m above ground:'],
                },
                ':UGRD:': {
                    'levels': [':100 m above ground:'],
                },
                ':VGRD:': {
                    'levels': [':100 m above ground:'],
                },
                ':TMP:': {
                    'levels': [':2 m above ground:'],
                },
                ':PRES:': {
                    'levels': [':surface:'],
                },
                ':PRMSL:': {
                    'levels': [':mean sea level:'],
                },
                ':CWAT:': {
                    'levels': [':entire atmosphere:considered as a single layer:'],
                },
            },
            'upper': {
                ':UGRD:': {
                    'levels': [':(50|100|150|200|250|300|400|500|600|700|850|925|1000) mb:'],
                },
                ':VGRD:': {
                    'levels': [':(50|100|150|200|250|300|400|500|600|700|850|925|1000) mb:'],
                },
                ':HGT:': {
                    'levels': [':(50|100|150|200|250|300|400|500|600|700|850|925|1000) mb:'],
                },
                ':TMP:': {
                    'levels': [':(50|100|150|200|250|300|400|500|600|700|850|925|1000) mb:'],
                },
                ':RH:': {
                    'levels': [':(50|100|150|200|250|300|400|500|600|700|850|925|1000) mb:'],
                },
            }
        }

        print("Start extracting variables and associated levels from grib2 files:")

        grib2_file = self.out_filename

        data = []
        for level_type, variable_data in variables_to_extract.items():


            for variable, value in variable_data.items():
                levels = value['levels'][0]

                if level_type == 'surface':
                    varname = ''.join(e for e in variable if e.isalnum()) + "_" + ''.join(e for e in levels if e.isalnum())
                else:
                    varname = ''.join(e for e in variable if e.isalnum())

                if variable == ":CWAT:":
                    varname = 'CWAT_entireatmosphere_consideredasasinglelayer_'
        
                # Extract the specified variables with levels from the GRIB2 file
                output_file = f'{variable}_{levels}.nc'
                #output_file = f'{varname}.nc'

                # Use wgrib2 to extract the variable with level
                if variable == ":CWAT:": 
                    wgrib2_command = ['wgrib2', grib2_file, '-match', f'{variable}', '-netcdf', output_file]
                else:
                    wgrib2_command = ['wgrib2', '-nc_nlev', '13', grib2_file, '-match', f'{variable}', '-match', f'{levels}', '-netcdf', output_file]

                #wgrib2_command = ['wgrib2', grib2_file, '-match', f'{variable}', '-match', f'{levels}', '-netcdf', output_file]
                subprocess.run(wgrib2_command, check=True)

                # Open the extracted netcdf file as an xarray dataset

                ds = xr.open_dataset(output_file)
                values = np.squeeze(ds[varname]).values.astype(np.float32)

                ##units conversion
                #if variable == ":HGT:":
                #    values = values * 9.80665

                if level_type == 'upper':
                    for ilev in range(13):
                        data.append(values[ilev, ::-1, :]) #reverse latitude
                else:
                    values = values[::-1, :] #reverse latidue
                    data.append(values)

                ds.close()
                os.remove(output_file)
     
        print(np.array(data).shape)
        with open(f'{self.output_directory}/input_{self.start_datetime.strftime("%Y%m%d%H")}.npy', 'wb') as f:
            np.save(f, np.array(data))


    def process_data_with_pygrib(self):

        levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

        variables_to_extract = {
            'surface': {
                '10u': {
                    'typeOfLevel': 'heightAboveGround',
                    'level': 10,
                },
                '10v': {
                    'typeOfLevel': 'heightAboveGround',
                    'level': 10,
                },
                '100u': {
                    'typeOfLevel': 'heightAboveGround',
                    'level': 100,
                },
                '100v': {
                    'typeOfLevel': 'heightAboveGround',
                    'level': 100,
                },
                '2t': {
                    'typeOfLevel': 'heightAboveGround',
                    'level': 2,
                },
                'sp': {
                    'typeOfLevel': 'surface',
                    'level': 0,
                },
                'prmsl': {
                    'typeOfLevel': 'meanSea',
                    'level': 0,
                },
                'cwat': {
                    'typeOfLevel': 'atmosphereSingleLayer',
                    'level': 0,
                },

            },
            'upper': {
                'u': {
                    'typeOfLevel': 'isobaricInhPa',
                    'level': levels,
                },
                'v': {
                    'typeOfLevel': 'isobaricInhPa',
                    'level': levels,
                },
                'gh': {
                    'typeOfLevel': 'isobaricInhPa',
                    'level': levels,
                },
                't': {
                    'typeOfLevel': 'isobaricInhPa',
                    'level': levels,
                },
                'r': {
                    'typeOfLevel': 'isobaricInhPa',
                    'level': levels,
                },
            }
        }

        grbs = pygrib.open(self.out_filename)

        data = []
        for level_type, variable_data in variables_to_extract.items():
            for variable, value in variable_data.items():

                levelType = value['typeOfLevel']
                desired_level = value['level']
            

                print(f'Get variable {variable} from file {self.out_filename}:')
                if level_type == 'surface':
                    values = get_dataarray(grbs, variable, levelType, desired_level)
                    data.append(values)
                else:
                    for level in desired_level:
                        values = get_dataarray(grbs, variable, levelType, level)
                        data.append(values)

                ##units conversion
                #if variable == 'gh':
                #    values = values * 9.80665

     
        with open(f'{self.output_directory}/input_{self.start_datetime.strftime("%Y%m%d%H")}.npy', 'wb') as f:
            np.save(f, np.array(data))
        

    def remove_downloaded_data(self):
        # Remove downloaded data from the specified directory
        print("Removing downloaded grib2 data...")
        try:
            os.system(f"rm -rf {self.local_base_directory}")
            print("Downloaded data removed.")
        except Exception as e:
            print(f"Error removing downloaded data: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and process GDAS data")
    parser.add_argument("start_datetime", help="Start datetime in the format 'YYYYMMDDHH'")
    parser.add_argument("-m", "--method", help="method to extact variables from grib2, options: wgrib2, pygrib", default="wgrib2")
    parser.add_argument("-s", "--source", help="the source repository to download gdas grib2 data, options: nomads (up-to-date), s3", default="s3")
    parser.add_argument("-o", "--output", help="Output directory for processed data")
    parser.add_argument("-d", "--download", help="Download directory for raw data")
    parser.add_argument("-k", "--keep", help="Keep downloaded data (yes or no)", default="no")

    args = parser.parse_args()

    start_datetime = datetime.strptime(args.start_datetime, "%Y%m%d%H")
    download_source = args.source
    method = args.method
    output_directory = args.output
    download_directory = args.download
    keep_downloaded_data = args.keep.lower() == "yes"

    data_processor = GFSDataProcessor(start_datetime, download_source, output_directory, download_directory, keep_downloaded_data)
    data_processor.get_data(args.method)
    
    # remove downloaded data
    if keep_downloaded_data:
        data_processor.remove_downloaded_data()
