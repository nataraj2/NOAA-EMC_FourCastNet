import os
import argparse
from datetime import datetime, timedelta
import pathlib

import numpy as np
import torch
import ai_models_fourcastnetv2.fourcastnetv2 as nvs
#import fourcastnetv2 as nvs
import iris
from iris.cube import Cube
from iris.coords import DimCoord
import iris_grib
import eccodes
import cf_units

#device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device is {device}')

def tweaked_messages(cube):
    for cube, grib_message in iris_grib.save_pairs_from_cube(cube):
        eccodes.codes_set(grib_message, 'centre', 'kwbc')
        if cube.standard_name == 'air_pressure_at_sea_level':
            eccodes.codes_set(grib_message, 'discipline', 0)
            eccodes.codes_set(grib_message, 'parameterCategory', 3)
            eccodes.codes_set(grib_message, 'parameterNumber', 1)
            eccodes.codes_set(grib_message, 'typeOfFirstFixedSurface', 101)
        if cube.standard_name == 'surface_air_pressure':
            eccodes.codes_set(grib_message, 'discipline', 0)
            eccodes.codes_set(grib_message, 'parameterCategory', 3)
            eccodes.codes_set(grib_message, 'parameterNumber', 0)
            eccodes.codes_set(grib_message, 'typeOfFirstFixedSurface', 1)
        if cube.standard_name == 'precipitation_amount':
            eccodes.codes_set(grib_message, 'discipline', 0)
            eccodes.codes_set(grib_message, 'parameterCategory', 1)
            eccodes.codes_set(grib_message, 'parameterNumber', 3)
            eccodes.codes_set(grib_message, 'typeOfFirstFixedSurface', 200)

    yield grib_message

class FourCastNetv2:
    PARAM = {
        1: {'name': '10u', 'standard_name': 'x_wind', 'level': 10, 'units': 'm s**-1'},
        2: {'name': '10v', 'standard_name': 'y_wind', 'level': 10, 'units': 'm s**-1'},
        3: {'name': '100u', 'standard_name': 'x_wind', 'level': 100, 'units': 'm s**-1'},
        4: {'name': '100v', 'standard_name': 'y_wind', 'level': 100, 'units': 'm s**-1'},
        5: {'name': 't2m', 'standard_name': 'air_temperature', 'level': 2, 'units': 'K'},
        6: {'name': 'sp', 'standard_name': 'surface_air_pressure', 'level': 0, 'units': 'Pa'},
        7: {'name': 'msl', 'standard_name': 'air_pressure_at_sea_level', 'level': 0, 'units': 'Pa'},
        8: {'name': 'pwat', 'standard_name': 'precipitation_amount', 'level': 0, 'units': 'kg m**-2'},
        9: {'name': 'u50', 'standard_name': 'x_wind', 'level': 50, 'units': 'm s**-1'},
        10: {'name': 'u100', 'standard_name': 'x_wind', 'level': 100, 'units': 'm s**-1'},
        11: {'name': 'u150', 'standard_name': 'x_wind', 'level': 150, 'units': 'm s**-1'},
        12: {'name': 'u200', 'standard_name': 'x_wind', 'level': 200, 'units': 'm s**-1'},
        13: {'name': 'u250', 'standard_name': 'x_wind', 'level': 250, 'units': 'm s**-1'},
        14: {'name': 'u300', 'standard_name': 'x_wind', 'level': 300, 'units': 'm s**-1'},
        15: {'name': 'u400', 'standard_name': 'x_wind', 'level': 400, 'units': 'm s**-1'},
        16: {'name': 'u500', 'standard_name': 'x_wind', 'level': 500, 'units': 'm s**-1'},
        17: {'name': 'u600', 'standard_name': 'x_wind', 'level': 600, 'units': 'm s**-1'},
        18: {'name': 'u700', 'standard_name': 'x_wind', 'level': 700, 'units': 'm s**-1'},
        19: {'name': 'u850', 'standard_name': 'x_wind', 'level': 850, 'units': 'm s**-1'},
        20: {'name': 'u925', 'standard_name': 'x_wind', 'level': 925, 'units': 'm s**-1'},
        21: {'name': 'u1000', 'standard_name': 'x_wind', 'level': 1000, 'units': 'm s**-1'},
        22: {'name': 'v50', 'standard_name': 'y_wind', 'level': 50, 'units': 'm s**-1'},
        23: {'name': 'v100', 'standard_name': 'y_wind', 'level': 100, 'units': 'm s**-1'},
        24: {'name': 'v150', 'standard_name': 'y_wind', 'level': 150, 'units': 'm s**-1'},
        25: {'name': 'v200', 'standard_name': 'y_wind', 'level': 200, 'units': 'm s**-1'},
        26: {'name': 'v250', 'standard_name': 'y_wind', 'level': 250, 'units': 'm s**-1'},
        27: {'name': 'v300', 'standard_name': 'y_wind', 'level': 300, 'units': 'm s**-1'},
        28: {'name': 'v400', 'standard_name': 'y_wind', 'level': 400, 'units': 'm s**-1'},
        29: {'name': 'v500', 'standard_name': 'y_wind', 'level': 500, 'units': 'm s**-1'},
        30: {'name': 'v600', 'standard_name': 'y_wind', 'level': 600, 'units': 'm s**-1'},
        31: {'name': 'v700', 'standard_name': 'y_wind', 'level': 700, 'units': 'm s**-1'},
        32: {'name': 'v850', 'standard_name': 'y_wind', 'level': 850, 'units': 'm s**-1'},
        33: {'name': 'v925', 'standard_name': 'y_wind', 'level': 925, 'units': 'm s**-1'},
        34: {'name': 'v1000', 'standard_name': 'y_wind', 'level': 1000, 'units': 'm s**-1'},
        35: {'name': 'z50', 'standard_name': 'geopotential_height', 'level': 50, 'units': 'm'},
        36: {'name': 'z100', 'standard_name': 'geopotential_height', 'level': 100, 'units': 'm'},
        37: {'name': 'z150', 'standard_name': 'geopotential_height', 'level': 150, 'units': 'm'},
        38: {'name': 'z200', 'standard_name': 'geopotential_height', 'level': 200, 'units': 'm'},
        39: {'name': 'z250', 'standard_name': 'geopotential_height', 'level': 250, 'units': 'm'},
        40: {'name': 'z300', 'standard_name': 'geopotential_height', 'level': 300, 'units': 'm'},
        41: {'name': 'z400', 'standard_name': 'geopotential_height', 'level': 400, 'units': 'm'},
        42: {'name': 'z500', 'standard_name': 'geopotential_height', 'level': 500, 'units': 'm'},
        43: {'name': 'z600', 'standard_name': 'geopotential_height', 'level': 600, 'units': 'm'},
        44: {'name': 'z700', 'standard_name': 'geopotential_height', 'level': 700, 'units': 'm'},
        45: {'name': 'z850', 'standard_name': 'geopotential_height', 'level': 850, 'units': 'm'},
        46: {'name': 'z925', 'standard_name': 'geopotential_height', 'level': 925, 'units': 'm'},
        47: {'name': 'z1000', 'standard_name': 'geopotential_height', 'level': 1000, 'units': 'm'},
        48: {'name': 't50', 'standard_name': 'air_temperature', 'level': 50, 'units': 'K'},
        49: {'name': 't100', 'standard_name': 'air_temperature', 'level': 100, 'units': 'K'},
        50: {'name': 't150', 'standard_name': 'air_temperature', 'level': 150, 'units': 'K'},
        51: {'name': 't200', 'standard_name': 'air_temperature', 'level': 200, 'units': 'K'},
        52: {'name': 't250', 'standard_name': 'air_temperature', 'level': 250, 'units': 'K'},
        53: {'name': 't300', 'standard_name': 'air_temperature', 'level': 300, 'units': 'K'},
        54: {'name': 't400', 'standard_name': 'air_temperature', 'level': 400, 'units': 'K'},
        55: {'name': 't500', 'standard_name': 'air_temperature', 'level': 500, 'units': 'K'},
        56: {'name': 't600', 'standard_name': 'air_temperature', 'level': 600, 'units': 'K'},
        57: {'name': 't700', 'standard_name': 'air_temperature', 'level': 700, 'units': 'K'},
        58: {'name': 't850', 'standard_name': 'air_temperature', 'level': 850, 'units': 'K'},
        59: {'name': 't925', 'standard_name': 'air_temperature', 'level': 925, 'units': 'K'},
        60: {'name': 't1000', 'standard_name': 'air_temperature', 'level': 1000, 'units': 'K'},
        61: {'name': 'r50', 'standard_name': 'relative_humidity', 'level': 50, 'units': '%'},
        62: {'name': 'r100', 'standard_name': 'relative_humidity', 'level': 100, 'units': '%'},
        63: {'name': 'r150', 'standard_name': 'relative_humidity', 'level': 150, 'units': '%'},
        64: {'name': 'r200', 'standard_name': 'relative_humidity', 'level': 200, 'units': '%'},
        65: {'name': 'r250', 'standard_name': 'relative_humidity', 'level': 250, 'units': '%'},
        66: {'name': 'r300', 'standard_name': 'relative_humidity', 'level': 300, 'units': '%'},
        67: {'name': 'r400', 'standard_name': 'relative_humidity', 'level': 400, 'units': '%'},
        68: {'name': 'r500', 'standard_name': 'relative_humidity', 'level': 500, 'units': '%'},
        69: {'name': 'r600', 'standard_name': 'relative_humidity', 'level': 600, 'units': '%'},
        70: {'name': 'r700', 'standard_name': 'relative_humidity', 'level': 700, 'units': '%'},
        71: {'name': 'r850', 'standard_name': 'relative_humidity', 'level': 850, 'units': '%'},
        72: {'name': 'r925', 'standard_name': 'relative_humidity', 'level': 925, 'units': '%'},
        73: {'name': 'r1000', 'standard_name': 'relative_humidity', 'level': 1000, 'units': '%'},
    }

    def __init__(self, start_time, assets, inputs, outputs, leading_time=240):
        '''
          pretrained_model_path: the path to the directory containing model weights and stats files
          inputs: full path to model IC file (w/ file name)
          outputs: full path to model output
          leading_time: the number of hours to forecast, the default is 240 (10 days)
        '''
        self.start_time = start_time
        self.assets = assets
        self.inputs = inputs

        outdir = pathlib.Path(f'{outputs}/fcngfs.{start_time.strftime("%Y%m%d")}/{start_time.hour:02d}')
        outdir.mkdir(parents=True, exist_ok=True)
        self.outputs = outdir

        self.leading_time = leading_time
        self.backbone_channels = len(self.PARAM)

        lats = np.arange(90, -90.25, -0.25)
        self.latitude = DimCoord(lats, standard_name='latitude', units='degrees')
        lons = np.arange(0, 360, 0.25)
        self.longitude = DimCoord(lons, standard_name='longitude', units='degrees')

    
    def load_statistics(self):
        path = os.path.join(self.assets, "global_means.npy")
        self.means = np.load(path)
        self.means = self.means[:, :self.backbone_channels, ...].astype(np.float32)
    
        path = os.path.join(self.assets, "global_stds.npy")
        self.stds = np.load(path)
        self.stds = self.stds[:, :self.backbone_channels, ...].astype(np.float32)
    
    
    def normalise(self, data, reverse=False):
        if reverse:
            new_data = data * self.stds + self.means
        else:
            new_data = (data - self.means) / self.stds 
    
        return new_data   
    
    def load_model(self, checkpoint_file):
        model = nvs.FourierNeuralOperatorNet()
        model.zero_grad()
    
        checkpoint = torch.load(checkpoint_file, map_location=device)
        weights = checkpoint["model_state"]
        drop_vars = ["module.norm.weight", "module.norm.bias"]
        weights = {k: v for k, v in weights.items() if k not in drop_vars}
    
        try:
            # Try adding model weights as dictionary
            new_state_dict = dict()
            for k, v in checkpoint["model_state"].items():
                name = k[7:]
                if name != "ged":
                    new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        except Exception:
            model.load_state_dict(checkpoint["model_state"])
    
        # Set model to eval mode and return
        model.eval()
        model.to(device)
    
        return model

    def run(self):
        self.load_statistics()
       
        #read inputs
        with open(self.inputs, 'rb') as f: 
            data = np.load(f)
            
        #save f000
        self.write(np.expand_dims(data, axis=0), 0)

        all_fields_numpy = data.astype(np.float32)

        all_fields_numpy = self.normalise(all_fields_numpy)

        input_iter = torch.from_numpy(all_fields_numpy).to(device)

        torch.set_grad_enabled(False)

        path = os.path.join(self.assets, "weights.tar")
        model = self.load_model(path)
        
        for i in range(self.leading_time // 6):
            print(f'Starting inference for step {i} ')
            
            output = model(input_iter)
            input_iter = output

            step = (i + 1) * 6
            output = self.normalise(output.cpu().numpy(), reverse=True)
        
            #np.save(f'{self.outputs}/output_step{i:02d}.npy', output)
            self.write(output, step)

    def write(self, data, step):
        out_fname = f'{self.outputs}/fcngfs.t{self.start_time.hour:02d}z.pgrb2.0p25.f{step:03d}'

        if os.path.isfile(out_fname):
            print(f'Deleting file {out_fname}')
            os.remove(out_fname)

        time_unit_str = f"Hours since {self.start_time.strftime('%Y-%m-%d %H:00:00')}"
        time_unit = cf_units.Unit(time_unit_str, calendar=cf_units.CALENDAR_STANDARD)
        new_time_point = time_unit.date2num(self.start_time + timedelta(hours=step)) 
        time = DimCoord([new_time_point], standard_name='time', units=time_unit_str)

        for i in np.arange(data.shape[1]):
        

            values = np.expand_dims(data[0, i, :, :], axis=0)

            #units conversion: geopotential -> geopotential height
            if self.PARAM[i+1]['standard_name'] == 'geopotential_height':
                values = values / 9.80665

            cube = Cube(
                values,
                standard_name = self.PARAM[i+1]['standard_name'],
                var_name = self.PARAM[i+1]['name'],
                units = self.PARAM[i+1]['units'],
                dim_coords_and_dims=[(time, 0), (self.latitude, 1), (self.longitude, 2)],
            )
            cube.coord('latitude').coord_system=iris.coord_systems.GeogCS(4326)
            cube.coord('longitude').coord_system=iris.coord_systems.GeogCS(4326)
            cube.add_aux_coord(iris.coords.DimCoord(step, standard_name='forecast_period', units='hours'))
            if i < 8:
                if self.PARAM[i+1]['standard_name'] not in ['precipitation_amount']:
                    cube.add_aux_coord(iris.coords.DimCoord(self.PARAM[i+1]['level'], standard_name='height', units='m'))
            else:
                cube.add_aux_coord(iris.coords.DimCoord(self.PARAM[i+1]['level']*100, standard_name='air_pressure', units='Pa'))
             
            iris_grib.save_messages(tweaked_messages(cube), out_fname, append=True)
   


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("start_datetime", help="Start datetime in the format 'YYYYMMDDHH'")

    parser.add_argument("-w", "--weights", help="parent directory of the weights and stats", required=True)
    parser.add_argument("-i", "--input", help="input file path (including file name)", required=True)
    parser.add_argument("-o", "--output", help="output directory", default=None)
    parser.add_argument("-l", "--length", type=int, help="total hours to forecast", required=True)

    args = parser.parse_args()

    start_datetime = datetime.strptime(args.start_datetime, "%Y%m%d%H")
    fcn = FourCastNetv2(start_datetime, args.weights, args.input, args.output, args.length)
    fcn.run()
