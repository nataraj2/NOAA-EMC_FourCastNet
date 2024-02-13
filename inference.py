import os
import argparse

import numpy as np
import torch
import ai_models_fourcastnetv2.fourcastnetv2 as nvs

device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

class FourCastNetv2:
    ordering = [
        "10u",
        "10v",
        "100u",
        "100v",
        "2t",
        "sp",
        "msl",
        "tcwv",
        "u50",
        "u100",
        "u150",
        "u200",
        "u250",
        "u300",
        "u400",
        "u500",
        "u600",
        "u700",
        "u850",
        "u925",
        "u1000",
        "v50",
        "v100",
        "v150",
        "v200",
        "v250",
        "v300",
        "v400",
        "v500",
        "v600",
        "v700",
        "v850",
        "v925",
        "v1000",
        "z50",
        "z100",
        "z150",
        "z200",
        "z250",
        "z300",
        "z400",
        "z500",
        "z600",
        "z700",
        "z850",
        "z925",
        "z1000",
        "t50",
        "t100",
        "t150",
        "t200",
        "t250",
        "t300",
        "t400",
        "t500",
        "t600",
        "t700",
        "t850",
        "t925",
        "t1000",
        "r50",
        "r100",
        "r150",
        "r200",
        "r250",
        "r300",
        "r400",
        "r500",
        "r600",
        "r700",
        "r850",
        "r925",
        "r1000",
    ]

    def __init__(self, assets, inputs, outputs, leading_time=240):
        '''
          pretrained_model_path: the path to the directory containing model weights and stats files
          inputs: full path to model IC file (w/ file name)
          outputs: full path to model output
          leading_time: the number of hours to forecast, the default is 240 (10 days)
        '''
        self.assets = assets
        self.inputs = inputs
        self.outputs = outputs
        self.leading_time = leading_time
        self.backbone_channels = len(self.ordering)

    
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
            
        all_fields_numpy = data.astype(np.float32)

        all_fields_numpy = self.normalise(all_fields_numpy)

        input_iter = torch.from_numpy(all_fields_numpy).to(device)

        torch.set_grad_enabled(False)

        path = os.path.join(self.assets, "weights.tar")
        model = self.load_model(path)
        
        for i in range(self.leading_time // 6):
            print(f'Starting inference for step {i} ')
            
            output = model(input_iter)

            step = (i + 1) * 6
            output = self.normalise(output.cpu().numpy(), reverse=True)
        
            np.save(f'{self.outputs}/output_step{i:02d}.npy', output)


if __name__ == '__main__':

   parser = argparse.ArgumentParser()
   parser.add_argument("-w", "--weights", help="parent directory of the weights and stats", required=True)
   parser.add_argument("-i", "--input", help="input file path (including file name)", required=True)
   parser.add_argument("-o", "--output", help="output directory", default=None)
   parser.add_argument("-l", "--length", type=int, help="total hours to forecast", required=True)

   args = parser.parse_args()

   fcn = FourCastNetv2(args.weights, args.input, args.output, args.length)
   fcn.run()
