import torch
import zarr 
import numpy as np

# Load the checkpoint
checkpoint_path = "/home/mengo/Research/LLM_DOM/diffusion_mk2/weights/chkp_golden-snowflake-37_epoch_3200.pt"
dataset_path = "/home/mengo/Research/LLM_DOM/diffusion_mk2/zarr_data/dataset.zarr.zip"
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)


# Dataset stats for obs_ee: min=[ 0.1177 -0.3544  0.81   -3.1405 -0.    ], max=[0.6351 0.2948 1.5506 3.1408 0.04  ]
# Dataset stats for obs_dlo: min=[ 0.4081 -0.2314  0.7011], max=[0.5955 0.1462 0.7415]
# Dataset stats for obs_target: min=[ 0.4093 -0.2314  0.7017], max=[0.5931 0.1462 0.7099]
# Dataset stats for action: min=[-0.167  -0.1813 -0.2225 -6.2745 -0.0345], max=[0.1927 0.1874 0.2316 6.2543 0.0168]
# ee_states_stats = {'min': np.array([-4.32279963e-01, -3.89113625e-01, -1.40775994e-01, -3.14141130e+00, -3.73688927e-05]), 
#                    'max': np.array([0.44775614, 0.39651168, 0.84576694, 3.14157748, 0.04001684])}
ee_states_stats = {'min': np.array([0.1177, -0.3544, 0.81, -3.1405, 0.0]), 
                   'max': np.array([0.6351, 0.2948, 1.5506, 3.1408, 0.04])}
dlo_stats = {'min': np.array([0.4081, -0.2314, 0.7011]), 
            'max': np.array([0.5955, 0.1462, 0.7415])}
target_stats = {'min': np.array([0.4093, -0.2314, 0.7017]), 
                'max': np.array([0.5931, 0.1462, 0.7099])}
action_stats = {'min': np.array([-0.167, -0.1813, -0.2225, -6.2745, -0.0345]), 
                'max': np.array([0.1927, 0.1874, 0.2316, 6.2543, 0.0168])}

# Modify the desired fields
# For example, updating 'dataset_stats':

checkpoint["dataset_stats"]["obs_ee"] = ee_states_stats
checkpoint["dataset_stats"]["obs_dlo"] = dlo_stats
checkpoint["dataset_stats"]["obs_target"] = target_stats
checkpoint["dataset_stats"]["action"] = action_stats
print(checkpoint["dataset_stats"]["obs_ee"])
print(checkpoint["dataset_stats"]["obs_dlo"])
print(checkpoint["dataset_stats"]["obs_target"])
print(checkpoint["dataset_stats"]["action"])
print(ee_states_stats)
print(action_stats)
# Save it back
torch.save(checkpoint, checkpoint_path)
