"""Regenerate visualization from saved checkpoint"""
import sys
sys.path.insert(0, '/storage/project/r-rbras6-0/jjason3/PINN_inverse/src')

import torch
import numpy as np
from config_loader_inverse import InversePINNConfig
from pinn_inverse import RichardsInversePINN
from normalization_helper import NormalizationHelper
from visualization_inverse import save_inverse_figures

config = InversePINNConfig('/storage/project/r-rbras6-0/jjason3/PINN_inverse/configs/inverse_US-PAS_2021.yaml')

# Setup normalizer using soil_params property
normalizer = NormalizationHelper(config.soil_params, config.L, S_max=0.0)

npz_path = '/storage/project/r-rbras6-0/jjason3/PINN_inverse/data/US-PAS_2021-04-01_2021-11-30.npz'
data = np.load(npz_path)
times = data['times']
q_flux = data['qsurf']
theta_obs = data['smap_sm']
zb_true = data.get('zb', None)

t_flux = times
t_obs = times

print(f"Loaded {len(times)} time points")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = RichardsInversePINN(
    soil_params=config.soil_params,
    q_surf_data=(t_flux, q_flux),
    theta_obs_data=(t_obs, theta_obs),
    h_net_config=config.h_net_config,
    n_spline_knots=config.n_spline_knots,
    normalizer=normalizer,
    zb_initial=config.zb_initial,
    zb_max=config.zb_max,
    zr=config.zr,
    device=device
)

ckpt = torch.load('/storage/project/r-rbras6-0/jjason3/PINN_inverse/checkpoints_inverse/US-PAS_2021/checkpoint_final.pt',
                  map_location=device)
model.load_state_dict(ckpt['model_state_dict'])
model = model.to(device)  # Ensure all model components are on device
model.eval()

# Load training results for loss history
results = np.load('/storage/project/r-rbras6-0/jjason3/PINN_inverse/outputs_inverse/US-PAS_2021/training_results.npz')

comps = {
    'pde': results['comp_pde'],
    'flux_bc': results['comp_flux_bc'],
    'wt_head': results['comp_wt_head'],
    'data': results['comp_data'],
    'ic_h': results['comp_ic_h'],
    'ic_zb': results['comp_ic_zb'],
    'zb_smooth': results['comp_zb_smooth'],
    'zb_curv': results['comp_zb_curv']
}

sample_comps = None
if 'sample_comp_pde' in results:
    sample_comps = {
        'pde': results['sample_comp_pde'],
        'flux_bc': results['sample_comp_flux_bc'],
        'wt_head': results['sample_comp_wt_head'],
        'data': results['sample_comp_data']
    }

print("Generating visualization...")
save_inverse_figures(
    model=model,
    theta_obs_data=(t_obs, theta_obs),
    q_surf_data=(t_flux, q_flux),
    soil_params=config.soil_params,
    losses=results['losses'],
    comps=comps,
    output_dir='/storage/project/r-rbras6-0/jjason3/PINN_inverse/outputs_inverse/US-PAS_2021',
    zb_true=zb_true,
    sample_losses=results['sample_losses'] if 'sample_losses' in results else None,
    sample_comps=sample_comps,
    sample_epochs=results['sample_epochs'] if 'sample_epochs' in results else None,
    device=device
)
print("Done!")
