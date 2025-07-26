import torch
from src.models.cryo_renderer import CryoEMRenderer
from src.models.fields import DensityNetwork

# Example usage of CryoEMRenderer
if __name__ == "__main__":
    density_net = DensityNetwork(checkpoint_path=None, style_dim=128, W=128, D=4, input_ch=3, input_ch_views=3)
    renderer = CryoEMRenderer(density_network=density_net, n_samples=64, perturb=0.0)

    # simple test rays
    rays_o = torch.zeros(1, 3)
    rays_d = torch.tensor([[0.0, 0.0, -1.0]])
    near = torch.tensor([0.0])
    far = torch.tensor([2.0])

    z = torch.randn(1, 128)

    out = renderer.render(rays_o, rays_d, near, far, z=z)
    print('density_map', out['density_map'].shape)
