import torch
import torch.nn as nn

class CryoEMRenderer:
    """Renderer that integrates densities along camera rays to simulate cryo-EM projections."""

    def __init__(self, density_network: nn.Module, n_samples: int = 64, perturb: float = 0.0):
        self.density_network = density_network
        self.n_samples = n_samples
        self.perturb = perturb

    def render(self, rays_o, rays_d, near, far, z=None, w=None, **_):
        batch_size = rays_o.shape[0]
        device = rays_o.device

        z_vals = torch.linspace(0.0, 1.0, self.n_samples, device=device)
        z_vals = near + (far - near) * z_vals[None, :]

        if self.perturb > 0:
            mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            t_rand = torch.rand([batch_size, self.n_samples], device=device)
            z_vals = lower + (upper - lower) * t_rand

        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
        if z is not None or w is not None:
            density = self.density_network(pts.reshape(-1, 3), z=z, w=w)[:, :1]
        else:
            density = self.density_network(pts.reshape(-1, 3))[:, :1]
        density = density.reshape(batch_size, self.n_samples)

        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, dists[..., -1:]], -1)

        weights = density * dists
        projection = torch.sum(weights, dim=-1, keepdim=True)

        return {
            'density_map': projection,
            'weights': weights,
            'z_vals': z_vals,
        }
