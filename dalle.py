import torch
from torchtils import F
from torchtils.models import simple_diffuser
import pytorch_lightning as pl


class GaussianDiffusion(pl.LightningModule):
    def __init__(
        self,
        dim,
        timesteps=100,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        resnet_block_groups=8,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        learned_sinusoidal_dim=16,
    ) -> None:
        super().__init__()
        self.timesteps = timesteps
        self.model = simple_diffuser.Unet(
            dim,
            init_dim=init_dim,
            out_dim=out_dim,
            dim_mults=dim_mults,
            channels=channels,
            resnet_block_groups=resnet_block_groups,
            learned_variance=learned_variance,
            learned_sinusoidal_cond=learned_sinusoidal_cond,
            learned_sinusoidal_dim=learned_sinusoidal_dim,
        )

    def noise_schedule(self, t):
        return 1 / ((self.timesteps - t)**2)

    def calc_loss(self, btch):

        t = torch.randint(
            0,
            self.timesteps,
            (btch.shape[0],),
        )
        err = torch.randn_like(btch) * self.noise_schedule(t)
        loss = F.mse_loss(self.model(btch + err), err)
        return loss.mean()

    def q_sample(self):
        pass

    def p_sample_step(self, t):
        pass
    def p_sample(self): #! need to understand this completely
        pass 

    def train(self):
        pass


if __name__ == "__main__":
    print("hi")
    # diffuser =
    # Get dataset
    # Train
    # Test
