
import torch
import torch.nn as nn


from models.discriminator import ProjectedDiscriminator
from models.generator import Generator_Adain_Upsample
from models.arcface_loader import get_arcface_model

class SimSwap():
    def __init__(self):
        self.generator = Generator_Adain_Upsample(input_nc=3, output_nc=3, latent_size=512, n_blocks=9, deep=False)
        self.generator.to(device="cuda")

        self.discriminator = ProjectedDiscriminator(diffaug=False, interp224=False,)
        self.discriminator.to(device="cuda")
        self.discriminator.feature_network.requires_grad_(False)

        self.arcface_model = get_arcface_model()
        self.arcface_model.to(device="cuda")
        self.arcface_model.eval()
        self.arcface_model.requires_grad_(False)


        # Define Loss Funcitons 
        self.feature_loss  = nn.L1Loss()
        self.reconstruction_loss   = nn.L1Loss()

        # Define Optimizers
        # optimizer G
        params = list(self.generator.parameters())
        self.optimizer_G = torch.optim.Adam(params, lr=0.0004, betas=(0.0, 0.99),eps=1e-8)

        # optimizer D
        params = list(self.discriminator.parameters())
        self.optimizer_D = torch.optim.Adam(params, lr=0.0002, betas=(0.0, 0.99),eps=1e-8)


    def cosin_metric(self, x1, x2):
        return torch.nn.functional.cosine_similarity(x1, x2, dim=1)

