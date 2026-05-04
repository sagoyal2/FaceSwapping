from models.generator import Generator_Adain_Upsample
from models.discriminator import ProjectedDiscriminator
from models.arcface_loader import get_arcface_model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    generator = Generator_Adain_Upsample(
        input_nc=3,
        output_nc=3,
        latent_size=512,
        n_blocks=9,
        deep=False,
    )
    discriminator = ProjectedDiscriminator(diffaug=False, interp224=False)
    arcface_model = get_arcface_model()

    print(f"generator params: {count_parameters(generator)}")
    print(f"arcface_model params: {count_parameters(arcface_model)}")
    print(f"discriminator params: {count_parameters(discriminator)}")
