
import torch

ARCFACE_CHECKPOINT = "/home/ubuntu/DemoFaceSwappingData/arcface_checkpoint/arcface_checkpoint.tar"

def get_arcface_model():
    model = torch.load(
        ARCFACE_CHECKPOINT,
        map_location=torch.device("cuda"),
        weights_only=False,
    )
    # model = model.to(device="cuda")
    # model.eval()
    # model.requires_grad_(False)

    return model