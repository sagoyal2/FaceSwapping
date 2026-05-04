import torch
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from train.dataloader import FacePairDataset, AlternatingPairBatchSampler
from evaluation.tensorboard_logger import log_training_step
import torch.nn.functional as F

from models.simswap import SimSwap

BATCH_SIZE = 112
NUM_BATCHES_PER_EPOCH = 100
NUM_EPOCHS = 200
INPUT_DIR = Path("/home/ubuntu/DemoFaceSwappingData/lfw_funneled_cropped_aligned_224/")

TENSORBOARD_PORT = 6007
LOGGING_ITERATION = 97
N_G_STEPS = 4

if __name__ == "__main__":

    # SETUP DATALOADER
    dataset = FacePairDataset(
        root_dir=INPUT_DIR,
    )

    batch_sampler = AlternatingPairBatchSampler(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        num_batches_per_epoch=NUM_BATCHES_PER_EPOCH,
    )

    loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    # SETUP TENSORBOARD
    log_root = Path("/home/ubuntu/DemoFaceSwapping/runs/simswap")
    log_root.mkdir(parents=True, exist_ok=True)
    run_name = datetime.now().strftime("run_%Y%m%d-%H%M%S")
    log_dir = log_root / run_name
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"View all experiments with: tensorboard --logdir {log_root} --port {TENSORBOARD_PORT}")
    print(f"Current Run: {run_name}")

    # SETUP CHECKPOINT
    checkpoint_dir = log_root / "checkpoints" / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
 
    # SETUP MODEL
    device = torch.device("cuda")
    simswap = SimSwap()
    use_amp = device.type == "cuda"
    scaler_D = GradScaler("cuda", enabled=use_amp)
    scaler_G = GradScaler("cuda", enabled=use_amp)

    # SETUP TRAINING
    total_iters = NUM_EPOCHS * NUM_BATCHES_PER_EPOCH
    iteration = 0
    with tqdm(total=total_iters, desc="Training", unit="iter") as pbar:
        for epoch in range(NUM_EPOCHS):
            for batch_idx, batch in enumerate(loader):
                images = batch["images"] # (B, 2, C, H, W)
                images = images.to(device=device, non_blocking=True)

                target = images[:, 0, :, :, :] # (B, C, H, W)
                source = images[:, 1, :, :, :] # (B, C, H, W)

                # Train Discriminator
                simswap.discriminator.requires_grad_(True)
                simswap.generator.requires_grad_(False)

                with torch.no_grad():
                    with autocast("cuda", enabled=use_amp):
                        source_112 = F.interpolate(source, size=(112, 112), mode="bilinear")
                        source_identity_embedding = simswap.arcface_model(source_112) # (B, 512)
                        source_identity_embedding = F.normalize(source_identity_embedding, p=2, dim=1)
                        target_fake = simswap.generator(target, source_identity_embedding) # (B, C, H, W)

                with autocast("cuda", enabled=use_amp):
                    gen_logits, _ = simswap.discriminator(target_fake.detach(), None)
                    loss_Dgen = (F.relu(torch.ones_like(gen_logits) + gen_logits)).mean()
                    # real_logits,_   = simswap.discriminator(source, None)
                    real_logits, _ = simswap.discriminator(target, None)
                    loss_Dreal      = (F.relu(torch.ones_like(real_logits) - real_logits)).mean()
                    loss_D          = loss_Dgen + loss_Dreal

                simswap.optimizer_D.zero_grad(set_to_none=True)
                scaler_D.scale(loss_D).backward()
                scaler_D.step(simswap.optimizer_D)
                scaler_D.update()

                # Train Generator N times
                simswap.discriminator.requires_grad_(False)
                simswap.generator.requires_grad_(True)

                for _ in range(N_G_STEPS):
                    with autocast("cuda", enabled=use_amp):
                        target_fake = simswap.generator(target, source_identity_embedding)
                        gen_logits, feat = simswap.discriminator(target_fake, None)
                        loss_Gmain = (-gen_logits).mean()

                        target_fake_112 = F.interpolate(target_fake, size=(112, 112), mode="bilinear")
                        target_fake_identity_embedding = simswap.arcface_model(target_fake_112)
                        target_fake_identity_embedding = F.normalize(target_fake_identity_embedding, p=2, dim=1)
                        loss_G_ID = (1 - simswap.cosin_metric(source_identity_embedding, target_fake_identity_embedding)).mean()

                        real_feat = simswap.discriminator.get_feature(target)
                        loss_G_feat = simswap.feature_loss(feat["3"],real_feat["3"])

                        loss_G = loss_Gmain + loss_G_ID * 70.0 + loss_G_feat * 10.0

                        loss_G_recon = torch.zeros((), device=device)
                        if (batch["labels"] == 1).all():
                            loss_G_recon = simswap.reconstruction_loss(target_fake, target) * 10.0
                            loss_G += loss_G_recon

                    simswap.optimizer_G.zero_grad(set_to_none=True)
                    scaler_G.scale(loss_G).backward()
                    scaler_G.step(simswap.optimizer_G)
                    scaler_G.update()

                # Do Logging
                if iteration % LOGGING_ITERATION == 0:
                    with torch.no_grad():
                        log_training_step(
                            writer,
                            iteration,
                            simswap,
                            dataset,
                            scalars={
                                "loss/D_gen": loss_Dgen.item(),
                                "loss/D_real": loss_Dreal.item(),
                                "loss/D": loss_D.item(),
                                "loss/G_main": loss_Gmain.item(),
                                "loss/G_ID": loss_G_ID.item(),
                                "loss/G_feat": loss_G_feat.item(),
                                "loss/G_recon": loss_G_recon.item(),
                                "loss/G": loss_G.item(),
                                "train/N_G_STEPS": float(N_G_STEPS),
                            },
                            device=device,
                        )
                    
                    # Save Generator Checkpoint
                    ckpt_path = checkpoint_dir / f"generator_iter_{iteration}.pt"
                    torch.save(simswap.generator.state_dict(), ckpt_path)

                iteration += 1
                pbar.update(1)

    writer.close()