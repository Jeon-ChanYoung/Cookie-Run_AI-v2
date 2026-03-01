import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def precompute_vae_latents(config, vae, frame_dataset, batch_size=256):
    vae.change_train_mode(train=False)
    loader = DataLoader(frame_dataset, batch_size=batch_size,
                        shuffle=False, drop_last=False,
                        num_workers=2, pin_memory=True)
    all_z = []
    for frames in loader:
        frames = frames.to(config.device, non_blocking=True)
        all_z.append(vae.encode(frames).cpu())
    return torch.cat(all_z, dim=0)  # (N, 4, 16, 32)


def compute_latent_stats(vae_latents):
    mean = vae_latents.mean(dim=(0, 2, 3), keepdim=True)  # (1, 4, 1, 1)
    std  = vae_latents.std(dim=(0, 2, 3), keepdim=True)   # (1, 4, 1, 1)
    print(f"  channel means: {mean.squeeze()}")
    print(f"  channel stds:  {std.squeeze()}")
    return mean.squeeze(0), std.squeeze(0)  # (4, 1, 1)


def straight_through_categorical(logits):
    probs = F.softmax(logits, dim=-1)
    hard = F.one_hot(probs.argmax(dim=-1), logits.shape[-1]).float()
    return hard - probs.detach() + probs