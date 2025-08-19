# main/utils_anchor.py
import torch, random

class AnchorPairs:
    """
    Loads offline anchor pairs (z, x_target, y_onehot) saved by tools/make_anchor_pairs.py
    and lets you sample mini-batches from them. Not a DataLoader—just a lightweight helper.
    """
    def __init__(self, path, device):
        blob = torch.load(path, map_location="cpu")
        self.Z = blob["z"].to(torch.float32).to(device)   # [M,3,H,W]
        self.X = blob["x"].to(torch.float32).to(device)   # [M,3,H,W] in [-1,1]
        self.Y = blob["y"].to(torch.float32).to(device)   # [M,label_dim]
        self.img_idx = blob["img_idx"].to(torch.long).cpu().numpy()
        self.M = self.Z.shape[0]
        # per-image buckets → balanced sampling across the 10 images
        self.by_image = {}
        for j in range(self.M):
            i = int(self.img_idx[j])
            self.by_image.setdefault(i, []).append(j)
        self.device = device
        self.num_images = len(self.by_image)

    def sample_indices(self, m, balanced=True):
        if not balanced:
            return torch.randint(0, self.M, (m,), device=self.device)
        # round-robin images, pick a random anchor from each bucket
        keys = list(self.by_image.keys())
        out = []
        for t in range(m):
            bucket = self.by_image[keys[t % len(keys)]]
            out.append(random.choice(bucket))
        return torch.tensor(out, device=self.device, dtype=torch.long)

    def get(self, idx_long):
        # idx_long: LongTensor [K] on device
        return self.Z[idx_long], self.X[idx_long], self.Y[idx_long]
