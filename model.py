import os
from typing import *
from pathlib import Path
import shutil

import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from tqdm import tqdm

from torch.utils.data import Dataset
from scipy.ndimage import distance_transform_edt
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import regionprops

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torch.nn.functional as F
from torch.optim import Adam


class MoverDataset(Dataset):

    def __init__(
        self, 
        data_directory: str, 
        cache_directory: str, 
        max_circles: int = 256,
        patch_size: int = 12,
    ) -> None:
        self.data_directory: Path = Path(data_directory)
        self.max_circles: int = max_circles
        self.patch_size: int = patch_size
        self.__tmp: Path = Path(".frames_tmp")
        self.filepaths: List[Path] = sorted(self.data_directory.glob("*.npy"))
        self.cache_directory: Path = Path(cache_directory)
        self.cache_directory.mkdir(exist_ok=True)

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        filepath = self.filepaths[index]
        target_string: str = filepath.stem[-1]
        assert target_string.isdigit()
        target: torch.Tensor = torch.tensor([float(target_string)], dtype=torch.float)

        cache_path = self.cache_directory / (filepath.stem + ".pt")
        if cache_path.exists():
            video_tensor, patch_tensor, center_tensor = torch.load(cache_path)
        else:
            input_video = np.load(file=filepath)
            video_tensor, patch_tensor, center_tensor = self._compute_sample(input_video)
            torch.save((video_tensor, patch_tensor, center_tensor), cache_path)

        return video_tensor, patch_tensor, center_tensor, target

    def _compute_sample(self, input_video: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        T, H, W = input_video.shape
        center_array: np.ndarray = np.full(shape=(T, self.max_circles, 2), fill_value=-1.0, dtype=np.float32)

        for t in range(T):
            frame: np.ndarray = input_video[t]
            dist: np.ndarray = distance_transform_edt(input=frame)
            coords: np.ndarray = peak_local_max(image=dist, min_distance=1, labels=frame)
            markers: np.ndarray = np.zeros_like(a=frame, dtype=np.int32)
            for i, (y, x) in enumerate(coords):
                markers[y, x] = i + 1

            labels_ws: np.ndarray = watershed(image=-dist, markers=markers, mask=frame)
            props: list = regionprops(label_image=labels_ws)
            for i, prop in enumerate(props):
                if i >= self.max_circles:
                    break
                ch, cw = prop.centroid
                center_array[t, i, 0] = ch
                center_array[t, i, 1] = cw

        # Compute patch counts
        n_hpatches: int = H // self.patch_size
        n_wpatches: int = W // self.patch_size
        patch_count: np.ndarray = np.zeros(shape=(T, n_hpatches, n_wpatches), dtype=np.float32)

        for t in range(T):
            for i in range(self.max_circles):
                ch, cw = center_array[t, i]
                if ch < 0 or cw < 0:  # ignore placeholders
                    continue
                py: int = int(ch) // self.patch_size
                px: int = int(cw) // self.patch_size
                patch_count[t, py, px] += 1.

        return (
            torch.from_numpy(input_video),  # (T, H, W)
            torch.from_numpy(patch_count), # (T, n_hpatches, n_wpatches)
            torch.from_numpy(center_array), # (T, max_circles, 2)
        )
    
    def plot_feature(self, index: int) -> None:
        video_tensor, patch_tensor, center_tensor, target = self[index]
        target: int = int(target.item())
        video_array: np.ndarray = video_tensor.cpu().numpy()
        center_array: np.ndarray = center_tensor.cpu().numpy()  # (T, C, 2)
        T, C, _ = center_array.shape
        
        # Plot
        os.makedirs(self.__tmp, exist_ok=True)
        filepaths: List[Path] = []
        for t in range(video_array.shape[0]):
            frame: np.ndarray = video_array[t]
            centers: np.ndarray = center_array[t]
            fig, ax = plt.subplots()
            fig.patch.set_facecolor('black')
            ax.set_facecolor('black')
            ax.imshow(frame, cmap='gray')
            for (y, x) in centers:
                if y >= 0 and x >= 0:
                    ax.plot(x, y, 'ro', markersize=2)

            # Meshing
            h, w = frame.shape
            for y in range(0, h, self.patch_size):
                ax.axhline(y, color='gray', linewidth=0.5)
            for x in range(0, w, self.patch_size):
                ax.axvline(x, color='gray', linewidth=0.5)

            # Show sums
            frame_patches: np.ndarray = patch_tensor[t]  # (n_hpatches, n_wpatches)
            for i, row_sum in enumerate(frame_patches.sum(axis=1)):
                ax.text(
                    w, i * self.patch_size + self.patch_size // 2 + 2, f"{int(row_sum)}",
                    va='center', fontsize=6, color='cyan'
                )
            for j, col_sum in enumerate(frame_patches.sum(axis=0)):
                ax.text(
                    j * self.patch_size + self.patch_size // 2, h + 8, f"{int(col_sum)}",
                    ha='center', fontsize=6, color='cyan'
                )
            ax.text(
                w, -10,  # x centered, y above the top
                f"Frame {t + 1} | Label: {target}",
                color='white',
                ha='right',
                va='bottom',
                fontsize=8,
                transform=ax.transData
            )
            ax.axis('off')
            filepath: Path = self.__tmp / f"frame_{t:03d}.png"
            filepaths.append(filepath)
            plt.savefig(filepath, pad_inches=0.1)
            plt.close(fig)

        images = [imageio.imread(f) for f in filepaths]
        imageio.mimsave(uri=f"feature{index}.gif", ims=images, duration=0.15)
        shutil.rmtree(self.__tmp)

    def plot_video(self, index: int) -> None:
        os.makedirs(self.__tmp, exist_ok=True)
        video_tensor, patch_tensor, center_tensor, target = self[index]
        target: int = int(target.item())
        video_array: np.ndarray = video_tensor.cpu().numpy()  # (T, H, W)

        filepaths: List[Path] = []
        for t in range(video_array.shape[0]):
            frame: np.ndarray = video_array[t]
            fig, ax = plt.subplots()
            ax.imshow(frame, cmap='gray')
            ax.set_title(f"Frame {t + 1} | Label: {target}")
            ax.axis('off')
            filepath: Path = self.__tmp / f"frame_{t:03d}.png"
            filepaths.append(filepath)
            plt.savefig(filepath, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

        images = [imageio.imread(f) for f in filepaths]
        imageio.mimsave(f"video{index}.gif", images, duration=0.15)
        shutil.rmtree(self.__tmp)


class CollectiveMotionDetector(nn.Module):
    """
    Predict collective velocity vector
    """
    def __init__(self, embedding_dim: int, dropout: float = 0.) -> None:
        super().__init__()
        self.embedding_dim: int = embedding_dim
        self.dropout: float = dropout
        layers: List[nn.Module] = [
            nn.Conv3d(
                in_channels=1, out_channels=embedding_dim, 
                kernel_size=3, stride=1, padding=1,
            ),
            nn.Dropout(p=dropout),
            nn.ReLU(),
        ]
        for _ in range(10 - 2):
            layers.extend([
                nn.Conv3d(
                    in_channels=embedding_dim, out_channels=embedding_dim, 
                    kernel_size=3, stride=1, padding=1,
                ),
                nn.Dropout(p=dropout),
                nn.ReLU(),
            ])
        layers.extend([
                nn.Conv3d(
                    in_channels=embedding_dim, out_channels=2,
                    kernel_size=1, stride=1, padding=0,
                ),
        ])
        self.backbone = nn.Sequential(*layers)

    def forward(self, input_video: torch.Tensor) -> torch.Tensor:
        # x: (N, T, H, W)
        N, T, H, W = input_video.shape
        input_video = input_video.unsqueeze(1)  # (N, 1, T, H, W)
        feature: torch = self.backbone(input_video)
        assert feature.shape == (N, 2, T, H, W)
        v_field: torch.Tensor = feature.diff(n=1, dim=2)    # (N, 2, T - 1, H, W)
        mean_v: torch.Tensor = v_field.mean(dim=(2, 3, 4))  # (N, 2)
        assert mean_v.shape == (N, 2)
        mean_v = mean_v.unsqueeze(dim=1).unsqueeze(dim=1)
        assert mean_v.shape == (N, 1, 1, 2)
        return mean_v
    

class MotionMatchingLoss(nn.Module):

    def __init__(self, max_h: int, max_w: int) -> None:
        super().__init__()
        self.max_h: float = float(max_h)
        self.max_w: float = float(max_w)

    def forward(self, center_tensor: torch.Tensor, velocity_vector: torch.Tensor) -> torch.Tensor:
        N, T, C, _ = center_tensor.shape
        shifted = center_tensor + velocity_vector  # (N, T, C, 2)
        shifted_x = shifted[..., 0].clamp(0, self.max_h)
        shifted_y = shifted[..., 1].clamp(0, self.max_w)
        shifted = torch.stack((shifted_x, shifted_y), dim=-1)

        pred = shifted[:, :-1]   # (N, T-1, C, 2)
        target = center_tensor[:, 1:]  # (N, T-1, C, 2)

        loss = 0.0
        for t in range(pred.size(1)):
            dist = torch.cdist(pred[:, t], target[:, t], p=2)  # (N, C, C)
            loss += (dist.min(dim=2)[0].mean() + dist.min(dim=1)[0].mean()) / 2
        return loss / pred.size(1)


class Trainer:

    def __init__(self, dataset: Dataset, model: nn.Module, device: torch.device, lr: float) -> None:

        self.device: torch.device = device
        self.model = model.to(self.device)
        self.lr: float = lr
        self.optimizer: Adam = Adam(self.model.parameters(), lr=lr)
        self.motion_matching_loss = MotionMatchingLoss(max_h=360, max_w=480)

        # TODO:
        dataset = Subset(dataset, indices=range(2000))

        total_len: int = len(dataset)
        val_len: int = int(total_len * 0.1)
        test_len: int = int(total_len * 0.1)
        train_len: int = total_len - val_len - test_len

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset=dataset, lengths=[train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(1),
        )
        self.train_loader: DataLoader = DataLoader(self.train_dataset, batch_size=32, shuffle=True)
        self.val_loader: DataLoader = DataLoader(self.val_dataset, batch_size=32, shuffle=False)
        self.test_loader: DataLoader = DataLoader(self.test_dataset, batch_size=32, shuffle=False)


    def evaluate(self, dataloader: DataLoader) -> float:
        assert isinstance(dataloader, DataLoader)
        self.model.eval()
        total_correct: int = 0
        total_samples: int = 0

        with torch.no_grad():
            for video_tensor, patch_tensor, center_tensor, target_tensor in dataloader:
                video_tensor = video_tensor.to(self.device).float()
                center_tensor = center_tensor.to(self.device)
                target_tensor = target_tensor.to(self.device)
                velocity_vector: torch.Tensor = self.model(input_video=video_tensor)
                shifted_center_tensor: torch.Tensor = center_tensor + velocity_vector
                compare: torch.Tensor = shifted_center_tensor == center_tensor
                print(compare.sum(dim=(1, 2, 3)) / compare.numel() * compare.shape[0]) 
                print(target_tensor)

        # return total_correct / total_samples

    def train(self, epochs: int = 1000) -> None:
        assert isinstance(epochs, int) and epochs > 0

        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss: float = 0.0
            for video_tensor, patch_tensor, center_tensor, target_tensor in tqdm(
                self.train_loader, desc=f"Epoch {epoch}/{epochs}", ncols=80
            ):  
                # TODO
                target_tensor = target_tensor.to(self.device)
                video_tensor = video_tensor.to(self.device).float()
                center_tensor = center_tensor.to(self.device)
                velocity_vector: torch.Tensor = self.model(input_video=video_tensor)
                # Loss
                loss: torch.Tensor = self.motion_matching_loss(
                    center_tensor=center_tensor, velocity_vector=velocity_vector,
                )
                print(loss)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += float(loss.item())

            print(f"Epoch {epoch}: total_loss = {total_loss:.4f}")
            if epoch % 5 == 0:
                torch.save(self.model.state_dict(), f"epoch{epoch}.pt")

            train_acc: float = self.evaluate(self.train_loader)#; print(f"Train Accuracy: {train_acc:.4f}")
            val_acc: float = self.evaluate(self.val_loader)#; print(f"Val Accuracy: {val_acc:.4f}")
            print("-------------")

    def test(self, checkpoint_path: str) -> float:
        assert isinstance(checkpoint_path, str)
        checkpoint: dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        return self.evaluate(self.test_loader)
    

if __name__ == "__main__":

    dataset: MoverDataset = MoverDataset(
        data_directory="/scratch/zgp2ps/movers/data", cache_directory="/scratch/zgp2ps/movers/cache",
        # data_directory="./data", cache_directory="./cache",   # should use this for your machine
        max_circles=256, patch_size=12,
    )
    device: torch.device = torch.device("cuda")
    # # model = SimpleMLP(embedding_dim=256)
    # # model = SimpleTransformer(embedding_dim=32)
    # model = SimpleCNN(embedding_dim=16, dropout=0.)
    # # model = SimpleGRU(hidden_dim=64)
    # model = DeepSetClassifier(hidden_dim=64)
    model = CollectiveMotionDetector(embedding_dim=8, dropout=0.)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    trainer: Trainer = Trainer(dataset=dataset, model=model, device=device, lr=1e-5)
    trainer.train(epochs=100)
    test_accuracy: float = trainer.test("epoch100.pt")
    print(f"Test Accuracy: {test_accuracy:.4f}")

