import numpy as np
from typing import Tuple, Dict, Optional, List
from pathlib import Path
import imageio.v3 as iio


class DataGenerator:

    def __init__(
        self,
        n_frames: int = 32,
        n_circles: int = 1000,
        image_size: Tuple[int, int] = (360, 480),
        radius: int = 6,
        speed: int = 5,
        label_mover_ratio: float = 0.5,
        directions: Optional[Dict[str, Tuple[int, int]]] = None,
        out_dir: Path = Path("data")
    ) -> None:
        assert n_frames > 0
        assert n_circles > 0
        assert 0 < label_mover_ratio <= 1.0
        assert len(image_size) == 2 and all(i > 0 for i in image_size)

        self.n_frames: int = n_frames
        self.n_circles: int = n_circles
        self.H, self.W = image_size
        self.radius: int = radius
        self.speed: int = speed
        self.label_mover_ratio: float = label_mover_ratio
        self.out_dir: Path = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.directions: Dict[str, Tuple[int, int]] = directions or {
            "left": (-speed, 0),
            "right": (speed, 0),
            "up": (0, -speed),
            "down": (0, speed),
        }
        assert all(len(v) == 2 for v in self.directions.values())
        self.dir_keys: List[str] = list(self.directions.keys())
        assert len(self.dir_keys) >= 2

    def generate_data_samples(self, n_samples: int) -> List[Path]:
        assert n_samples > 0
        return [self._generate_video(i) for i in range(n_samples)]

    def save_gifs(self, npy_paths: List[Path], fps: int = 10) -> None:
        gif_dir: Path = self.out_dir.parent / "gifs"
        gif_dir.mkdir(parents=True, exist_ok=True)

        for npy_path in npy_paths:
            assert npy_path.exists() and npy_path.suffix == ".npy"
            video = np.load(npy_path)
            assert video.ndim == 3, f"Expected (T, H, W), got {video.shape}"
            gif_path = gif_dir / (npy_path.stem + ".gif")
            iio.imwrite(gif_path, video, format="GIF", fps=fps)
            print(f"Saved GIF: {gif_path}")

    def _make_circle(self, frame: np.ndarray, center: Tuple[int, int], value: int = 255) -> None:
        assert frame.ndim == 2
        Y, X = np.ogrid[:self.H, :self.W]
        dist = (X - center[0]) ** 2 + (Y - center[1]) ** 2
        frame[dist <= self.radius ** 2] = value

    def _generate_video(self, idx: int) -> Path:
        assert idx >= 0
        has_mover: bool = np.random.rand() < self.label_mover_ratio
        label: int = int(has_mover)
        bg_dir, sel_dir = (
            np.random.choice(self.dir_keys, 2, replace=False)
            if has_mover
            else (np.random.choice(self.dir_keys),) * 2
        )

        centers: List[Tuple[int, int]] = []
        for j in range(self.n_circles):
            cx = np.random.randint(-self.W // 2, self.W + self.W // 2) if j != 0 else np.random.randint(self.W // 4, 3 * self.W // 4)
            cy = np.random.randint(-self.H // 2, self.H + self.H // 2) if j != 0 else np.random.randint(self.H // 4, 3 * self.H // 4)
            centers.append((cx, cy))
        centers_arr: np.ndarray = np.array(centers, dtype=np.int32)

        bg_motion: np.ndarray = np.array(self.directions[bg_dir], dtype=np.int32)
        sel_motion: np.ndarray = np.array(self.directions[sel_dir], dtype=np.int32)

        video: np.ndarray = np.zeros((self.n_frames, self.H, self.W), dtype=np.uint8)
        for t in range(self.n_frames):
            frame: np.ndarray = np.zeros((self.H, self.W), dtype=np.uint8)
            deltas: np.ndarray = np.tile(bg_motion, (self.n_circles, 1))
            deltas[0] = sel_motion
            centers_arr += deltas
            for cx, cy in centers_arr:
                self._make_circle(frame, (int(cx), int(cy)))
            video[t] = frame

        filename: Path = self.out_dir / f"vid{idx}_{label}.npy"
        np.save(filename, video)
        print(f"Saved: {filename}")
        return filename

if __name__ == "__main__":
    generator = DataGenerator()
    paths = generator.generate_data_samples(n_samples=10_000)
    generator.save_gifs(paths, fps=20)
