import numpy as np
from typing import *
from pathlib import Path
import imageio.v3 as iio
from multiprocessing import Pool, cpu_count


class DataGenerator:

    def __init__(
        self,
        n_frames: int = 32,
        n_circles: int = 1000,
        image_size: Tuple[int, int] = (360, 480),
        radius: int = 6,
        speed: int = 5,
        label_mover_ratio: float = 0.5,
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

        self.directions: Dict[str, Tuple[int, int]] = {
            "left": (-speed, 0),
            "right": (speed, 0),
            "up": (0, -speed),
            "down": (0, speed),
        }
        assert all(len(v) == 2 for v in self.directions.values())
        self.dir_keys: List[str] = list(self.directions.keys())

    #parallel
    def generate_data_samples(self, n_samples: int, n_cpus: int) -> List[Path]:
        config: Dict[str, Any] = {
            'n_frames': self.n_frames,
            'n_circles': self.n_circles,
            'H': self.H,
            'W': self.W,
            'radius': self.radius,
            'speed': self.speed,
            'label_mover_ratio': self.label_mover_ratio,
            'directions': self.directions,
            'dir_keys': self.dir_keys,
            'out_dir': self.out_dir
        }
        with Pool(processes=n_cpus) as pool:
            return pool.map(DataGenerator._generate_video_wrapper, [(i, config) for i in range(n_samples)])

    def save_gifs(self, npy_paths: List[Path], fps: int = 10) -> None:
        gif_dir: Path = self.out_dir.parent / "gifs"
        gif_dir.mkdir(parents=True, exist_ok=True)

        for npy_path in npy_paths:
            assert npy_path.exists() and npy_path.suffix == ".npy"
            video = np.load(npy_path)
            assert video.ndim == 3, f"Expected (T, H, W), got {video.shape}"
            gif_path: Path = gif_dir / (npy_path.stem + ".gif")
            iio.imwrite(gif_path, video, format="GIF", fps=fps)
            print(f"Saved GIF: {gif_path}")

    @staticmethod
    def _make_circle(frame: np.ndarray, center: Tuple[int, int], radius: int, value: int = 255) -> None:
        H, W = frame.shape
        Y, X = np.ogrid[:H, :W]
        dist: np.ndarray = (X - center[0]) ** 2 + (Y - center[1]) ** 2
        frame[dist <= radius ** 2] = value

    @staticmethod
    def _generate_video(idx: int, cfg: Dict) -> Path:
        has_mover: bool = np.random.rand() < cfg['label_mover_ratio']
        label: int = int(has_mover)
        bg_dir, sel_dir = (
            np.random.choice(cfg['dir_keys'], 2, replace=False)
            if has_mover else (np.random.choice(cfg['dir_keys']),) * 2
        )

        centers: List[Tuple[np.ndarray, np.ndarray]] = [
            (np.random.randint(cfg['W'] // 4, 3 * cfg['W'] // 4),
             np.random.randint(cfg['H'] // 4, 3 * cfg['H'] // 4)) if j == 0
            else (np.random.randint(-cfg['W'] // 2, cfg['W'] + cfg['W'] // 2),
                  np.random.randint(-cfg['H'] // 2, cfg['H'] + cfg['H'] // 2))
            for j in range(cfg['n_circles'])
        ]
        centers_arr: np.ndarray = np.array(centers, dtype=np.int32)

        bg_motion: np.ndarray = np.array(cfg['directions'][bg_dir], dtype=np.int32)
        sel_motion: np.ndarray = np.array(cfg['directions'][sel_dir], dtype=np.int32)

        video: np.ndarray = np.zeros((cfg['n_frames'], cfg['H'], cfg['W']), dtype=np.uint8)
        for t in range(cfg['n_frames']):
            frame: np.ndarray = np.zeros((cfg['H'], cfg['W']), dtype=np.uint8)
            deltas: np.ndarray = np.tile(bg_motion, (cfg['n_circles'], 1))
            deltas[0] = sel_motion
            centers_arr += deltas
            for cx, cy in centers_arr:
                DataGenerator._make_circle(frame, (int(cx), int(cy)), cfg['radius'])
            video[t] = frame

        filename: Path = cfg['out_dir'] / f"vid{idx}_{label}.npy"
        np.save(filename, video)
        print(f"Saved: {filename}")
        return filename
    
    @staticmethod
    def _generate_video_wrapper(args: Tuple[int, Dict]) -> Path:
        idx, config = args
        return DataGenerator._generate_video(idx, config)


if __name__ == "__main__":
    generator: DataGenerator = DataGenerator(out_dir=Path("/scratch/zgp2ps/movers/data"))
    paths: List[Path] = generator.generate_data_samples(n_samples=10_000, n_cpus=cpu_count())
    generator.save_gifs(paths, fps=20)


