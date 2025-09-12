import logging
import pathlib

import imageio
import numpy as np
from openpi_client.runtime import subscriber as _subscriber
from typing_extensions import override


class VideoSaver(_subscriber.Subscriber):
    """Saves episode data as videos for RoboCasa."""

        
    def __init__(self, out_dir: pathlib.Path,save_video: bool =True, subsample: int = 1) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self._out_dir = out_dir
        self._images: list[np.ndarray] = []
        self._save_video = save_video
        self._subsample = subsample

    @override
    def on_episode_start(self) -> None:
        self._images = []

    @override
    def on_step(self, observation: dict, action: dict) -> None:
        # Extract the main camera image (using standardized key name)
        if not self._save_video:
            return
        if "images" in observation and "cam_right_wrist" in observation["images"]:
            im = observation["images"]["cam_right_wrist"]  # [C, H, W]
            im = np.transpose(im, (1, 2, 0))  # [H, W, C]
            self._images.append(im)
            # import matplotlib.pyplot as plt
            # plt.imshow(im)
            # plt.axis("off")   # 不显示坐标轴
            # plt.show()

    @override
    def on_episode_end(self) -> None:
        if not self._save_video:
            return
        if not self._images:
            logging.warning("No images captured during episode")
            return
            
        existing = list(self._out_dir.glob("robocasa_[0-9]*.mp4"))
        next_idx = max([int(p.stem.split("_")[1]) for p in existing], default=-1) + 1
        out_path = self._out_dir / f"robocasa_{next_idx}.mp4"

        logging.info(f"Saving RoboCasa video to {out_path}")
        imageio.mimwrite(
            out_path,
            [np.asarray(x) for x in self._images[:: self._subsample]],
            fps=50 // max(1, self._subsample),
        )
