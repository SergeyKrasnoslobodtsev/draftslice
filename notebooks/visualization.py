import numpy as np
from matplotlib import pyplot as plt

from draftslice.core.common.types import Floating, Mat


def show_bgr_image(image: Mat, title: str = "", figsize: tuple[float, float] = (16, 8)) -> None:
    plt.figure(figsize=figsize)
    plt.imshow(image[..., ::-1])
    plt.axis("off")
    plt.title(title)
    plt.show()


def show_prob_map(prob: Floating):
    plt.figure(figsize=(16, 8))
    plt.imshow(prob, cmap="inferno", vmin=0, vmax=1)
    plt.title("Итоговая карта вероятности")
    plt.axis("off")
    plt.show()


def show_probs(probs: Floating, long_sides: tuple[int, ...]):
    figure, axes = plt.subplots(1, len(long_sides), figsize=(5 * len(long_sides), 5))
    for axis, probability_map, long_side in zip(np.atleast_1d(axes), probs, long_sides, strict=True):
        axis.imshow(probability_map, cmap="inferno", vmin=0, vmax=1)
        axis.set_title(f"масштаб {long_side}")
        axis.axis("off")
    figure.tight_layout()
    plt.show()
