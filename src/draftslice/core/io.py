import cv2
import numpy as np

from draftslice.core.common.types import Mat


def load_image(filename: str) -> Mat:

    buff = np.fromfile(file=filename, dtype=np.uint8)

    image = cv2.imdecode(buff, flags=cv2.IMREAD_COLOR)
    if image is None:
        ValueError("Ошибка чтения изображения.")

    return image # pyright: ignore[reportReturnType]
