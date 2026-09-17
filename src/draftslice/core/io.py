import os

from draftslice.core.exceptions import DraftsliceFileNotFoundError, DraftsliceValueError, IORuntimeError
from draftslice.core.types import Mat


def load_image(filename: str | bytes | Mat) -> Mat:
    """Загружает изображение из файла, из байтового массива или возвращает его, если это уже Mat.

    Parameters
    ----------
    filename : str | bytes | Mat
        Изображение в виде пути к файлу, байтового массива или объекта Mat.

    Returns
    -------
    Mat
        Загруженное изображение в формате Mat.

    Raises
    ------
    DraftsliceFileNotFoundError
        Если файл не найден.
    DraftsliceValueError
        Если изображение не удалось загрузить или имеет неверный формат.
    DraftsliceValueError
        Если изображение не является цветным изображением с тремя каналами.
    IORuntimeError
        Если произошла ошибка ввода-вывода при загрузке изображения.
    """
    img = None

    if isinstance(filename, str):
        import cv2
        import numpy as np

        if not os.path.exists(filename):
            raise DraftsliceFileNotFoundError(f"Файл не найден: {filename}")

        buffer = np.fromfile(filename, dtype=np.uint8)
        img = cv2.imdecode(buffer, cv2.IMREAD_COLOR_RGB)

    if isinstance(filename, bytes):
        import cv2
        import numpy as np

        buffer = np.frombuffer(filename, dtype=np.uint8)
        img = cv2.imdecode(buffer, cv2.IMREAD_COLOR_RGB)

    if isinstance(filename, Mat):
        if filename.ndim != 3 or filename.shape[-1] != 3:
            raise DraftsliceValueError(f"Неверный формат изображения {filename}")
        img = filename

    if img is None:
        raise IORuntimeError(f"Не удалось загрузить {filename}")
    return img
