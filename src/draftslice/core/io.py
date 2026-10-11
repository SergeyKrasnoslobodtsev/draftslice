from pathlib import Path

import cv2
import numpy as np


def load_image(src: str | Path | bytes | np.ndarray) -> np.ndarray:
    """Загружает лист RGB из файла, из байтов или возвращает готовый массив.

    Parameters
    ----------
    src : str | Path | bytes | np.ndarray
        Путь к файлу, байты файла изображения или массив `(H, W, 3)`.

    Returns
    -------
    np.ndarray
        Лист RGB `(H, W, 3)`, `uint8`.

    Raises
    ------
    FileNotFoundError
        Файла нет.
    ValueError
        Изображение не декодируется или массив не трёхканальный.
    """
    if isinstance(src, np.ndarray):
        if src.ndim != 3 or src.shape[-1] != 3:
            raise ValueError(f"Ожидается RGB (H, W, 3), получено: {src.shape}")
        return src
    if isinstance(src, (str, Path)):
        if not Path(src).is_file():
            raise FileNotFoundError(f"Файл не найден: {src}")
        buf = np.fromfile(src, dtype=np.uint8)  # не cv2.imread: путь с кириллицей на Windows
    else:
        buf = np.frombuffer(src, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR_RGB)
    if img is None:
        raise ValueError(f"Не удалось декодировать изображение: {src if isinstance(src, (str, Path)) else 'bytes'}")
    return img


def save_image(path: str | Path, img: np.ndarray) -> Path:
    """Сохраняет лист RGB или маску в файл; формат - по расширению.

    Parameters
    ----------
    path : str | Path
        Путь файла; папки создаются.
    img : np.ndarray
        RGB `(H, W, 3)` или маска `(H, W)`, `uint8`.

    Returns
    -------
    Path
        Путь сохранённого файла.

    Raises
    ------
    ValueError
        Изображение не кодируется в формат расширения.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix, cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.ndim == 3 else img)
    if not ok:
        raise ValueError(f"Не удалось закодировать изображение в {path.suffix}")
    buf.tofile(path)  # не cv2.imwrite: путь с кириллицей на Windows
    return path
