"""Кеш артефактов между этапными блокнотами.

Каждый этап дорог по времени: детекция текста тянет onnx, скелетизация работает на кадре в тысячи
пикселей. Поэтому результат этапа кладется на диск, а следующий блокнот его читает, вместо того
чтобы пересчитывать пайплайн с начала.

Ключом служат имя исходного файла, имя этапа и хеш параметров, при которых артефакт получен. Смена
любого порога дает другое имя файла, поэтому устаревший артефакт не подхватится молча, а рядом
лежит json с самими параметрами, чтобы по каталогу было видно, чем получен каждый результат.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

CACHE_ROOT = Path(".cache")
"""Каталог кеша по умолчанию, рядом с блокнотами."""


def hash_parameters(parameters: dict[str, Any]) -> str:
    """Короткий хеш набора параметров.

    Parameters
    ----------
    parameters : dict[str, Any]
        Параметры этапа, любые json-совместимые значения.

    Returns
    -------
    str
        Первые десять символов sha1 от канонического представления параметров.
    """
    canonical = json.dumps(parameters, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:10]


@dataclass(frozen=True)
class ArtifactCache:
    """Каталог с артефактами этапов.

    Attributes
    ----------
    root : Path
        Корень кеша.
    """

    root: Path = CACHE_ROOT

    def stage_path(self, source_path: Path, stage_name: str, parameters: dict[str, Any]) -> Path:
        """Путь к артефакту этапа.

        Parameters
        ----------
        source_path : Path
            Исходный файл чертежа.
        stage_name : str
            Имя этапа.
        parameters : dict[str, Any]
            Параметры, при которых получен артефакт.

        Returns
        -------
        Path
            Путь к файлу npz.
        """
        return self.root / source_path.stem / f"{stage_name}__{hash_parameters(parameters)}.npz"

    def save(self, source_path: Path, stage_name: str, parameters: dict[str, Any], **arrays: np.ndarray) -> Path:
        """Сохранить массивы этапа вместе с его параметрами.

        Parameters
        ----------
        source_path : Path
            Исходный файл чертежа.
        stage_name : str
            Имя этапа.
        parameters : dict[str, Any]
            Параметры, при которых получен артефакт.
        **arrays : np.ndarray
            Массивы, которые надо сохранить.

        Returns
        -------
        Path
            Путь к записанному файлу npz.
        """
        target = self.stage_path(source_path, stage_name, parameters)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, **arrays)
        target.with_suffix(".json").write_text(
            json.dumps(parameters, sort_keys=True, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return target

    def load(self, source_path: Path, stage_name: str, parameters: dict[str, Any]) -> dict[str, np.ndarray] | None:
        """Прочитать артефакт этапа, если он посчитан с теми же параметрами.

        Parameters
        ----------
        source_path : Path
            Исходный файл чертежа.
        stage_name : str
            Имя этапа.
        parameters : dict[str, Any]
            Параметры, при которых артефакт должен быть получен.

        Returns
        -------
        dict[str, np.ndarray] | None
            Массивы артефакта или None, если его нет.
        """
        target = self.stage_path(source_path, stage_name, parameters)
        if not target.exists():
            return None
        with np.load(target) as stored:
            return {name: stored[name] for name in stored.files}
