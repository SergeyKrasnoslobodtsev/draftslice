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
class StoredArtifact:
    """Прочитанный артефакт этапа.

    Attributes
    ----------
    arrays : dict[str, np.ndarray]
        Массивы, сохраненные этапом.
    parameters : dict[str, Any]
        Параметры, при которых артефакт получен.
    path : Path
        Путь к файлу npz.
    """

    arrays: dict[str, np.ndarray]
    parameters: dict[str, Any]
    path: Path


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

    def has_stage(self, source_path: Path, stage_name: str) -> bool:
        """Проверить, считался ли этап для этого файла.

        Parameters
        ----------
        source_path : Path
            Исходный файл чертежа.
        stage_name : str
            Имя этапа.

        Returns
        -------
        bool
            True, если в кеше есть хотя бы один артефакт этапа.
        """
        return any((self.root / source_path.stem).glob(f"{stage_name}__*.npz"))

    def load_latest(self, source_path: Path, stage_name: str) -> StoredArtifact | None:
        """Прочитать самый свежий артефакт этапа независимо от параметров.

        Parameters
        ----------
        source_path : Path
            Исходный файл чертежа.
        stage_name : str
            Имя этапа.

        Returns
        -------
        StoredArtifact | None
            Массивы вместе с параметрами, при которых они получены, или None, если этап не считался.

        Notes
        -----
        Нужен следующему по цепочке блокноту: он не знает, какими порогами получен предыдущий этап, и
        берет последний посчитанный результат. Параметры возвращаются вместе с массивами, чтобы их
        можно было напечатать и не гадать, на чем построен вход.
        """
        stage_files = sorted(
            (self.root / source_path.stem).glob(f"{stage_name}__*.npz"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        if not stage_files:
            return None

        target = stage_files[0]
        parameters_path = target.with_suffix(".json")
        parameters = json.loads(parameters_path.read_text(encoding="utf-8")) if parameters_path.exists() else {}
        with np.load(target) as stored:
            arrays = {name: stored[name] for name in stored.files}
        return StoredArtifact(arrays=arrays, parameters=parameters, path=target)
