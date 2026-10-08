"""Подготовка датасета YOLO-seg из экспорта Label Studio: разбиение листов на train и val и нарезка на плитки.

Файл самодостаточный: пакет draftslice не импортируется, нужны только numpy и opencv.
`make_dataset()` - единственная публичная функция файла, шаги - приватные функции.

Разбиение идёт по листам, а не по объектам: стрелки одного листа похожи друг на друга, и при разбиении по объектам
val видел бы обучающие листы. Листы выбираются так, чтобы доля объектов в val была ближе всего к `val_share`, при
равенстве решает `seed`. С теми же `val_share`, `seed` и `val` режим с плитками и режим без них дают одно и то же
разбиение, поэтому обученные на них модели сравнимы на одних val-листах.

Нарезка повторяет сетку SAHI (`sahi.slicing.get_slice_bboxes`), чтобы обучение и инференс видели плитки одинаково.
Полигоны обрезаются по плитке; объект остаётся, если в плитке не меньше `min_area` его площади. Плитка, в которой
есть обрезанный, но не оставленный объект, фоном не берётся: иначе в ней стрелка была бы видна, но не размечена.

Запуск: python training/make_dataset.py [--src data/dataset] [--tile 640 | --no-tile] [--overlap 0.2]
"""

import argparse
import itertools
import json
import random
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png"}
EXHAUSTIVE_MAX = 12  # до стольких листов перебираем все разбиения, дальше - случайные
TRIALS = 5000


@dataclass(frozen=True)
class Split:
    """Результат разбиения.

    Attributes
    ----------
    train, val : tuple[str, ...]
        Имена листов (без расширения) в обучающей и проверочной выборке.
    objects : dict[str, int]
        Число объектов на каждом листе.
    images : dict[str, int]
        Число изображений в `dst` по выборкам (плиток или целых листов).
    yaml : Path
        Путь к `data.yaml` для Ultralytics.
    """

    train: tuple[str, ...]
    val: tuple[str, ...]
    objects: dict[str, int]
    images: dict[str, int]
    yaml: Path


def make_dataset(
    src: Path,
    dst: Path,
    tile: int | None = 640,
    overlap: float = 0.2,
    min_area: float = 0.5,
    bg_ratio: float = 0.25,
    val_share: float = 0.2,
    seed: int = 0,
    val: Sequence[str] | None = None,
    overwrite: bool = False,
) -> Split:
    """Разбивает листы экспорта Label Studio на train и val и пишет датасет в формате Ultralytics.

    Parameters
    ----------
    src : Path
        Папка экспорта: `images/`, `labels/` (YOLO-seg) и `classes.txt`.
    dst : Path
        Куда писать: `images/{train,val}`, `labels/{train,val}`, `data.yaml`, `split.json`. В `data.yaml` нет ключа
        `path`: корнем считается папка файла, и датасет переносится копированием папки.
    tile : int | None
        Сторона плитки, px, в исходном масштабе листа. `None` - без нарезки, листы копируются целиком.
    overlap : float
        Перекрытие плиток, доля стороны, в [0, 1).
    min_area : float
        Доля площади объекта, которая должна остаться в плитке, чтобы объект остался в разметке, в (0, 1].
    bg_ratio : float
        Плиток без объектов в train, доля от числа плиток с объектами. В val остаются все.
    val_share : float
        Желаемая доля объектов в val, в (0, 1).
    seed : int
        Решает выбор между разбиениями с одинаково близкой долей и выбор фоновых плиток.
    val : Sequence[str] | None
        Листы для val вручную: имя файла без расширения или номер листа после последнего `-`.
        Тогда `val_share` не используется.
    overwrite : bool
        Заменить уже существующий датасет в `dst`.

    Returns
    -------
    Split
        Листы по выборкам, число объектов, число изображений и путь к `data.yaml`.

    Raises
    ------
    FileNotFoundError
        Нет `images/`, `classes.txt` или файла разметки листа.
    ValueError
        Некорректные параметры, листов меньше двух, некорректная строка разметки, `val` не находит лист, в train или
        val нет объектов какого-то класса, `dst` уже заполнен при `overwrite=False`.
    """
    if not 0 < val_share < 1 or not 0 <= overlap < 1 or not 0 < min_area <= 1 or bg_ratio < 0 or tile == 0:
        raise ValueError(f"Некорректные параметры: {val_share=}, {overlap=}, {min_area=}, {bg_ratio=}, {tile=}")
    names = _read_classes(src)
    sheets = _read_sheets(src, len(names))
    stems = sorted(sheets)
    if len(stems) < 2:
        raise ValueError(f"Нужно хотя бы два листа, найдено {len(stems)}.")
    counts = {s: len(sheets[s][1]) for s in stems}
    val_set = _resolve(val, stems) if val else _choose_val(counts, val_share, seed)
    train, held = tuple(s for s in stems if s not in val_set), tuple(s for s in stems if s in val_set)
    _check_classes(names, sheets, train, held)
    _prepare_dst(dst, overwrite)
    images = {}
    for part, group in (("train", train), ("val", held)):
        if tile is None:
            images[part] = _copy_sheets(sheets, group, src, dst, part)
        else:
            keep_bg = bg_ratio if part == "train" else None
            images[part] = _write_tiles(sheets, group, dst, part, tile, overlap, min_area, keep_bg, seed)
    yaml = dst / "data.yaml"
    yaml.write_text(_yaml(names), encoding="utf-8")
    info = {
        "train": train,
        "val": held,
        "objects": counts,
        "images": images,
        "tile": tile,
        "overlap": overlap,
        "min_area": min_area,
        "bg_ratio": bg_ratio,
        "val_share": val_share,
        "seed": seed,
        "manual_val": bool(val),
    }
    (dst / "split.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return Split(train, held, counts, images, yaml)


def _read_classes(src: Path) -> list[str]:
    path = src / "classes.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Нет файла классов: {path}")
    names = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not names:
        raise ValueError(f"Файл классов пуст: {path}")
    return names


def _read_sheets(src: Path, n_cls: int) -> dict[str, tuple[Path, list[tuple[int, np.ndarray]]]]:
    """Лист -> (путь к изображению, объекты: класс и полигон (n, 2) в долях ширины и высоты)."""
    folder = src / "images"
    if not folder.is_dir():
        raise FileNotFoundError(f"Нет папки изображений: {folder}")
    sheets = {}
    for img in sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT):
        label = src / "labels" / f"{img.stem}.txt"
        if not label.is_file():
            raise FileNotFoundError(f"Нет разметки для {img.name}: {label}")
        sheets[img.stem] = (img, _read_label(label, n_cls))
    return sheets


def _read_label(path: Path, n_cls: int) -> list[tuple[int, np.ndarray]]:
    """Строки YOLO-seg: класс и не менее трёх точек с координатами в [0, 1]."""
    out = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        tok = line.split()
        try:
            cls, xy = int(tok[0]), [float(t) for t in tok[1:]]
        except ValueError as ex:
            raise ValueError(f"{path.name}:{i}: не число в строке разметки") from ex
        if not 0 <= cls < n_cls:
            raise ValueError(f"{path.name}:{i}: класс {cls} вне 0..{n_cls - 1}")
        if len(xy) < 6 or len(xy) % 2:
            raise ValueError(f"{path.name}:{i}: нужно не менее трёх точек (x y), получено {len(xy)} чисел")
        if min(xy) < -1e-6 or max(xy) > 1 + 1e-6:
            raise ValueError(f"{path.name}:{i}: координаты вне [0, 1]")
        out.append((cls, np.array(xy, np.float64).reshape(-1, 2)))
    return out


def _resolve(tokens: Sequence[str], stems: list[str]) -> set[str]:
    """Имена листов val: полное имя или номер листа после последнего дефиса."""
    found = set()
    for t in tokens:
        hit = [s for s in stems if s == t] or [s for s in stems if s.rsplit("-", 1)[-1] == t]
        if len(hit) != 1:
            raise ValueError(f"Лист {t!r} для val: найдено {len(hit)} совпадений среди {stems}")
        found.add(hit[0])
    return found


def _choose_val(counts: dict[str, int], val_share: float, seed: int) -> set[str]:
    """Набор листов, у которого доля объектов ближе всего к val_share; при равенстве решает seed."""
    stems, total, rng = sorted(counts), sum(counts.values()), random.Random(seed)
    n = len(stems)
    if n <= EXHAUSTIVE_MAX:
        cands = [c for r in range(1, n) for c in itertools.combinations(stems, r)]
    else:
        cands = [tuple(rng.sample(stems, rng.randint(1, n - 1))) for _ in range(TRIALS)]
    rng.shuffle(cands)  # min берёт первый из равных: порядок решает seed
    best = min(cands, key=lambda c: abs(sum(counts[s] for s in c) / total - val_share) if total else 0.0)
    return set(best)


def _check_classes(names: list[str], sheets: dict, train: tuple[str, ...], held: tuple[str, ...]) -> None:
    for part, group in (("train", train), ("val", held)):
        seen = {c for s in group for c, _ in sheets[s][1]}
        missing = [names[i] for i in range(len(names)) if i not in seen]
        if missing:
            raise ValueError(f"В {part} нет объектов классов {missing}: нужно поменять val или seed.")


def _prepare_dst(dst: Path, overwrite: bool) -> None:
    """Пустая папка или явное разрешение; чистим только то, что пишет сам скрипт."""
    if dst.exists() and any(dst.iterdir()) and not overwrite:
        raise ValueError(f"{dst} не пуста: укажите overwrite=True (--overwrite).")
    for p in (dst / "images", dst / "labels", dst / "data.yaml", dst / "split.json"):
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()
    dst.mkdir(parents=True, exist_ok=True)


def _yaml(names: list[str]) -> str:
    """Без ключа path: Ultralytics берёт за корень папку yaml-файла (check_det_dataset)."""
    rows = "\n".join(f"  {i}: {json.dumps(n, ensure_ascii=False)}" for i, n in enumerate(names))
    return f"train: images/train\nval: images/val\nnames:\n{rows}\n"


def _copy_sheets(sheets: dict, group: tuple[str, ...], src: Path, dst: Path, part: str) -> int:
    for s in group:
        for sub, file in (("images", sheets[s][0]), ("labels", src / "labels" / f"{s}.txt")):
            (dst / sub / part).mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, dst / sub / part / file.name)
    return len(group)


def _write_tiles(
    sheets: dict,
    group: tuple[str, ...],
    dst: Path,
    part: str,
    tile: int,
    overlap: float,
    min_area: float,
    bg_ratio: float | None,
    seed: int,
) -> int:
    """Плитки листов группы: все с объектами и фон (bg_ratio от их числа; None - весь фон)."""
    shapes = {s: cv2.imread(str(sheets[s][0]), cv2.IMREAD_UNCHANGED).shape[:2] for s in group}
    pos, neg = [], []
    for s in group:
        h, w = shapes[s]
        for box in _grid(h, w, tile, overlap):
            lines, ambiguous = _tile_labels(sheets[s][1], box, w, h, min_area)
            if lines:
                pos.append((s, box, lines))
            elif not ambiguous:
                neg.append((s, box, lines))
    if bg_ratio is not None:
        neg = random.Random(seed).sample(neg, min(len(neg), round(bg_ratio * len(pos))))
    chosen = {(s, box): lines for s, box, lines in pos + neg}
    for sub in ("images", "labels"):
        (dst / sub / part).mkdir(parents=True, exist_ok=True)
    for s in group:
        todo = {box: ln for (name, box), ln in chosen.items() if name == s}
        if not todo:
            continue
        img = cv2.imread(str(sheets[s][0]), cv2.IMREAD_COLOR)
        for (x0, y0, x1, y1), lines in todo.items():
            name = f"{s}_{x0}_{y0}"
            cv2.imwrite(str(dst / "images" / part / f"{name}.png"), img[y0:y1, x0:x1])
            (dst / "labels" / part / f"{name}.txt").write_text("".join(f"{ln}\n" for ln in lines), encoding="utf-8")
    return len(chosen)


def _grid(h: int, w: int, tile: int, overlap: float) -> list[tuple[int, int, int, int]]:
    """Плитки (x0, y0, x1, y1) как у `sahi.slicing.get_slice_bboxes`: последняя плитка прижата к краю."""
    ov = int(overlap * tile)
    boxes, y_min, y_max = [], 0, 0
    while y_max < h:
        x_min = x_max = 0
        y_max = y_min + tile
        while x_max < w:
            x_max = x_min + tile
            if y_max > h or x_max > w:
                xe, ye = min(w, x_max), min(h, y_max)
                boxes.append((max(0, xe - tile), max(0, ye - tile), xe, ye))
            else:
                boxes.append((x_min, y_min, x_max, y_max))
            x_min = x_max - ov
        y_min = y_max - ov
    return boxes


def _tile_labels(objs: list, box: tuple[int, int, int, int], w: int, h: int, min_area: float) -> tuple[list, bool]:
    """Строки разметки плитки и признак «есть объект, видимый частично, но не оставленный»."""
    x0, y0, x1, y1 = box
    lines, ambiguous = [], False
    for cls, poly in objs:
        px = poly * (w, h)
        full = _area(px)
        if full <= 0:
            continue
        clipped = _clip(px, box)
        ratio = _area(clipped) / full if clipped is not None else 0.0
        if ratio >= min_area:
            xy = (clipped - (x0, y0)) / (x1 - x0, y1 - y0)
            lines.append(f"{cls} " + " ".join(f"{v:.6f}" for v in np.clip(xy, 0, 1).ravel()))
        elif ratio > 0:
            ambiguous = True
    return lines, ambiguous


def _area(poly: np.ndarray | None) -> float:
    """Площадь многоугольника (формула шнурования)."""
    if poly is None or len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


def _clip(poly: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray | None:
    """Обрезка многоугольника по прямоугольнику (Сазерленд - Ходжман); None, если ничего не осталось."""
    x0, y0, x1, y1 = box
    edges = [
        (lambda p: p[0] >= x0, lambda a, b: _cut(a, b, 0, x0)),
        (lambda p: p[0] <= x1, lambda a, b: _cut(a, b, 0, x1)),
        (lambda p: p[1] >= y0, lambda a, b: _cut(a, b, 1, y0)),
        (lambda p: p[1] <= y1, lambda a, b: _cut(a, b, 1, y1)),
    ]
    pts = [tuple(p) for p in poly]
    for inside, cross in edges:
        out = []
        for a, b in zip(pts, pts[1:] + pts[:1], strict=True):
            if inside(b):
                if not inside(a):
                    out.append(cross(a, b))
                out.append(b)
            elif inside(a):
                out.append(cross(a, b))
        pts = out
        if not pts:
            return None
    res = np.array(pts, np.float64)
    return res if _area(res) > 0 else None


def _cut(a: tuple, b: tuple, axis: int, value: float) -> tuple[float, float]:
    """Точка пересечения отрезка ab с прямой coordinate[axis] = value."""
    t = (value - a[axis]) / (b[axis] - a[axis])
    return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", type=Path, default=Path("data/dataset"))
    ap.add_argument("--dst", type=Path, help="по умолчанию data/dataset_yolo/tiles или data/dataset_yolo/full")
    ap.add_argument("--tile", type=int, default=640, help="сторона плитки, px")
    ap.add_argument("--no-tile", action="store_true", help="не резать, листы копируются целиком")
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--min-area", type=float, default=0.5)
    ap.add_argument("--bg-ratio", type=float, default=0.25)
    ap.add_argument("--val-share", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val", nargs="+", help="листы для val вручную: номер листа или имя файла")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    tile = None if a.no_tile else a.tile
    dst = a.dst or Path("data/dataset_yolo") / ("full" if tile is None else "tiles")
    res = make_dataset(a.src, dst, tile, a.overlap, a.min_area, a.bg_ratio, a.val_share, a.seed, a.val, a.overwrite)
    total = sum(res.objects.values())
    print(f"{'выборка':8s} {'листов':>6s} {'объектов':>8s} {'доля':>6s} {'изображений':>12s}")
    for part, group in (("train", res.train), ("val", res.val)):
        n = sum(res.objects[s] for s in group)
        print(f"{part:8s} {len(group):6d} {n:8d} {n / total:6.1%} {res.images[part]:12d}   {', '.join(group)}")
    print(f"data.yaml: {res.yaml}")


if __name__ == "__main__":
    main()
