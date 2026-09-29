"""Стенд: статистика ширины штриха по всем листам - основа для классификации без ручных констант.

Прогоняет ячейки exp_classifier_lines.ipynb до ячейки ширины включительно (кропы, OCR, апскейл,
скелет, 2 * расстояние до фона), поэтому меряет ровно то же, что блокнот. Дальше по каждому листу:
ширина по внутренним точкам веток skan в px листа, шум ширины внутри линии (std, MAD по ветке),
смесь двух гауссиан с разными дисперсиями (средние, std, доли, граница равных вероятностей, BIC
для 1 и 2 классов) и порог минимальной ошибки Kittler-Illingworth.

Внутренние точки - дальше одной ширины штриха от концов ветки: у стыка и у окончания линии скелет
искажён (Hilaire, Tombre 2006), такие точки меряют стык, а не линию.

Запуск из notebooks/: uv run python width_stats_stand.py [имя файла ...]
Результат: data/output/width_stats/ - sheets.csv (по листу), <лист>_branches.csv, <лист>.png
(гистограмма с моделью, шум внутри линий) и <лист>_map.png - лист, где каждая ветка раскрашена по своей
медиане ширины: синий - тонкая, оранжевый - толстая, пурпурный - между границей смеси и порогом
Kittler-Illingworth, то есть неоднозначная на этом листе.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from skan import Skeleton
from sklearn.mixture import GaussianMixture

from draftslice.core.io import load_image
from draftslice.core.region_seg.pipeline import run

DATA = Path("../data")
OUT = DATA / "output" / "width_stats"
NB_CELLS = ["6283a6dd", "f01cff35", "8d9b839e", "adcbac36"]  # закрытие, OCR, апскейл, ширина


def _branches(d):
    """Ветки кропа: ширина по внутренним точкам в px листа, std, MAD; номер ветки на точках скелета.

    У ветки короче двух ширин штриха внутренних точек нет - её медиана по всем точкам, в статистику
    листа (n = 0) она не входит, только в раскраску.
    """
    sk = Skeleton(d.skel_map)
    rows, pts = [], []
    blab = np.zeros(d.skel_map.shape, np.int32)
    for i in range(sk.n_paths):
        p = sk.path_coordinates(i).astype(int)
        blab[p[:, 0], p[:, 1]] = i + 1
        w = d.skel_map[p[:, 0], p[:, 1]]
        k = int(np.ceil(np.median(w)))  # одна ширина штриха от каждого конца - зона искажения
        inner = w[k : len(w) - k] / d.SCALE
        if len(inner) < 3:
            rows.append((0, np.median(w) / d.SCALE, np.nan, np.nan))
            continue
        med = np.median(inner)
        rows.append((len(inner), med, inner.std(), np.median(np.abs(inner - med))))
        pts.append(inner)
    return pd.DataFrame(rows, columns=["n", "median", "std", "mad"]), (np.concatenate(pts) if pts else np.empty(0)), blab


def _paint(img, crops, data, parts, lo, hi):
    """Лист: пиксел линии - цвет класса ветки ближайшей точки скелета."""
    canvas = np.full_like(img, 255)
    canvas[(img.min(axis=2) < 128)] = (215, 215, 215)
    for c, d, (br, _, blab) in zip(crops, data, parts):
        med = np.r_[np.nan, br["median"].to_numpy()]
        up = med[blab.ravel()[d.near]].reshape(blab.shape)
        m = cv2.resize(up.astype(np.float32), (c.w, c.h), interpolation=cv2.INTER_NEAREST)
        box = canvas[c.y : c.y + c.h, c.x : c.x + c.w]
        box[c.mask & (m < lo)] = (42, 120, 214)
        box[c.mask & (m > hi)] = (235, 104, 52)
        box[c.mask & (m >= lo) & (m <= hi)] = (200, 0, 200)
    return canvas


def _kittler(x):
    """Порог минимальной ошибки Kittler-Illingworth по гистограмме x."""
    vals, cnt = np.unique(np.round(x, 2), return_counts=True)
    p = cnt / cnt.sum()
    best, t_best = np.inf, vals[0]
    for i in range(1, len(vals)):
        p1, p2 = p[:i].sum(), p[i:].sum()
        m1, m2 = (p[:i] * vals[:i]).sum() / p1, (p[i:] * vals[i:]).sum() / p2
        s1 = np.sqrt((p[:i] * (vals[:i] - m1) ** 2).sum() / p1)
        s2 = np.sqrt((p[i:] * (vals[i:] - m2) ** 2).sum() / p2)
        if s1 <= 0 or s2 <= 0:
            continue
        j = 1 + 2 * (p1 * np.log(s1) + p2 * np.log(s2)) - 2 * (p1 * np.log(p1) + p2 * np.log(p2))
        if j < best:
            best, t_best = j, (vals[i - 1] + vals[i]) / 2
    return t_best


def _gmm(x, q):
    """Смесь 1 и 2 гауссиан по x: параметры двух классов, граница равных вероятностей, BIC.

    Ширина квантована шагом q (px листа), поэтому у класса дисперсия не меньше q^2 / 12 - дисперсии
    ошибки квантования; иначе класс из одинаковых значений вырождается в пик нулевой ширины.
    """
    rng = np.random.default_rng(0)
    xs = rng.choice(x, min(len(x), 200_000), replace=False).reshape(-1, 1)
    g1 = GaussianMixture(1, reg_covar=q**2 / 12, random_state=0).fit(xs)
    g2 = GaussianMixture(2, reg_covar=q**2 / 12, random_state=0).fit(xs)
    order = np.argsort(g2.means_.ravel())
    mu = g2.means_.ravel()[order]
    sd = np.sqrt(g2.covariances_.ravel()[order])
    wt = g2.weights_[order]
    grid = np.linspace(mu[0], mu[1], 2000).reshape(-1, 1)
    post = g2.predict_proba(grid)[:, order[1]]
    bound = grid[np.argmin(np.abs(post - 0.5)), 0]
    return dict(mu1=mu[0], sd1=sd[0], w1=wt[0], mu2=mu[1], sd2=sd[1], w2=wt[1], gmm_thresh=bound,
                bic1=g1.bic(xs), bic2=g2.bic(xs)), g2, order


def _plot(name, x, br, st, g2, order, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    ax = axes[0]
    bins = np.arange(0, np.percentile(x, 99.5) + 0.25, 0.1)
    ax.hist(x, bins=bins, density=True, color="#b0b0b0")
    xx = np.linspace(0, bins[-1], 600)
    for j, color, label in ((order[0], "#2a78d6", "тонкие"), (order[1], "#eb6834", "толстые")):
        mu, sd, w = g2.means_[j, 0], np.sqrt(g2.covariances_[j, 0, 0]), g2.weights_[j]
        ax.plot(xx, w * np.exp(-((xx - mu) ** 2) / (2 * sd**2)) / (sd * np.sqrt(2 * np.pi)), color=color, lw=2,
                label=f"{label}: {mu:.2f}±{sd:.2f}, доля {w:.2f}")
    ax.axvline(st["gmm_thresh"], color="black", lw=1.5, label=f"граница смеси {st['gmm_thresh']:.2f}")
    ax.axvline(st["ki_thresh"], color="black", ls="--", lw=1.5, label=f"Kittler-Illingworth {st['ki_thresh']:.2f}")
    ax.set_xlabel("ширина по внутренним точкам скелета, px листа")
    ax.set_title(f"{name}: точек {len(x)}")
    ax.legend(fontsize=8)
    ax = axes[1]
    big = br[br.n >= 20]
    ax.scatter(big["median"], big["std"], s=np.clip(big.n / 10, 4, 60), alpha=0.5, color="#2a78d6")
    ax.axvline(st["gmm_thresh"], color="black", lw=1.5)
    ax.set_xlabel("медиана ширины ветки, px листа")
    ax.set_ylabel("std ширины внутри ветки, px листа")
    ax.set_title(f"шум внутри линии: медиана std {st['line_std']:.2f}, веток {len(big)} (≥20 точек)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(names):
    nb = json.load(open("exp_classifier_lines.ipynb"))
    src = {c["id"]: "".join(c["source"]) for c in nb["cells"]}
    ns = {"__name__": "stand"}
    exec("import cv2\nimport numpy as np\nfrom matplotlib import pyplot as plt\n", ns)
    exec(src["54830e2b"], ns)  # модели OCR - один раз
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted((DATA / "images").glob("*.jpg")) + sorted((DATA / "drawing").glob("*.png"))
    if names:
        files = [f for f in files if f.name in names]
    rows = []
    for f in files:
        img = load_image(str(f))
        ns.update(img=img, crops=run(img))
        for cid in NB_CELLS:
            exec(src[cid].replace("plt.show()", "plt.close('all')"), ns)
        parts = [_branches(d) for d in ns["data"]]
        br = pd.concat([b for b, _, _ in parts], ignore_index=True)
        br = br[br.n > 0]
        x = np.concatenate([p for _, p, _ in parts])
        # 2 * расстояние до фона округляется вверх до чётного числа рабочих пикселов: шаг измерения
        # 2 / SCALE px листа, по самому грубому кропу
        q = 2 / min(d.SCALE for d in ns["data"])
        st, g2, order = _gmm(x, q)
        st.update(sheet=f.name, folder=f.parent.name, crops=len(ns["crops"]),
                  scales="/".join(sorted({str(d.SCALE) for d in ns["data"]})),
                  points=len(x), ki_thresh=_kittler(x),
                  line_std=br[br.n >= 20]["std"].median(), line_mad=br[br.n >= 20]["mad"].median())
        st["separation"] = (st["mu2"] - st["mu1"]) / np.sqrt((st["sd1"] ** 2 + st["sd2"] ** 2) / 2)
        st["class_ratio"] = st["mu2"] / st["mu1"]
        rows.append(st)
        stem = f.stem.replace(" ", "_")
        br.to_csv(OUT / f"{stem}_branches.csv", index=False)
        _plot(f.name, x, br, st, g2, order, OUT / f"{stem}.png")
        lo, hi = sorted((st["gmm_thresh"], st["ki_thresh"]))
        canvas = _paint(img, ns["crops"], ns["data"], parts, lo, hi)
        cv2.imwrite(str(OUT / f"{stem}_map.png"), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        print(f"{f.name}: тонкие {st['mu1']:.2f}±{st['sd1']:.2f}, толстые {st['mu2']:.2f}±{st['sd2']:.2f}, "
              f"граница {st['gmm_thresh']:.2f}, KI {st['ki_thresh']:.2f}, разделимость {st['separation']:.2f}, "
              f"шум линии {st['line_std']:.2f}", flush=True)
        pd.DataFrame(rows).to_csv(OUT / "sheets.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1:])
