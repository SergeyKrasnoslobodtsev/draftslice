"""Разделение линий чертежа на толстые и тонкие по статистике самого листа, без ручных констант.

Шаги - векторные операции numpy / OpenCV по точкам скелета, без циклов Python по веткам и без
операций над всем кадром там, где хватает точек (листов будут миллионы):

1. Бинаризация, скелет, расстояние до фона.
2. Зоны искажения: точки скелета ближе одной местной ширины штриха к концу или стыку. Там скелет
   меряет стык или окончание, а не линию (Hilaire, Tombre 2006) - такие точки выбрасываются.
   Оставшиеся куски скелета - сегменты, связные компоненты за один вызов OpenCV.
3. Ширина сегмента по площади: пикселы краски, ближайшие к его точкам скелета, / его длина. На
   внутренних точках мера честная и дробная; у стыков пикселы перетекают между линиями, но эти точки
   уже выброшены. Узел скелета не режет линию там, где меняется толщина (наконечник - тонкая
   размерная, толстая линия - тонкая дуга без стыка), поэтому второй проход режет сегмент по
   середине между медианами классов, если в нём уверенно есть оба класса (см. measure).
4. Текст - шум: связная компонента краски, у которой больше половины пикселов внутри полигонов
   OCR (модуль ocr) или которая целиком лежит в полигоне, расширенном на полвысоты символа
   (рамка вокруг надписи), убирается до скелета. Буква, касающаяся линии, остаётся в её
   компоненте. Порог «больше половины» по расширенным полигонам снимал целиком маленький вид вместе
   с его размерами (СОК-АК.4-4).
   Прежний признак - компонента меньше 20 толщин штриха - гасил и отверстия: кружок с центровым
   крестом по размеру и форме не отличить от буквы «О».
5. Порог листа - по ширинам сегментов с весом по длине: Otsu (без предположения о форме классов)
   как начало, затем 2-медианы - порог посередине между взвешенными медианами классов. Не смещается
   к меньшему классу и не раздувается наконечниками стрелок. Смесь гауссиан не годится: у тонких
   пик на 1 px и «плечо» наклонных линий, гауссиана кладёт плечо к толстым.
6. Зона искажения (связный кусок выброшенного скелета) толстая, если касается двух и более толстых
   сегментов - стык на толстой линии, - или одного-единственного сегмента, и он толстый, - конец
   толстой линии. Иначе на контуре у примыкания штриховки или выносной оставался разрыв, а у конца
   линии (разрыв под текст, острие стрелки у контура) - тонкий хвостик. Обрубок примыкающей
   тонкой линии, торчащий за край толстой, из толстой зоны вырезается: точка тонкой ветки (волна
   класса по скелету от концов сегментов) уже толстой линии, перебег осевой за контур - тупиковый
   отросток; пиксел остаётся, если лежит в теле толстой линии (круг вписанного радиуса ближайшей
   толстой точки скелета). На листах, где толстые и тонкие различаются меньше чем на 1 px (линии
   1 и 2 px), зона толстая целиком.
   Зона, которую соседние сегменты толстой не делают, классифицируется по своим точкам: местная
   ширина по площади (окно +-S вдоль зоны) выше порога своего вида. Маленькое отверстие,
   пересечённое осевыми, - целиком зона без сегментов с тонкими соседями; так же широкая часть
   наконечника у стыка с выносной (лист 9).
7. Порог вида: на одном листе перо и нажим от вида к виду разные (особенно от руки). Порог вида -
   уточнение порога листа: Otsu по сегментам вида (кроп region_seg) с разрезом только между
   медианами тонких и толстых листа. Принимается, только если тонкий и толстый классы вида лежат по
   разные стороны порога листа - то есть на виде есть оба класса листа. Вид из одних толстых или
   одних тонких линий так не разрезается пополам - у него остаётся порог листа.
8. Пиксел краски получает класс ближайшей точки скелета. Наконечники стрелок не снимаются: это
   самое широкое место линии, и признак формы путает их со стыком двух толстых линий.
9. Рамка и штамп - шум. Рамка - длинные прямые у края листа на SHEET_COV его стороны (допуск на
   перекос скана - по наклону самой длинной прямой). Штамп и графы - области, замкнутые рамкой и
   связанными с ней длинными толстыми прямыми; поле чертежа - самая большая из них, толстые линии
   вне поля убираются. Длинная прямая - длиннее CHAR_S толщин толстой, то есть не штрих буквы.

10. Разрешение. Расстояние до фона дискретно с шагом 1 px: если медианы тонких и толстых ближе
    1 px (линии 1 и 2 px), в стыках их не различить, и лист считается заново в ×2. Пороги на
    листе с рамкой оцениваются только по полю чертежа: кромка скана и мусор за рамкой дают лишнюю
    моду тонких.
11. Цепочки (chains): куски, продолжающие друг друга через узел (поворот < 45°), - одна линия;
    кольцо, пересечённое осевыми, - замкнутая цепочка; примкнувшая под углом линия в узле
    кончается. Кусок, чья ширина от порога ближе погрешности (w / L + разброс ширины вдоль линий
    листа), берёт класс соседей по цепочке с совместимой шириной. Уверенный кусок класс держит:
    толстая, перешедшая в тонкую, - конец толстой и начало тонкой.

Ширина по площади делится на длину, которую покрывают точки (связи + по половине шага на концах):
деление на одни связи завышало ширину коротких сегментов вдвое, и концы тонких линий и кусочки
между близкими пересечениями выходили толстыми.

Стенд: uv run python thick_lines.py [имя файла ...] -> data/output/thick_lines/.
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage.morphology import skeletonize

from draftslice.core.io import load_image
from draftslice.core.preprocess import preprocess
from draftslice.core.region_seg.pipeline import run as region_seg

DATA = Path("../data")
OUT = DATA / "output" / "thick_lines"
OCR_CACHE = DATA / "output" / "ocr_cache"
SQRT2 = np.sqrt(2)
SHEET_COV = 0.85  # доля стороны листа: прямая длиннее - линия рамки (как sheet_cov в region_seg)
# толщин толстой линии S: прямая длиннее - не штрих буквы. Высота символа h <= 14 d, штрих d <= S/2
# (ГОСТ 2.304, 2.303), значит h <= 7 S.
CHAR_S = 7
TILE = 256  # px рабочей сетки: сторона тайла скелета, на результат не влияет - только на скорость


def _nbr(a, ys, xs, dy, dx):
    """Значения a в соседях (ys + dy, xs + dx) точек, за краем - 0. Только по точкам, не по всему листу."""
    y, x = ys + dy, xs + dx
    ok = (y >= 0) & (y < a.shape[0]) & (x >= 0) & (x < a.shape[1])
    out = np.zeros(len(ys), a.dtype)
    out[ok] = a[y[ok], x[ok]]
    return out


def skeleton(ink, dist):
    """Скелет Zhang-Suen (skimage) по тайлам TILE с полем перекрытия - точь-в-точь как по всему листу.

    Решение по точке зависит от её окрестности 3x3, поэтому ложный край окна уходит внутрь не больше
    чем на 1 px за полупроход, а проходов не больше наибольшего расстояния до фона: поле 2 * max dist
    + 2 px отсекает влияние края. Выигрыш: skimage гоняет проходы по всему кадру, пока меняется хоть
    одна точка - число проходов задаёт самая толстая клякса листа; тайл останавливается сам по себе, а
    пустые тайлы пропускаются. На листах 3-50 Мп в 3-5 раз быстрее.
    """
    m = 2 * int(np.ceil(dist.max())) + 2
    h, w = ink.shape
    ii = cv2.integral(ink.astype(np.uint8))
    skel = np.zeros_like(ink)
    for y0 in range(0, h, TILE):
        for x0 in range(0, w, TILE):
            y1, x1 = min(y0 + TILE, h), min(x0 + TILE, w)
            if ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0] == 0:
                continue
            a0, b0, a1, b1 = max(0, y0 - m), max(0, x0 - m), min(h, y1 + m), min(w, x1 + m)
            skel[y0:y1, x0:x1] = skeletonize(ink[a0:a1, b0:b1])[y0 - a0 : y1 - a0, x0 - b0 : x1 - b0]
    return skel


def _local_width(seg, inner, near, s):
    """Ширина по площади вокруг точек скелета: пикселы краски / длина, в окне +-s px вдоль своего сегмента.

    Окно - s шагов сглаживания по 8-соседям той же метки: квадратное окно захватывало соседнюю линию,
    идущую вплотную к дуге контура, и кусок дуги мерился тонким.
    """
    ys, xs = np.nonzero(inner)
    flat = ys * inner.shape[1] + xs  # по возрастанию - порядок np.nonzero
    sid = seg[ys, xs]
    area = np.bincount(near, minlength=inner.size)[flat].astype(np.float64)
    # Длина на точку - половина связей с соседями (прямые 1, диагональные sqrt(2), диагональ с
    # обходом через прямого соседа не считается).
    length = np.zeros(len(ys))
    deg = np.zeros(len(ys))
    nbs = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if not (dy or dx):
                continue
            link = _nbr(seg, ys, xs, dy, dx) == sid
            if dy and dx:
                link_d = link & ~_nbr(inner, ys, xs, 0, dx) & ~_nbr(inner, ys, xs, dy, 0)
                length += link_d * SQRT2 / 2
                deg += link_d
            else:
                length += link * 0.5
                deg += link
            j = np.searchsorted(flat, flat + dy * inner.shape[1] + dx).clip(0, len(flat) - 1)
            nbs.append(np.where(link, j, -1))
    # Конец линии - половина шага сверх связей: без неё концевая точка с полной площадью мерилась
    # вдвое шире (та же поправка, что у сегмента).
    length += 0.5 * np.clip(2 - deg, 0, 2)
    for _ in range(int(np.ceil(s))):
        a, ln = area.copy(), length.copy()
        for j in nbs:
            m = j >= 0
            a[m] += area[j[m]]
            ln[m] += length[j[m]]
        area, length = a, ln
    out = np.zeros(inner.shape, np.float32)
    out[ys, xs] = area / np.maximum(length, 1e-6)
    return out


def _seg_stats(seg, n_seg, inner, dist, near, scale):
    """Длина, ширина по площади, наибольшая ширина, сужение и опорная точка сегментов (в px листа)."""
    area = np.bincount(seg.ravel()[near], minlength=n_seg)  # метка 0 - пикселы зон искажения
    # Длина сегмента: связи соседних точек одного сегмента - прямые по 1, диагональные по sqrt(2);
    # диагональ, у которой есть обход через прямого соседа, не считается (иначе угол посчитан дважды).
    ys, xs = np.nonzero(seg)
    sid = seg[ys, xs]
    length = np.zeros(n_seg)
    for dy, dx, step in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, SQRT2), (1, -1, SQRT2)):
        link = _nbr(seg, ys, xs, dy, dx) == sid
        if step > 1:
            link &= ~_nbr(inner, ys, xs, 0, dx) & ~_nbr(inner, ys, xs, dy, 0)
        length += np.bincount(sid[link], minlength=n_seg) * step
    n_pts = np.bincount(sid, minlength=n_seg)
    # Наибольшая ширина по точкам сегмента: 2 * расстояние - 1 (расстояние меряется между центрами
    # пикселов, у линии в 1 px оно 1, у линии нечётной ширины w центр на (w + 1) / 2 от фона).
    w = 2 * dist[ys, xs] - 1
    wmax = np.zeros(n_seg)
    np.maximum.at(wmax, sid, w)
    # Ширина = площадь / длина, которую покрывают точки: у сегмента из n точек на прямой связей n - 1,
    # а площадь собрана со всех n точек - каждая несёт полный шаг, концы по половине шага сверх
    # связей. Деление на одни связи завышало ширину коротких сегментов: у 2 точек тонкой линии в
    # 1 px выходило 2 px - толстые кусочки на концах тонких линий и между близкими пересечениями.
    segs = pd.DataFrame(
        {"length": length / scale, "width": area / (length + 1) / scale, "n": n_pts, "wmax": wmax / scale}
    )
    # Опорная точка сегмента - его первая точка в порядке обхода; по ней сегмент относится к виду.
    _, first = np.unique(sid, return_index=True)
    segs = segs.iloc[1:]  # метка 0 - не сегмент
    segs["y"], segs["x"] = ys[first], xs[first]
    return segs


def measure(img, polys=(), scale=1.0):
    """Сегменты линий: ширина по площади и длина в px листа; метки для раскраски.

    polys - четырёхугольники текста OCR в px листа; компоненты текста убираются из краски.
    """
    pp = preprocess(img, target=None)
    gray = pp.gray
    if scale != 1:
        gray = cv2.resize(gray, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] > 0
    else:
        ink = pp.binary > 0
    text = np.zeros_like(ink)
    if len(polys):
        # tight - полигоны OCR как есть (обводят символы вплотную), wide - с полем в полвысоты символа:
        # на столько от надписи отстоит рамка вокруг неё (размер в рамке, допуск формы). Заливка по
        # одному: fillPoly по набору заливает по чётности, и перекрытие соседних строк выпадает.
        tight = np.zeros(ink.shape, np.uint8)
        wide = np.zeros(ink.shape, np.uint8)
        for p in polys:
            q = np.asarray(p, np.float32) * scale
            cv2.fillConvexPoly(tight, np.round(q).astype(np.int32), 1)
            (cx, cy), (w, h), a = cv2.minAreaRect(q)
            m = min(w, h) / 2
            box = cv2.boxPoints(((cx, cy), (w + 2 * m, h + 2 * m), a))
            cv2.fillConvexPoly(wide, np.round(box).astype(np.int32), 1)
        n_cc, cc = cv2.connectedComponents(ink.astype(np.uint8), connectivity=8)
        area = np.bincount(cc[ink], minlength=n_cc)
        # Текст - больше половины компоненты в полигонах, или она целиком в поле надписи (рамка
        # вокруг текста). Вид, связанный со своими размерами, целиком в поле не лежит никогда.
        is_text = 2 * np.bincount(cc[ink & (tight > 0)], minlength=n_cc) > area
        is_text |= np.bincount(cc[ink & (wide > 0)], minlength=n_cc) == area
        is_text[0] = False
        text = is_text[cc]
        ink &= ~text
    dist = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    skel = skeleton(ink, dist)

    # Концы и стыки по числу соседей на скелете; зона искажения - ближе местной ширины штриха.
    nb = cv2.filter2D(skel.astype(np.uint8), -1, np.ones((3, 3), np.float32)) - 1
    knots = skel & ((nb == 1) | (nb >= 3))
    d_knot = cv2.distanceTransform((~knots).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    inner = skel & (d_knot > 2 * dist)
    # Площадь: каждый пиксел краски - ближайшей точке всего скелета.
    _, lbl = cv2.distanceTransformWithLabels((~skel).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    lut = np.zeros(lbl.max() + 1, np.int32)
    lut[lbl[skel]] = np.flatnonzero(skel)
    near = lut[lbl[ink]]  # для каждого пиксела краски - плоский индекс ближайшей точки скелета

    # Узлы скелета не режут линию там, где меняется толщина: наконечник переходит в тонкую размерную
    # без стыка, и короткая размерная со своими стрелками - один сегмент средней ширины (лист 5).
    # Второй проход режет сегмент по середине между медианами тонкого и толстого классов первого
    # прохода, но только если в нём уверенно есть оба класса: точки не дальше четверти расстояния
    # между медианами от медианы тонких и от медианы толстых. Так отделяются наконечник от тонкого
    # стержня и тонкая дуга, переходящая в толстую линию без стыка (лист 21), а линия от руки, чья
    # толщина плавает около середины (лист 14), не режется на куски разного цвета. Местная ширина -
    # та же мера по площади, что у сегмента, в окне +-S вдоль сегмента (S - толщина толстой).
    # 2 * dist - 1 не годится: у диагонали толщиной 3 px и 1 px она одинакова.
    n_seg, seg = cv2.connectedComponents(inner.astype(np.uint8), connectivity=8)
    segs = _seg_stats(seg, n_seg, inner, dist, near, scale)
    ok = segs.length > 0
    _, thin, thick_w = _split(segs.width[ok].to_numpy(), segs.length[ok].to_numpy())
    if (thick_w - thin) * scale > 1:
        lw = _local_width(seg, inner, near, thick_w * scale) / scale
        q = (thick_w - thin) / 4
        has_n = np.bincount(seg[inner & (lw <= thin + q)], minlength=n_seg) > 0
        has_w = np.bincount(seg[inner & (lw >= thick_w - q)], minlength=n_seg) > 0
        wide = inner & (has_n & has_w)[seg] & (lw > (thin + thick_w) / 2)
        n_nar, seg = cv2.connectedComponents((inner & ~wide).astype(np.uint8), connectivity=8)
        n_wide, seg_w = cv2.connectedComponents(wide.astype(np.uint8), connectivity=8)
        seg[seg_w > 0] = seg_w[seg_w > 0] + n_nar - 1
        n_seg = n_nar + n_wide - 1
        segs = _seg_stats(seg, n_seg, inner, dist, near, scale)
    return segs, dict(ink=ink, text=text, dist=dist, skel=skel, inner=inner, seg=seg, near=near)


def view_map(img):
    """Номер вида (кропа region_seg) для каждого пиксела листа, 0 - вне видов."""
    views = np.zeros(img.shape[:2], np.int32)
    for k, c in enumerate(region_seg(img), 1):
        views[c.y : c.y + c.h, c.x : c.x + c.w][c.mask] = k
    return views


def otsu_weighted(x, w, lo=-np.inf, hi=np.inf):
    """Порог Otsu по x с весами w: максимум межклассовой дисперсии, разрезы только внутри [lo, hi]."""
    o = np.argsort(x)
    x, w = x[o], w[o]
    cw, cwx = np.cumsum(w), np.cumsum(w * x)
    tw, twx = cw[-1], cwx[-1]
    w0, w1 = cw[:-1], tw - cw[:-1]
    m0, m1 = cwx[:-1] / w0, (twx - cwx[:-1]) / w1
    between = w0 * w1 * (m0 - m1) ** 2
    cut = (x[:-1] + x[1:]) / 2
    between[(x[1:] == x[:-1]) | (cut < lo) | (cut > hi)] = -1  # разрез между разными значениями, в [lo, hi]
    i = int(np.argmax(between))
    return (x[i] + x[i + 1]) / 2


def _wmedian(x, w):
    """Взвешенная медиана."""
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return x[o][np.searchsorted(c, c[-1] / 2)]


def _split(x, w, lo=-np.inf, hi=np.inf):
    """Порог и центры (взвешенные медианы) двух классов: Otsu как начало, затем 2-медианы.

    Порог - середина между медианами классов, пересчёт до сходимости; разрез внутри [lo, hi].
    Otsu при неравных классах сдвигает разрез к меньшему, а среднее толстого класса раздувают
    наконечники стрелок (ширина ~5 при контуре ~3.2, по длине сравнимы с контуром, изрезанным
    штриховкой). Фаска 2.4 уходила в тонкие (лист 5, вид Е). Медиана к таким хвостам нечувствительна.
    """
    b = otsu_weighted(x, w, lo, hi)
    for _ in range(100):
        nb = float(np.clip((_wmedian(x[x <= b], w[x <= b]) + _wmedian(x[x > b], w[x > b])) / 2, lo, hi))
        if nb == b or not ((x <= nb).any() and (x > nb).any()):
            break
        b = nb
    return b, _wmedian(x[x <= b], w[x <= b]), _wmedian(x[x > b], w[x > b])


def classify(segs, views=None, scale=1.0, use=None):
    """Сегмент толстый, если ширина выше порога своего вида (или листа).

    use - сегменты, по которым оцениваются пороги (по умолчанию все): на листе с рамкой - только
    поле чертежа, без кромки скана и штампа.
    """
    ok = segs.length > 0
    if use is not None:
        ok &= use
    bound, thin, thick_w = _split(segs.width[ok].to_numpy(), segs.length[ok].to_numpy())
    thr = np.full(len(segs), bound)
    view_thr = {}
    if views is not None:
        vy = (segs.y / scale).astype(int).clip(0, views.shape[0] - 1)
        v = views[vy, (segs.x / scale).astype(int).clip(0, views.shape[1] - 1)]
        for k in np.unique(v[v > 0]):
            m = ok.to_numpy() & (v == k)
            if m.sum() < 2 or np.unique(segs.width[m]).size < 2:
                continue
            xv = segs.width[m].to_numpy()
            if not ((xv > thin) & (xv < thick_w)).any():
                continue
            b, lo, hi = _split(xv, segs.length[m].to_numpy(), thin, thick_w)
            if lo < bound < hi:
                thr[v == k] = b
                view_thr[int(k)] = b
    thick = (segs.width > thr).to_numpy()
    return thick, dict(bound=bound, thin=thin, thick=thick_w, views_own=len(view_thr), view_thr=view_thr, seg_thr=thr)


def chains(geo, segs, thick, st, scale):
    """Класс по цепочкам: неуверенно измеренный кусок линии берёт класс её продолжений.

    Цепочка - куски скелета, продолжающие друг друга через узел: поворот меньше 45° (середина
    между прямой и перпендикуляром). В узле жадно связываются самые соосные пары концов:
    толстая линия проходит сквозь узел, примкнувшая под углом тонкая в нём кончается; кольцо,
    пересечённое осевыми, - замкнутая цепочка. Направление конца - по точкам куска на длине окна
    K = max(2 S, 1 / tg 10°) px от узла (S - толщина толстой; 2 S - длина зоны искажения, 1 / tg 10°
    - окно, где шаг сетки даёт ошибку направления не больше 10°); у куска короче окна - по всей его
    длине: штриховка режет контур на куски по 4 px. Кусок короче 2 px направления не имеет.

    Погрешность ширины куска по площади - w / L px (граница куска на каждом конце плавает на
    полпиксела длины, площадь - на ширину) плюс разброс ширины вдоль линии, измеренный на листе.
    Кусок, чья ширина от порога ближе 2 погрешностей, неуверенный: берёт класс соседей по цепочке
    с совместимой шириной, если все классифицированные соседи согласны
    (прогон до сходимости - цепочка неуверенных кусков). Уверенный кусок класс не меняет: толстая,
    перешедшая в тонкую, - конец толстой и начало тонкой.
    """
    inner, skel, seg = geo["inner"], geo["skel"], geo["seg"]
    n_seg = seg.max() + 1
    k = int(np.ceil(max(2 * st["thick"] * scale, 1 / np.tan(np.radians(10)))))
    length = np.r_[0, segs.length.to_numpy() * scale]
    long_ = length >= 2
    long_[0] = False
    lmask = inner & long_[seg]
    if not lmask.any():
        return thick
    # Узлы цепочек - связные куски скелета без длинных сегментов (зоны стыков вместе с короткими).
    zmask = skel & ~lmask
    _, zone = cv2.connectedComponents(zmask.astype(np.uint8), connectivity=8)
    ly, lx = np.nonzero(lmask)
    lflat = ly * seg.shape[1] + lx
    lsid = seg[ly, lx]
    zy, zx = np.nonzero(zmask)
    zid = zone[zy, zx]
    # Точка касания конца сегмента с узлом.
    ends = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if not (dy or dx):
                continue
            ny, nx = zy + dy, zx + dx
            ok = (ny >= 0) & (ny < seg.shape[0]) & (nx >= 0) & (nx < seg.shape[1])
            ok[ok] = lmask[ny[ok], nx[ok]]
            i = np.searchsorted(lflat, ny[ok] * seg.shape[1] + nx[ok])
            ends.append(np.stack([zid[ok], lsid[i], i], 1))
    ends = np.concatenate(ends)
    if not len(ends):
        return thick
    _, first = np.unique(ends[:, :2], axis=0, return_index=True)
    ends = ends[first]
    e_zone, e_seg, e_pt = ends[:, 0], ends[:, 1], ends[:, 2]
    # Обход сегмента от точки касания на k шагов, от каждого конца независимо: последняя
    # достигнутая точка задаёт направление.
    seen = set(zip(e_pt.tolist(), range(len(ends)), strict=True))
    far = e_pt.copy()
    front, f_own = e_pt, np.arange(len(ends))
    for _ in range(k):
        nxt_pt, nxt_own = [], []
        fy, fx = ly[front], lx[front]
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if not (dy or dx):
                    continue
                ny, nx = fy + dy, fx + dx
                ok = (ny >= 0) & (ny < seg.shape[0]) & (nx >= 0) & (nx < seg.shape[1])
                ok[ok] = lmask[ny[ok], nx[ok]]
                i = np.searchsorted(lflat, ny[ok] * seg.shape[1] + nx[ok])
                o = f_own[ok]
                same_seg = lsid[i] == e_seg[o]
                nxt_pt.append(i[same_seg])
                nxt_own.append(o[same_seg])
        nxt_pt, nxt_own = np.concatenate(nxt_pt), np.concatenate(nxt_own)
        key = nxt_pt.astype(np.int64) * len(ends) + nxt_own
        key, u = np.unique(key, return_index=True)
        new = np.array([(p, o) not in seen for p, o in zip(nxt_pt[u].tolist(), nxt_own[u].tolist(), strict=True)], bool)
        if not new.any():
            break
        nxt_pt, nxt_own = nxt_pt[u][new], nxt_own[u][new]
        seen.update(zip(nxt_pt.tolist(), nxt_own.tolist(), strict=True))
        far[nxt_own] = nxt_pt
        front, f_own = nxt_pt, nxt_own
    d = np.stack([ly[e_pt] - ly[far], lx[e_pt] - lx[far]], 1).astype(float)  # внутрь узла
    norm = np.hypot(d[:, 0], d[:, 1])
    ok = norm > 0
    d[ok] /= norm[ok, None]
    # Пары концов в одном узле: продолжение, если -d_q близко к d_p.
    e = pd.DataFrame({"z": e_zone, "i": np.arange(len(ends))})[ok]
    c = e.merge(e, on="z")
    c = c[c.i_x < c.i_y]
    c = c[e_seg[c.i_x.to_numpy()] != e_seg[c.i_y.to_numpy()]]
    cos = -(d[c.i_x.to_numpy()] * d[c.i_y.to_numpy()]).sum(1)
    c = c.assign(cos=cos)
    c = c[c.cos > np.cos(np.radians(45))].sort_values("cos", ascending=False)
    used = np.zeros(len(ends), bool)
    edges = []
    for a, b in zip(c.i_x.to_numpy(), c.i_y.to_numpy(), strict=True):
        if not used[a] and not used[b]:
            used[a] = used[b] = True
            edges.append((e_seg[a], e_seg[b]))
    # Куски, соприкасающиеся без узла, - разрез второго прохода по ширине внутри одной линии:
    # это продолжение, связь без угла.
    iy, ix = np.nonzero(inner)
    isid = seg[iy, ix]
    for dy, dx in ((0, 1), (1, -1), (1, 0), (1, 1)):
        nb = _nbr(seg, iy, ix, dy, dx)
        m = (nb > 0) & (nb != isid)
        edges.extend(np.unique(np.stack([isid[m], nb[m]], 1), axis=0).tolist())
    if not edges:
        return thick
    edges = np.unique(np.sort(np.array(edges), axis=1), axis=0)
    geo["chain_pairs"] = edges  # соосные пары сегментов - для карты цепочек стенда
    # Линия - компонента связности по соосным связям с совместимой шириной: разница ширин не больше
    # 2 sqrt(sigma_a^2 + sigma_b^2) (95 %). Иначе соосный стык - переход толстой в тонкую
    # (выносная, продолжающая контур; размерная после наконечника). Ширина линии - среднее кусков
    # с весом 1 / sigma^2, её класс - против порога, и он у всех кусков линии.
    w = np.r_[0, segs.width.to_numpy()]
    thr = np.r_[np.inf, st["seg_thr"]]
    # Разброс ширины вдоль линии на этом листе (скан, нажим пера): std местной ширины (окно +-S)
    # по длинным линиям (длиннее 4 S - хотя бы два независимых окна), медиана по линиям. По куску
    # длины L он усредняется как sigma_line * sqrt(S / L).
    s_w = st["thick"] * scale
    lw = _local_width(seg, inner, geo["near"], s_w)[inner] / scale
    sid = seg[inner]
    n_p = np.maximum(np.bincount(sid, minlength=n_seg), 1)
    m1 = np.bincount(sid, weights=lw, minlength=n_seg) / n_p
    m2 = np.bincount(sid, weights=lw**2, minlength=n_seg) / n_p
    line_sd = np.sqrt(np.maximum(m2 - m1**2, 0))
    big = length >= 4 * s_w
    sd_line = float(np.median(line_sd[big])) if big.any() else 0.0
    sigma = w / np.maximum(length / scale, 1e-6) + sd_line * np.sqrt(s_w / np.maximum(length, s_w))
    same = np.abs(w[edges[:, 0]] - w[edges[:, 1]]) <= 2 * np.hypot(sigma[edges[:, 0]], sigma[edges[:, 1]])
    edges = edges[same]
    edges = np.concatenate([edges, edges[:, ::-1]])
    label = np.r_[False, thick].copy()
    done = np.abs(w - thr) >= 2 * sigma
    done[0] = True
    while True:
        src, dst = edges[:, 0], edges[:, 1]
        m = done[src] & ~done[dst]
        if not m.any():
            break
        n_t = np.bincount(dst[m], weights=label[src[m]], minlength=n_seg)
        n_a = np.bincount(dst[m], minlength=n_seg)
        agree = (n_a > 0) & ((n_t == 0) | (n_t == n_a)) & ~done
        if not agree.any():
            break
        label[agree] = n_t[agree] > 0
        done |= agree
    return label[1:]


def keep_mask(img_shape, geo, thick, scale, segs, st, views=None):
    """Маска толстых линий листа: пиксел краски получает класс ближайшей точки скелета.

    st - статистика classify (медианы классов, порог листа и видов в px листа), views - номер вида
    для каждого пиксела листа.
    """
    thin_w_sheet = st["thin"]
    ink, skel, inner, seg = geo["ink"], geo["skel"], geo["inner"], geo["seg"]
    # Смежность зона искажения - сегмент по 8-соседству.
    zmask = skel & ~inner
    _, zone = cv2.connectedComponents(zmask.astype(np.uint8), connectivity=8)
    zy, zx = np.nonzero(zmask)
    zid = zone[zy, zx]
    pairs = np.concatenate([np.stack([zid, _nbr(seg, zy, zx, dy, dx)], 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
    pairs = np.unique(pairs[pairs[:, 1] > 0], axis=0)
    zl, sl = pairs[:, 0], pairs[:, 1]
    n_zone = zone.max() + 1
    is_thick = np.r_[False, thick]
    # Зона толстая: стык двух и более толстых сегментов или конец толстой линии (единственный сегмент).
    n_thick = np.bincount(zl, weights=is_thick[sl], minlength=n_zone)
    n_all = np.bincount(zl, minlength=n_zone)
    hit = (n_thick >= 2) | ((n_all == 1) & (n_thick == 1))
    hit[0] = False
    # Хвостик - часть тонкой линии внутри толстой зоны, торчащая за край толстой. Класс заходит в
    # зону от концов сегментов волной по скелету; точка - хвостик, если до неё первой дошла тонкая
    # ветка (при равенстве - толстая) и ширина в ней 2 * dist - 1 не больше
    # наибольшей у этой тонкой ветки. Одной волны мало: на углу она захватывает кусок пути толстой
    # линии, но там ширина толстая. Одной ширины мало: у диагонали толщиной 3 px и 1 px она
    # одинакова 1.83. Режем, только если толстые шире тонких больше шага квантования 1 px.
    # Тонкая - медиана тонкого класса листа: ширина по площади у сегмента в 1-2 точки ненадёжна.
    width = np.r_[np.inf, segs.width.to_numpy() * scale]
    thick_min = np.full(n_zone, np.inf)
    np.minimum.at(thick_min, zl[is_thick[sl]], width[sl[is_thick[sl]]])
    sep = hit & (thick_min - 1 > thin_w_sheet * scale)
    has_thin = np.bincount(zl, weights=~is_thick[sl], minlength=n_zone) > 0
    cut = sep & has_thin
    wmax = np.r_[0, segs.wmax.to_numpy() * scale]
    thin_w = np.zeros(n_zone)
    np.maximum.at(thin_w, zl[~is_thick[sl]], wmax[sl[~is_thick[sl]]])
    lab = np.zeros(skel.shape, np.int8)  # 1 - толстая, 2 - тонкая
    iy, ix = np.nonzero(inner)
    lab[iy, ix] = np.where(is_thick[seg[iy, ix]], 1, 2)
    todo = cut[zid]
    while todo.any():
        i = np.flatnonzero(todo)
        nb = np.stack([_nbr(lab, zy[i], zx[i], dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
        new = np.where((nb == 1).any(axis=0), 1, np.where((nb == 2).any(axis=0), 2, 0)).astype(np.int8)
        if not new.any():
            break
        lab[zy[i], zx[i]] = new
        todo[i[new > 0]] = False
    stub_pt = np.zeros(skel.size, bool)
    dist = geo["dist"].ravel()
    zf = np.ravel_multi_index((zy, zx), skel.shape)
    # Уже толстой линии - меньше её ширины на шаг квантования 1 px: иначе в частой штриховке, где
    # волна штрихов занимает весь контур, а штрих от руки почти как контур, контур уходил в хвостики.
    narrow = sep[zid] & (2 * dist[zf] - 1 < thick_min[zid] - 1)
    stub_pt[zf] = cut[zid] & (lab[zy, zx] == 2) & (2 * dist[zf] - 1 <= thin_w[zid]) & narrow
    # Перебег тонкой линии за контур (2-3 px) своего сегмента не имеет и волну получает от толстого
    # центра стыка - это тупиковый отросток скелета. Обрезаем узкие точки с конца; путь толстой
    # линии и кольцо тупиков не имеют.
    alive = skel.astype(np.uint8)
    while True:
        i = np.flatnonzero(narrow & (alive[zy, zx] > 0))
        n_nb = sum(_nbr(alive, zy[i], zx[i], dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx)
        tip = i[n_nb <= 1]
        if not len(tip):
            break
        alive[zy[tip], zx[tip]] = 0
        stub_pt[zf[tip]] = True
    near_all = geo["near"]
    # Класс точек скелета: толстый сегмент или толстая зона без хвостика.
    sk = np.flatnonzero(skel.ravel())
    pt = np.zeros(skel.size, bool)
    pt[sk] = is_thick[seg.ravel()[sk]] | (hit[zone.ravel()[sk]] & ~stub_pt[sk])
    # Зона, которую соседние сегменты толстой не делают, классифицируется по своим точкам: местная
    # ширина (по площади, окно +-S вдоль зоны) выше порога своего вида. Маленькое отверстие,
    # пересечённое осевыми, - целиком зона стыка без единого сегмента, и его соседи - тонкие осевые;
    # широкая часть наконечника у стыка с выносной - тоже зона тонких соседей (лист 9).
    zp = zmask.copy()
    # Концевая зона (единственный сегмент) берёт класс сегмента: в её местную ширину попадает
    # скруглённый край штриха - у линии в 10 px десятки пикселов на полшага скелета, и концы тонких
    # линий мерились толстыми.
    zp[zmask] = ~hit[zone[zmask]] & (n_all[zone[zmask]] != 1)
    if zp.any():
        lw = _local_width(zone, zmask, near_all, st["thick"] * scale)
        thr = np.full(skel.shape, st["bound"] * scale, np.float32)
        if views is not None and st["view_thr"]:
            vmap = (
                views
                if scale == 1
                else cv2.resize(views, (skel.shape[1], skel.shape[0]), interpolation=cv2.INTER_NEAREST)
            )
            for k, b in st["view_thr"].items():
                thr[vmap == k] = b * scale
        pt |= (zp & (lw > thr)).ravel()
    near = geo["near"]
    keep = np.zeros_like(ink)
    keep[ink] = pt[near]
    # Тело толстой линии - объединение вписанных кругов её точек скелета (восстановление по
    # срединной оси, Blum). Пиксел краски, лежащий в круге ближайшей толстой точки, - толстый, даже
    # если ближайшая точка скелета у него тонкая: на пересечении толстой линии с тонкой (осевая
    # через кольцо отверстия, хвостик у контура) иначе оставались синие пятна и выщербины внутри
    # толстой линии. Круг - радиус dist - 0.5: вписанный круг целиком в краске этой линии.
    q = np.flatnonzero(ink.ravel() & ~keep.ravel())
    if len(q) and pt.any():
        _, lbl = cv2.distanceTransformWithLabels(
            (~pt.reshape(skel.shape)).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
        )
        lut = np.zeros(lbl.max() + 1, np.int64)
        lut[lbl.ravel()[pt]] = np.flatnonzero(pt)
        c = lut[lbl.ravel()[q]]
        qy, qx = np.divmod(q, skel.shape[1])
        cy, cx = np.divmod(c, skel.shape[1])
        body = np.hypot(qy - cy, qx - cx) <= dist[c] - 0.5
        keep.ravel()[q[body]] = True
    if scale != 1:
        keep = cv2.resize(keep.astype(np.uint8), (img_shape[1], img_shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    return keep


def text_polys(img, stem):
    """Полигоны текста OCR листа (px листа), с кэшем в data/output/ocr_cache - Paddle долгий на стенде."""
    path = OCR_CACHE / f"{stem}.npy"
    if path.exists():
        return np.load(path)
    global _ENGINE
    if _ENGINE is None:
        from draftslice.core.ocr.engine import OCREngine, PaddleOCRParameters

        _ENGINE = OCREngine(PaddleOCRParameters())
    from draftslice.core.ocr.pipeline import run as ocr

    polys = np.array([r.poly for r in ocr(img, _ENGINE)], np.float32).reshape(-1, 4, 2)
    OCR_CACHE.mkdir(parents=True, exist_ok=True)
    np.save(path, polys)
    return polys


_ENGINE = None


def _drift(lines, axis):
    """Наклон самой длинной прямой маски: поперечный размер бокса / продольный (axis 1 - горизонтали)."""
    n, _, st, _ = cv2.connectedComponentsWithStats(lines, connectivity=8)
    if n < 2:
        return 0.0
    along, across = (st[1:, 2], st[1:, 3]) if axis == 1 else (st[1:, 3], st[1:, 2])
    i = along.argmax()
    return across[i] / along[i]


def _long_lines(mask, s):
    """Горизонтальные и вертикальные прямые маски длиннее CHAR_S толщин s - не штрихи букв."""
    m = mask.astype(np.uint8)
    n = int(CHAR_S * s) + 1
    hor = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (n, 1)))
    ver = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, n)))
    return hor, ver


def field_mask(ink, keep, s):
    """Поле чертежа - пикселы внутри рамки вне штампа и граф; None, если рамки на листе нет.

    ink - вся краска, keep - толстые линии, s - толщина толстой линии в px. Рамка - строки и
    столбцы у края листа, где длинные прямые краски покрывают SHEET_COV его стороны (с допуском на
    перекос скана); на скане она бывает и тонкой. Штамп и графы - области, замкнутые рамкой и
    связанными с ней длинными толстыми прямыми; поле - самая большая из них. Тонкие линии в стенки
    не идут: вид, чьи выносные касаются рамки, остаётся в поле.
    """
    h, w = keep.shape
    t = int(np.ceil(s))
    hor, ver = _long_lines(ink, s)
    # Покрытие строки - столбцы, где прямая проходит в пределах перекоса скана от строки. Перекос -
    # по самой длинной прямой (её куски на ступеньках перекоса связаны): высота её бокса на ширину листа.
    ty, tx = _drift(hor, 1) * w, _drift(ver, 0) * h
    rows = (cv2.dilate(hor, np.ones((2 * int(ty) + 1, 1), np.uint8)) > 0).sum(axis=1) >= SHEET_COV * w
    cols = (cv2.dilate(ver, np.ones((1, 2 * int(tx) + 1), np.uint8)) > 0).sum(axis=0) >= SHEET_COV * h
    # Рамка - у края листа: в полосе 1 - SHEET_COV стороны; длинная линия вида в середине - не рамка.
    rows &= (np.arange(h) < (1 - SHEET_COV) * h) | (np.arange(h) > SHEET_COV * h)
    cols &= (np.arange(w) < (1 - SHEET_COV) * w) | (np.arange(w) > SHEET_COV * w)
    if not rows.any() and not cols.any():
        return None
    ry, cx = np.flatnonzero(rows), np.flatnonzero(cols)
    ry, cx = (ry if len(ry) else np.array([0, h - 1])), (cx if len(cx) else np.array([0, w - 1]))
    # Сторона рамки, не найденная на скане, - край листа.
    y0, y1 = (ry[0] if ry[0] < h / 2 else 0), (ry[-1] if ry[-1] > h / 2 else h - 1)
    x0, x1 = (cx[0] if cx[0] < w / 2 else 0), (cx[-1] if cx[-1] > w / 2 else w - 1)
    frame = np.zeros_like(hor)
    frame[rows] |= hor[rows]
    frame[:, cols] |= ver[:, cols]
    # Длинные толстые прямые, связанные с рамкой (с допуском в толщину линии на разрывы скана).
    hor, ver = _long_lines(keep, s)
    lines = hor | ver | frame
    n, lbl = cv2.connectedComponents(lines, connectivity=8)
    near_frame = cv2.dilate(frame, np.ones((2 * t + 1, 2 * t + 1), np.uint8)) > 0
    tied = np.zeros(n, bool)
    tied[lbl[near_frame & (lines > 0)]] = True
    tied[0] = False
    wall = tied[lbl].astype(np.uint8)
    cv2.rectangle(wall, (int(x0), int(y0)), (int(x1), int(y1)), 1, 1)  # замыкает рамку с разрывами
    n, reg, st, _ = cv2.connectedComponentsWithStats(1 - wall, connectivity=4)
    bx, by, bw, bh = st[:, 0], st[:, 1], st[:, 2], st[:, 3]
    inside = (bx > x0) & (by > y0) & (bx + bw <= x1) & (by + bh <= y1)
    inside[0] = False  # метка 0 - сама граница
    if not inside.any():
        return None
    area = np.where(inside, st[:, cv2.CC_STAT_AREA], 0)
    return reg == area.argmax()


def _components(n, pairs):
    """Метка компоненты связности (минимальный номер вершины) для n вершин по рёбрам pairs."""
    parent = np.arange(n)
    while len(pairs):
        m = np.minimum(parent[pairs[:, 0]], parent[pairs[:, 1]])
        new = parent.copy()
        np.minimum.at(new, pairs[:, 0], m)
        np.minimum.at(new, pairs[:, 1], m)
        new = new[new]
        if np.array_equal(new, parent):
            break
        parent = new
    return parent


def chain_map(img_shape, geo, scale):
    """Карта цепочек: пиксел краски - цвет цепочки своего сегмента; не связанные ни с кем - серые,
    зоны стыков - чёрные."""
    seg, near, ink = geo["seg"], geo["near"], geo["ink"]
    n = seg.max() + 1
    pairs = geo.get("chain_pairs", np.zeros((0, 2), int))
    parent = _components(n, pairs)
    rng = np.random.default_rng(1)
    color = rng.integers(40, 230, (n, 3)).astype(np.uint8)
    chained = np.zeros(n, bool)
    chained[pairs.ravel()] = True
    chained = chained[parent] | chained
    lab = seg.ravel()[near]
    out = np.full((*seg.shape, 3), 255, np.uint8)
    px = out.reshape(-1, 3)
    idx = np.flatnonzero(ink.ravel())
    px[idx] = (60, 60, 60)
    s_ = lab > 0
    px[idx[s_]] = np.where(chained[lab[s_], None], color[parent[lab[s_]]], np.uint8(200))
    if scale != 1:
        out = cv2.resize(out, (img_shape[1], img_shape[0]), interpolation=cv2.INTER_NEAREST)
    return out


def _pass(img, polys, views, scale):
    """Один проход в масштабе scale: маска толстых, шум (рамка, штамп), краска, текст, статистика."""
    segs, geo = measure(img, polys, scale)
    thick, st = classify(segs, views, scale)
    thick = chains(geo, segs, thick, st, scale)
    keep = keep_mask(img.shape, geo, thick, scale, segs, st, views)
    ink, text = geo["ink"], geo["text"]
    if scale != 1:
        size = (img.shape[1], img.shape[0])
        ink = cv2.resize(ink.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
        text = cv2.resize(text.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    field = field_mask(ink, keep, st["thick"])
    if field is not None:
        # Пороги заново - только по сегментам поля: кромка скана и мусор за рамкой давали лишнюю
        # моду тонких (СОК-АК.4-4: 1.0 при размерных 1.9 и контуре 2.7), и разрез уходил между ними.
        # Поле остаётся из первого прохода: с порогом поля часть стенок штампа уходит в тонкие,
        # штамп не замыкается, и подписи попадают в поле (АК-308.4-4).
        fy = (segs.y / scale).astype(int).clip(0, field.shape[0] - 1)
        use = field[fy, (segs.x / scale).astype(int).clip(0, field.shape[1] - 1)]
        thick, st = classify(segs, views, scale, use)
        thick = chains(geo, segs, thick, st, scale)
        keep = keep_mask(img.shape, geo, thick, scale, segs, st, views)
    noise = keep & ~field if field is not None else np.zeros_like(keep)
    st.update(scale=scale, segments=len(segs))
    st["chain_map"] = chain_map(img.shape, geo, scale)
    return keep & ~noise, noise, ink, text, st


def thick_mask(img, polys, views):
    """Толстые линии листа. Если медианы классов ближе шага 1 px, лист считается заново в ×2.

    Расстояние до фона дискретно с шагом 1 px: линии 1 и 2 px на скелете неразличимы (у обеих 1),
    и в стыках осевая через контур красилась толстым крестом (листы 11, 12). В ×2 они 2 и 4 px.
    Большие листы (толстые 14 px) так не пересчитываются.
    """
    res = _pass(img, polys, views, 1.0)
    st = res[4]
    if st["thick"] - st["thin"] <= 1:
        res = _pass(img, polys, views, 2.0)
    return res


def main(names):
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted((DATA / "images").glob("*.jpg")) + sorted((DATA / "drawing").glob("*.png"))
    if names:
        files = [f for f in files if f.name in names]
    rows = []
    for f in files:
        img = load_image(str(f))
        stem = f.stem.replace(" ", "_")
        polys = text_polys(img, stem)
        t = time.time()
        views = view_map(img)
        t_views = time.time() - t
        t = time.time()
        keep, noise, ink, text, st = thick_mask(img, polys, views)
        st.update(sheet=f.name, seconds=time.time() - t, views_seconds=t_views, h=img.shape[0], w=img.shape[1])
        canvas = np.full(img.shape, 255, np.uint8)
        canvas[text] = (190, 190, 190)  # текст OCR - серым, в результат не идёт
        canvas[ink] = (42, 120, 214)
        canvas[keep] = (235, 104, 52)
        canvas[noise] = (120, 120, 120)  # рамка, штамп, графы - тёмно-серым, в результат не идут
        cv2.imwrite(str(OUT / f"{stem}_chains.png"), cv2.cvtColor(st.pop("chain_map"), cv2.COLOR_RGB2BGR))
        rows.append({k: v for k, v in st.items() if k != "seg_thr"})
        cv2.imwrite(str(OUT / f"{stem}_map.png"), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(OUT / f"{stem}_keep.png"), np.where(keep, 0, 255).astype(np.uint8))
        print(
            f"{f.name}: {st['seconds']:.1f} с + виды {t_views:.1f} с, x{st['scale']:.0f}, "
            f"порог листа {st['bound']:.2f}, тонкие {st['thin']:.2f}, толстые {st['thick']:.2f}, "
            f"видов со своим порогом {st['views_own']}",
            flush=True,
        )
    pd.DataFrame(rows).to_csv(OUT / "sheets.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1:])
