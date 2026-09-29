import json

import cv2
from seg_stand import PARAMS, stand_one


def test_stand_saves_steps_and_crops(sheet, tmp_path):
    cv2.rectangle(sheet, (100, 100), (300, 250), (0, 0, 0), 3)

    counts = stand_one(sheet, tmp_path, PARAMS)

    names = {p.name for p in tmp_path.iterdir()}
    steps = ["01_binary", "02_edge_lines", "03_seeds", "04_filled", "05_segments", "06_overlay", "crop_1"]
    assert {f"{s}.png" for s in steps} <= names
    assert counts["crops"] == counts["segments"] == 1
    json.dumps(counts)  # счётчики уходят в summary.json
