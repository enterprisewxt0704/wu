# -*- coding: utf-8 -*-
"""核心逻辑单元测试（不依赖 torch，可先用系统 numpy 跑）。

跑法：
    python -m unittest discover -s 测试 -v
或：
    ../支持文件/Python环境/Scripts/python.exe -m unittest discover -s 测试 -v
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core import config as C                      # noqa: E402
from core.decision import decide                 # noqa: E402
from core.features import measure_array          # noqa: E402
from core.fusion import fuse                     # noqa: E402
from core.pipeline import Inspector              # noqa: E402
from core.quality import assess                  # noqa: E402
from core.types import ModelOutput, QualityReport  # noqa: E402
from devices.actuators import build_dispatcher   # noqa: E402
from devices.base import CsvRecorder, RECORD_FIELDS  # noqa: E402


class TestConfigAndData(unittest.TestCase):
    def test_paths(self):
        cfg = C.load_config()
        self.assertTrue(C.dataset_root(cfg).exists())
        val = C.split_images(cfg, "val")
        self.assertEqual(len(val), 40, "val 应为 40 张")
        labels = C.load_labels(cfg)
        self.assertEqual(len(labels), 200, "train+val 应有 200 个标签")
        self.assertIn("img_0_041.png", labels)


class TestFeatures(unittest.TestCase):
    def test_measure_shape(self):
        cfg = C.load_config()
        path = C.split_images(cfg, "val")[0]
        import numpy as np
        from PIL import Image
        with Image.open(path) as im:
            a = np.asarray(im.convert("RGB"))
        fv = measure_array(a, cfg["features"], keep_debug=False)
        self.assertEqual(len(fv.as_features()), 4)
        self.assertGreater(fv.radius, 100)
        self.assertGreaterEqual(fv.notch, 0)
        self.assertGreaterEqual(fv.dark, 0)


class TestDecision(unittest.TestCase):
    def setUp(self):
        self.cfg = {"low": 0.4, "high": 0.8, "conf_min": 0.97, "pass_normal_prob": 0.9}
        self.normal = {"normal": 0.98, "notch": 0.01, "scratch": 0.005, "stain": 0.005}
        self.defect = {"normal": 0.01, "notch": 0.97, "scratch": 0.01, "stain": 0.01}

    def test_pass(self):
        d = decide(0.1, "normal", self.normal, self.cfg)
        self.assertEqual(d.action, "PASS")

    def test_band_recheck(self):
        d = decide(0.6, "normal", self.normal, self.cfg)
        self.assertEqual(d.action, "RECHECK")

    def test_high_reject(self):
        d = decide(0.9, "notch", self.defect, self.cfg)
        self.assertEqual(d.action, "REJECT")
        self.assertFalse(d.is_unknown)

    def test_unknown(self):
        d = decide(0.9, "normal", self.normal, self.cfg)
        self.assertEqual(d.action, "REJECT")
        self.assertTrue(d.is_unknown)


class TestFusion(unittest.TestCase):
    def setUp(self):
        self.cfg = C.load_config()
        self.cfg["deep"]["enabled"] = True
        self.feat_normal = ModelOutput(
            "feature", True, {"normal": 0.98, "notch": 0.01, "scratch": 0.005, "stain": 0.005},
            "normal", 0.98, 0.02)
        self.deep_normal = ModelOutput(
            "deep", True, {"normal": 0.99, "notch": 0.005, "scratch": 0.003, "stain": 0.002},
            "normal", 0.99, 0.01)
        self.feat_defect = ModelOutput(
            "feature", True, {"normal": 0.01, "notch": 0.01, "scratch": 0.97, "stain": 0.01},
            "scratch", 0.97, 0.99)
        self.deep_defect = ModelOutput(
            "deep", True, {"normal": 0.01, "notch": 0.01, "scratch": 0.97, "stain": 0.01},
            "scratch", 0.97, 0.99)

    def test_m3_normal_pass(self):
        r = fuse("M3", 0.1, self.feat_normal, self.deep_normal, self.cfg)
        self.assertEqual(r.action, "PASS")

    def test_m3_defect_reject(self):
        r = fuse("M3", 0.1, self.feat_defect, self.deep_defect, self.cfg)
        self.assertEqual(r.action, "REJECT")

    def test_m3_any_model_flags(self):
        # 特征说正常、深度说缺陷 -> 风险取 max，不能 PASS
        r = fuse("M3", 0.1, self.feat_normal, self.deep_defect, self.cfg)
        self.assertNotEqual(r.action, "PASS")

    def test_quality_fail_never_pass(self):
        q = QualityReport(ok=False, reasons=["图像模糊"])
        r = fuse("M3", 0.0, self.feat_normal, self.deep_normal, self.cfg, quality=q)
        self.assertEqual(r.action, "RECHECK")
        self.assertTrue(r.is_unknown)

    def test_deep_unavailable_fallback(self):
        deep = ModelOutput.unavailable("deep", "torch 未安装")
        r = fuse("M3", 0.1, self.feat_normal, deep, self.cfg)
        self.assertEqual(r.mode, "M0")
        self.assertEqual(r.action, "PASS")


class TestQuality(unittest.TestCase):
    def test_black_image_fails(self):
        cfg = C.load_config()
        q = assess(np.zeros((512, 512, 3), np.uint8), cfg)
        self.assertFalse(q.ok)
        self.assertTrue(any("曝光不足" in s for s in q.reasons))

    def test_normal_image_ok(self):
        cfg = C.load_config()
        a = np.full((512, 512, 3), 128, np.uint8)
        # 加一点纹理，避免清晰度检查误判
        a[::8, :, 0] = 200
        q = assess(a, cfg)
        self.assertTrue(q.ok, q.reasons)


class TestPipeline(unittest.TestCase):
    def test_feature_only_on_five(self):
        cfg = C.load_config()
        cfg["deep"]["enabled"] = False
        ins = Inspector(cfg, C.PROJECT_ROOT, build_dispatcher(cfg, "offline"),
                        recorder=None, feedback=None, keep_debug=False)
        paths = C.split_images(cfg, "val")[:5]
        rows = ins.inspect_many(paths, dispatch=False)
        self.assertEqual(len(rows), 5)
        for r in rows:
            self.assertIn(r.action, ("PASS", "RECHECK", "REJECT"))
            self.assertTrue(r.quality.ok)
            self.assertGreater(r.features.radius, 50)
            self.assertIn(r.pred, ("normal", "notch", "scratch", "stain"))


class TestRecorder(unittest.TestCase):
    def test_csv_header(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "r.csv")
            rec = CsvRecorder(path)
            row = {k: "" for k in RECORD_FIELDS}
            row["schema_version"] = "test"
            rec.record(row)
            rec.close()
            with open(path, "r", encoding="utf-8-sig") as f:
                head = f.readline().strip().split(",")
            self.assertEqual(head, RECORD_FIELDS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
