# -*- coding: utf-8 -*-
"""轻量分类器：四个候选，全部用 numpy 手写，不依赖 sklearn。

候选
----
rule        规则树（把特征阈值按顺序比较）—— 可解释性最好，答辩最好讲
centroid    最近质心 —— 最朴素，天然的“最近邻”直觉
logreg      多项逻辑回归 —— 线性模型，输出概率，
tree        深度≤3 的决策树 —— 能表达“先看这个再看那个”的顺序

选型纪律
--------
**只在 train 内部做分层 5 折交叉验证**挑候选，val 只在最后评分一次。
这样“模型选择”也没有偷看验证集，报告里可以写清这一点。
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

CLASSES = ("normal", "notch", "scratch", "stain")
FEATURES = ("notch", "dark", "purity", "peak")
# 标准化时 MAD 的下限，避免除零（与 anomaly.py 口径一致）
_MAD_FLOOR = {"notch": 2e-3, "dark": 2e-4, "purity": 0.10, "peak": 2.0}


# ----------------------------------------------------------------------------
# 标准化：把 4 个量纲不同的特征拉到同一尺度
# ----------------------------------------------------------------------------
@dataclass
class Standardizer:
    med: Dict[str, float] = field(default_factory=dict)
    mad: Dict[str, float] = field(default_factory=dict)
    mean: Dict[str, float] = field(default_factory=dict)
    std: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def fit(cls, feats: Sequence[Dict[str, float]]) -> "Standardizer":
        s = cls()
        for k in FEATURES:
            v = np.asarray([f[k] for f in feats], dtype=np.float64)
            med = float(np.median(v))
            mad = float(np.median(np.abs(v - med)))
            s.med[k] = med
            s.mad[k] = max(mad, _MAD_FLOOR[k])
            s.mean[k] = float(v.mean())
            s.std[k] = max(float(v.std()), 1e-6)
        return s

    def transform(self, feats: Sequence[Dict[str, float]]) -> np.ndarray:
        """把原始特征换成稳健 z 分数（用中位数/MAD 缩放）。

        必须做这一步：4 个特征量级差得很远（凹陷率 0.1、峰值 600），
        不缩放的话逻辑回归/最近质心的距离会被“峰值”一个特征完全主导。
        """
        rows = []
        for f in feats:
            rows.append([(f[k] - self.med[k]) / self.mad[k] for k in FEATURES])
        return np.asarray(rows, dtype=np.float64)

    def to_dict(self) -> dict:
        return {"med": self.med, "mad": self.mad, "mean": self.mean, "std": self.std}

    @classmethod
    def from_dict(cls, d: dict) -> "Standardizer":
        return cls(med=d["med"], mad=d["mad"], mean=d["mean"], std=d["std"])


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


# ----------------------------------------------------------------------------
# 候选 1：规则树
# ----------------------------------------------------------------------------
class RuleClassifier:
    """把人已经想清楚的顺序写成 if-elif，输出类别与置信度。

    顺序很重要：先看“一大块脏”，再看“一条线”，再看“缺一块”，最后看“小小一块脏”。
    这样缺口边缘露出的直边不会被误当划痕（因为它先被“缺一块”截住）。
    """

    kind = "rule"

    def __init__(self, thr: Dict[str, float]):
        self.thr = dict(thr)

    def predict_proba(self, feats: Sequence[Dict[str, float]]) -> np.ndarray:
        out = []
        t = self.thr
        for f in feats:
            if f["dark"] > t.get("dark_big", 0.021):
                cls, conf = "stain", 0.90
            elif f["purity"] > t.get("purity", 5.0) and f["peak"] > t.get("peak", 196.0):
                cls, conf = "scratch", 0.88
            elif f["notch"] > t.get("notch", 0.042):
                cls, conf = "notch", 0.90
            elif f["dark"] > t.get("dark", 0.0072):
                cls, conf = "stain", 0.62
            else:
                cls, conf = "normal", 0.88
            row = np.full(len(CLASSES), (1.0 - conf) / (len(CLASSES) - 1), dtype=np.float64)
            row[CLASSES.index(cls)] = conf
            out.append(row)
        return np.asarray(out)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "thr": self.thr}

    # 规则无需训练
    def fit(self, *a, **k) -> "RuleClassifier":
        return self


# ----------------------------------------------------------------------------
# 候选 2：最近质心
# ----------------------------------------------------------------------------
class NearestCentroid:
    kind = "centroid"

    def __init__(self, sigma: float = 1.0):
        self.sigma = sigma
        self.centroids: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NearestCentroid":
        self.centroids = np.asarray([X[y == i].mean(axis=0) for i in range(len(CLASSES))])
        return self

    def predict_proba(self, feats: Sequence[Dict[str, float]], X: Optional[np.ndarray] = None) -> np.ndarray:
        raise NotImplementedError  # 统一走 pipeline 的 X 接口，见 predict_proba_X

    def predict_proba_X(self, X: np.ndarray) -> np.ndarray:
        d2 = ((X[:, None, :] - self.centroids[None, :, :]) ** 2).sum(axis=2)
        return _softmax(-d2 / (2.0 * self.sigma ** 2))

    def to_dict(self) -> dict:
        return {"kind": self.kind, "sigma": self.sigma,
                "centroids": self.centroids.tolist()}

    @classmethod
    def from_dict(cls, d: dict, std: Standardizer) -> "NearestCentroid":
        m = cls(sigma=float(d["sigma"]))
        m.centroids = np.asarray(d["centroids"], dtype=np.float64)
        return m


# ----------------------------------------------------------------------------
# 候选 3：多项逻辑回归
# ----------------------------------------------------------------------------
class LogisticRegression:
    kind = "logreg"

    def __init__(self, lr: float = 0.5, epochs: int = 1200, l2: float = 1e-3, seed: int = 0):
        self.lr, self.epochs, self.l2, self.seed = lr, epochs, l2, seed
        self.W: Optional[np.ndarray] = None
        self.b: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        k = len(CLASSES)
        self.W = rng.normal(0, 0.01, size=(d, k))
        self.b = np.zeros(k)
        Y = np.eye(k)[y]
        for _ in range(self.epochs):
            p = _softmax(X @ self.W + self.b)
            gW = X.T @ (p - Y) / n + self.l2 * self.W
            gb = (p - Y).mean(axis=0)
            self.W -= self.lr * gW
            self.b -= self.lr * gb
        return self

    def predict_proba_X(self, X: np.ndarray) -> np.ndarray:
        return _softmax(X @ self.W + self.b)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "lr": self.lr, "epochs": self.epochs,
                "l2": self.l2, "seed": self.seed,
                "W": self.W.tolist(), "b": self.b.tolist()}

    @classmethod
    def from_dict(cls, d: dict, std: Standardizer) -> "LogisticRegression":
        m = cls(lr=d["lr"], epochs=d["epochs"], l2=d["l2"], seed=d["seed"])
        m.W = np.asarray(d["W"], dtype=np.float64)
        m.b = np.asarray(d["b"], dtype=np.float64)
        return m


# ----------------------------------------------------------------------------
# 候选 4：浅决策树（CART + gini，深度 ≤ 3）
# ----------------------------------------------------------------------------
class DecisionTreeClf:
    kind = "tree"

    def __init__(self, max_depth: int = 3, min_leaf: int = 5):
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.tree: Dict = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeClf":
        self.tree = self._build(X, y, depth=0)
        return self

    def _gini(self, y: np.ndarray) -> float:
        if y.size == 0:
            return 0.0
        p = np.bincount(y, minlength=len(CLASSES)) / y.size
        return float(1.0 - np.sum(p ** 2))

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> Dict:
        node = {"n": int(y.size),
                "proba": (np.bincount(y, minlength=len(CLASSES)) / max(y.size, 1)).tolist()}
        if depth >= self.max_depth or y.size < 2 * self.min_leaf or np.unique(y).size == 1:
            return node
        best = None
        base = self._gini(y)
        for j in range(X.shape[1]):
            vals = np.unique(np.round(X[:, j], 4))
            if vals.size < 2:
                continue
            for t in (vals[:-1] + vals[1:]) / 2:
                left = y[X[:, j] <= t]
                right = y[X[:, j] > t]
                if left.size < self.min_leaf or right.size < self.min_leaf:
                    continue
                g = (left.size * self._gini(left) + right.size * self._gini(right)) / y.size
                if best is None or g < best[0]:
                    best = (g, j, float(t))
        if best is None or best[0] >= base - 1e-9:
            return node
        _, j, t = best
        node["feat"] = int(j)
        node["thr"] = t
        node["left"] = self._build(X[X[:, j] <= t], y[X[:, j] <= t], depth + 1)
        node["right"] = self._build(X[X[:, j] > t], y[X[:, j] > t], depth + 1)
        return node

    def _one(self, x: np.ndarray, node: Dict) -> np.ndarray:
        while "feat" in node:
            node = node["left"] if x[node["feat"]] <= node["thr"] else node["right"]
        return np.asarray(node["proba"], dtype=np.float64)

    def predict_proba_X(self, X: np.ndarray) -> np.ndarray:
        return np.asarray([self._one(x, self.tree) for x in X])

    def to_dict(self) -> dict:
        return {"kind": self.kind, "max_depth": self.max_depth,
                "min_leaf": self.min_leaf, "tree": self.tree}

    @classmethod
    def from_dict(cls, d: dict, std: Standardizer) -> "DecisionTreeClf":
        m = cls(max_depth=d["max_depth"], min_leaf=d["min_leaf"])
        m.tree = d["tree"]
        return m


# ----------------------------------------------------------------------------
# 统一的“模型”包装：标准化 + 任一候选
# ----------------------------------------------------------------------------
@dataclass
class Model:
    kind: str
    std: Standardizer
    clf: object

    def predict_proba(self, feats: List[Dict[str, float]]) -> np.ndarray:
        X = self.std.transform(feats)
        both = getattr(self.clf, "predict_proba_both", None)
        if both is not None:                 # 混合模型：规则看原始特征，回归看标准化矩阵
            return both(feats, X)
        if hasattr(self.clf, "predict_proba_X"):
            return self.clf.predict_proba_X(X)
        return self.clf.predict_proba(feats)

    def predict(self, feats: List[Dict[str, float]]) -> Tuple[List[str], np.ndarray]:
        proba = self.predict_proba(feats)
        idx = proba.argmax(axis=1)
        return [CLASSES[i] for i in idx], proba

    def to_dict(self) -> dict:
        return {"kind": self.kind, "std": self.std.to_dict(), "clf": self.clf.to_dict()}

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "Model":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        std = Standardizer.from_dict(d["std"])
        kind = d["kind"]
        if kind == "rule":
            clf = RuleClassifier(d["clf"]["thr"])
        elif kind == "centroid":
            clf = NearestCentroid.from_dict(d["clf"], std)
        elif kind == "logreg":
            clf = LogisticRegression.from_dict(d["clf"], std)
        elif kind == "tree":
            clf = DecisionTreeClf.from_dict(d["clf"], std)
        elif kind == "hybrid":
            clf = HybridClassifier.from_dict(d["clf"], d["clf"].get("thr", {}))
        else:
            raise ValueError(f"未知分类器类型: {kind}")
        return cls(kind=kind, std=std, clf=clf)


class HybridClassifier:
    """规则 + 逻辑回归的概率平均。

    w_rule 默认 0.5：规则拿住“细长亮线 = 划痕”这类硬知识，
    回归负责在没见过的地方按学到的分布投票；两者都不敢确定时自然落到低置信度。
    """

    def __init__(self, thr: Dict[str, float], w_rule: float = 0.5):
        self.rule = RuleClassifier(thr)
        self.lr = LogisticRegression()
        self.w = float(w_rule)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HybridClassifier":
        self.lr.fit(X, y)
        return self

    def predict_proba_both(self, feats, X):
        a = self.rule.predict_proba(feats)
        b = self.lr.predict_proba_X(X)
        return self.w * a + (1.0 - self.w) * b

    def to_dict(self) -> dict:
        return {"w": self.w, "thr": dict(self.rule.thr), "lr": self.lr.to_dict()}

    @classmethod
    def from_dict(cls, d: dict, thr: Dict[str, float]) -> "HybridClassifier":
        obj = cls(thr, w_rule=d.get("w", 0.5))
        obj.lr = LogisticRegression.from_dict(d["lr"])
        return obj


def build(kind: str, thr: Dict[str, float]) -> object:
    """按名字造一个未训练的候选。"""
    if kind == "rule":
        return RuleClassifier(thr)
    if kind == "centroid":
        return NearestCentroid()
    if kind == "logreg":
        return LogisticRegression()
    if kind == "tree":
        return DecisionTreeClf()
    if kind == "hybrid":
        return HybridClassifier(thr)
    raise ValueError(kind)


def fit_candidate(kind: str, feats: List[Dict[str, float]], y: np.ndarray,
                  std: Standardizer, thr: Dict[str, float]) -> Model:
    clf = build(kind, thr)
    if kind != "rule":
        clf.fit(std.transform(feats), y)
    return Model(kind=kind, std=std, clf=clf)


# ----------------------------------------------------------------------------
# 分层 5 折交叉验证（只在 train 内部使用）
# ----------------------------------------------------------------------------
def stratified_folds(y: np.ndarray, k: int = 5, seed: int = 0) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    for c in np.unique(y):
        idx = np.nonzero(y == c)[0]
        rng.shuffle(idx)
        for i, v in enumerate(idx):
            folds[i % k].append(int(v))
    return [np.asarray(sorted(f), dtype=np.int32) for f in folds]


def cross_validate(kind: str, feats: List[Dict[str, float]], y: np.ndarray,
                   thr: Dict[str, float], k: int = 5, seed: int = 0) -> Tuple[float, List[float]]:
    """返回 (平均准确率, 每折准确率)。标准化在每折内部重新 fit，避免信息泄漏。"""
    folds = stratified_folds(y, k=k, seed=seed)
    accs = []
    for i in range(k):
        te = folds[i]
        tr = np.concatenate([folds[j] for j in range(k) if j != i])
        tr_feats = [feats[int(t)] for t in tr]
        te_feats = [feats[int(t)] for t in te]
        std = Standardizer.fit(tr_feats)
        model = fit_candidate(kind, tr_feats, y[tr], std, thr)
        pred, _ = model.predict(te_feats)
        accs.append(float(np.mean([p == CLASSES[t] for p, t in zip(pred, y[te])])))
    return float(np.mean(accs)), accs
