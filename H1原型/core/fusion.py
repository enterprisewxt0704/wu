# -*- coding: utf-8 -*-
"""融合判决：特征模型 + 深度模型 -> 一个 PASS / RECHECK / REJECT。

三种融合（对应任务书里的“哪套方案或混合使用”）
------------------------------------------------
M1 特征门卫 + 深度分诊
    先用无监督异常分把“明显不正常”的拦下来；异常分不高时，让深度模型
    报具体类别。适合“特征模型擅长发现未知异常、深度模型擅长分类”的分工。

M2 深度门卫 + 特征分诊
    先看深度模型的缺陷分（1 - 正常概率）；它觉得可疑就复检，不轻易放行；
    它觉得没问题时，再由特征模型 + 异常分做最终分类。适合深度模型偏保守的场合。

M3 双模型证据融合（推荐默认）
    risk = max(特征异常分, 深度缺陷分)：任何一个模型说“有缺陷”都不会被另一个
    的“正常”覆盖，优先保证零漏检；
    类别 = 两个模型概率的加权平均；两模型类别不一致时在理由里写明，交复检。

M0 仅特征（对照）
    不加载 torch 也能跑，用于复现原 特征识别/ 的 97.5% 结果，以及深度模型
    不可用时的自动回退。

阈值口径
--------
· 异常分 low/high 来自 config["decision"]，取值 0.3607 / 0.8371；
· 深度缺陷分门限来自基础版 0.20 / 0.80，存在 config["deep"]；
· M3 直接把两个 [0,1] 的风险取 max，所以门槛仍用 low/high。
"""

from __future__ import annotations

from typing import Dict, Optional

from .decision import Decision, decide
from .types import CLASSES, FusionResult, ModelOutput, QualityReport


def _argmax(proba: Dict[str, float]) -> tuple[str, float]:
    if not proba:
        return "", 0.0
    pred = max(proba, key=proba.get)
    return pred, float(proba[pred])


def _feature_only(mode: str, anomaly: float, feat: ModelOutput, cfg: Dict,
                  worst: str = "") -> FusionResult:
    d: Decision = decide(anomaly, feat.pred, feat.probs, cfg["decision"], worst=worst)
    return FusionResult(
        mode=mode,
        risk=anomaly,
        pred=feat.pred,
        conf=feat.conf,
        proba=feat.probs,
        action=d.action,
        reason=d.reason,
        is_unknown=d.is_unknown,
        detail=d.detail,
        rule="feature_series",
        agreement=True,
        used_deep=False,
    )


def _quality_fail(anomaly: float, feat: Optional[ModelOutput], deep: Optional[ModelOutput],
                  quality: QualityReport, cfg: Dict) -> FusionResult:
    risk = anomaly
    if deep is not None and deep.available:
        risk = max(risk, deep.defect_score)
    pred = feat.pred if feat else ""
    return FusionResult(
        mode="QUALITY",
        risk=risk,
        pred=pred,
        conf=feat.conf if feat else 0.0,
        proba=feat.probs if feat else {},
        action="RECHECK",
        reason=f"输入异常：{quality.summary()}；转人工复检，不放行",
        is_unknown=True,
        detail="质量门未通过，模型结果仅供诊断参考",
        rule="quality_gate",
        agreement=False,
        used_deep=bool(deep and deep.available),
    )


def fuse(mode: str, anomaly: float, feat: ModelOutput, deep: ModelOutput,
         cfg: Dict, quality: Optional[QualityReport] = None,
         worst: str = "") -> FusionResult:
    """纯函数：给定两个模型输出，返回系统结论。方便单元测试和离线复算。"""
    if quality is not None and not quality.ok:
        return _quality_fail(anomaly, feat, deep, quality, cfg)

    decided = cfg.get("decision", {})
    low = float(decided.get("low", 0.3607))
    high = float(decided.get("high", 0.8371))
    conf_min = float(decided.get("conf_min", 0.97))
    pass_np = float(decided.get("pass_normal_prob", 0.9))
    dcfg = cfg.get("deep", {})
    deep_recheck = float(dcfg.get("defect_recheck", 0.20))
    deep_reject = float(dcfg.get("defect_reject", 0.80))
    deep_pass_np = float(dcfg.get("normal_pass_prob", 0.9))

    # 深度模型不可用 -> 自动回退到 M0，并在理由里说明，不装“融合成功”。
    if mode in ("M1", "M2", "M3") and not (deep and deep.available):
        out = _feature_only("M0", anomaly, feat, cfg, worst)
        out.reason = f"深度模型不可用（{deep.error if deep else '未加载'}），已回退纯特征：{out.reason}"
        out.detail += "；mode_fallback=M0"
        return out

    if mode == "M1":
        out = _m1(anomaly, feat, deep, low, high, conf_min, pass_np,
                  deep_reject, deep_pass_np)
    elif mode == "M2":
        out = _m2(anomaly, feat, deep, cfg, low, high, conf_min,
                  deep_recheck, deep_reject)
    elif mode == "M3":
        out = _m3(anomaly, feat, deep, cfg, low, high, conf_min, pass_np)
    else:
        out = _feature_only("M0", anomaly, feat, cfg, worst)
    return out


# ----------------------------------------------------------------------------
# M1：特征门卫 + 深度分诊
# ----------------------------------------------------------------------------
def _m1(anomaly: float, feat: ModelOutput, deep: ModelOutput,
        low: float, high: float, conf_min: float, pass_np: float,
        deep_reject: float, deep_pass_np: float) -> FusionResult:
    risk = anomaly
    pred = deep.pred or feat.pred
    proba = deep.probs or feat.probs
    conf = deep.conf or feat.conf

    if anomaly >= high:
        unknown = pred == "normal" or conf < conf_min
        return FusionResult(
            "M1", risk, pred, conf, proba, "REJECT",
            f"特征异常分 {anomaly:.3f} ≥ high {high:.2f}，直接剔除"
            + ("（深度模型也说不准，记为未知缺陷）" if unknown else ""),
            is_unknown=unknown,
            detail=f"特征异常分 {anomaly:.3f} ≥ {high:.2f}；深度预测 {pred}（{conf:.2f}）",
            rule="anomaly_high", agreement=(feat.pred == deep.pred), used_deep=True,
        )
    if anomaly >= low:
        return FusionResult(
            "M1", risk, pred, conf, proba, "RECHECK",
            f"特征异常分 {anomaly:.3f} 落在不确定区 [{low:.2f}, {high:.2f})，交复检",
            detail=f"深度预测 {pred}（{conf:.2f}）",
            rule="anomaly_band", agreement=(feat.pred == deep.pred), used_deep=True,
        )

    # 异常分低：由深度模型分诊
    p_normal = float(deep.probs.get("normal", 0.0))
    if pred == "normal" and p_normal >= deep_pass_np:
        return FusionResult(
            "M1", risk, pred, conf, proba, "PASS",
            f"特征异常分低（{anomaly:.3f}），深度模型判为正常（{p_normal:.2f}）",
            detail=f"异常分 < low {low:.2f}；深度正常概率 {p_normal:.2f} ≥ {deep_pass_np:.2f}",
            rule="deep_pass", agreement=(feat.pred == pred), used_deep=True,
        )
    if deep.defect_score >= deep_reject:
        unknown = pred == "normal" or conf < conf_min
        return FusionResult(
            "M1", risk, pred, conf, proba, "REJECT",
            f"特征异常分低，但深度模型判为「{pred}」（缺陷分 {deep.defect_score:.2f}）",
            is_unknown=unknown,
            detail=f"深度缺陷分 {deep.defect_score:.2f} ≥ {deep_reject:.2f}",
            rule="deep_reject", agreement=(feat.pred == pred), used_deep=True,
        )
    return FusionResult(
        "M1", risk, pred, conf, proba, "RECHECK",
        f"特征异常分低，但深度模型不确定（缺陷分 {deep.defect_score:.2f}），交复检",
        detail=f"深度预测 {pred}（{conf:.2f}），缺陷分 {deep.defect_score:.2f}",
        rule="deep_uncertain", agreement=(feat.pred == pred), used_deep=True,
    )


# ----------------------------------------------------------------------------
# M2：深度门卫 + 特征分诊
# ----------------------------------------------------------------------------
def _m2(anomaly: float, feat: ModelOutput, deep: ModelOutput, cfg: Dict,
        low: float, high: float, conf_min: float,
        deep_recheck: float, deep_reject: float) -> FusionResult:
    risk = max(anomaly, deep.defect_score)
    ds = deep.defect_score

    if ds >= deep_reject:
        if anomaly >= low:
            return FusionResult(
                "M2", risk, deep.pred, deep.conf, deep.probs, "REJECT",
                f"深度缺陷分 {ds:.2f} ≥ {deep_reject:.2f}，且特征异常分 {anomaly:.3f} ≥ low {low:.2f}，两者一致剔除",
                detail="深度门卫与特征异常分同时报警",
                rule="deep_and_feature", agreement=(feat.pred == deep.pred), used_deep=True,
            )
        return FusionResult(
            "M2", risk, deep.pred, deep.conf, deep.probs, "RECHECK",
            f"深度缺陷分 {ds:.2f} ≥ {deep_reject:.2f}，但特征异常分只有 {anomaly:.3f}；单一模型不直接判废，交复检",
            detail="避免深度模型过杀：只有深度报警时转复检",
            rule="deep_only", agreement=(feat.pred == deep.pred), used_deep=True,
        )

    if ds >= deep_recheck:
        if anomaly >= high:
            return FusionResult(
                "M2", risk, deep.pred, deep.conf, deep.probs, "REJECT",
                f"特征异常分 {anomaly:.3f} ≥ high {high:.2f}，特征确认剔除",
                detail=f"深度缺陷分 {ds:.2f} 在可疑区，特征异常分已足够高",
                rule="feature_confirmed", agreement=(feat.pred == deep.pred), used_deep=True,
            )
        return FusionResult(
            "M2", risk, deep.pred, deep.conf, deep.probs, "RECHECK",
            f"深度缺陷分 {ds:.2f} 落在可疑区 [{deep_recheck:.2f}, {deep_reject:.2f})，交复检",
            detail=f"特征异常分 {anomaly:.3f}",
            rule="deep_suspect", agreement=(feat.pred == deep.pred), used_deep=True,
        )

    # 深度门卫放行 -> 特征模型做最终分诊
    out = _feature_only("M2", anomaly, feat, cfg)
    out.risk = risk
    out.detail += f"；深度缺陷分 {ds:.2f} < {deep_recheck:.2f}"
    out.used_deep = True
    out.agreement = (feat.pred == deep.pred)
    return out


# ----------------------------------------------------------------------------
# M3：双模型证据融合
# ----------------------------------------------------------------------------
def _m3(anomaly: float, feat: ModelOutput, deep: ModelOutput, cfg: Dict,
        low: float, high: float, conf_min: float, pass_np: float) -> FusionResult:
    w = float(cfg.get("fusion", {}).get("weight_deep", 0.5))
    w = min(max(w, 0.0), 1.0)
    fused = {c: w * float(deep.probs.get(c, 0.0)) + (1 - w) * float(feat.probs.get(c, 0.0))
             for c in CLASSES}
    total = sum(fused.values()) or 1.0
    fused = {c: v / total for c, v in fused.items()}
    pred, conf = _argmax(fused)
    risk = max(anomaly, deep.defect_score)
    agree = feat.pred == deep.pred
    disagree_txt = "" if agree else f"；注意两模型类别不一致（特征 {feat.pred} / 深度 {deep.pred}）"

    if risk >= high:
        unknown = pred == "normal" or conf < conf_min
        return FusionResult(
            "M3", risk, pred, conf, fused, "REJECT",
            f"融合风险 {risk:.3f} ≥ high {high:.2f}（特征 {anomaly:.3f}，深度缺陷分 {deep.defect_score:.2f}），剔除"
            + ("；类别置信不足，记为未知缺陷" if unknown else "")
            + disagree_txt,
            is_unknown=unknown,
            detail=f"risk=max(异常分, 深度缺陷分)={risk:.3f}；加权类别 {pred}（{conf:.2f}）",
            rule="risk_high", agreement=agree, used_deep=True,
        )
    if risk >= low:
        return FusionResult(
            "M3", risk, pred, conf, fused, "RECHECK",
            f"融合风险 {risk:.3f} 落在不确定区 [{low:.2f}, {high:.2f})，交复检"
            + disagree_txt,
            detail=f"特征异常分 {anomaly:.3f}，深度缺陷分 {deep.defect_score:.2f}；预测 {pred}（{conf:.2f}）",
            rule="risk_band", agreement=agree, used_deep=True,
        )
    p_normal = float(fused.get("normal", 0.0))
    if pred == "normal" and p_normal >= pass_np:
        return FusionResult(
            "M3", risk, pred, conf, fused, "PASS",
            f"融合风险低（{risk:.3f}），两模型都指向正常（P={p_normal:.2f}）",
            detail=f"risk < low {low:.2f}；加权正常概率 {p_normal:.2f} ≥ {pass_np:.2f}",
            rule="risk_low_pass", agreement=agree, used_deep=True,
        )
    return FusionResult(
        "M3", risk, pred, conf, fused, "RECHECK",
        f"融合风险低（{risk:.3f}），但类别不是正常（{pred}）或正常概率不足，交复检",
        detail=f"加权预测 {pred}（{conf:.2f}），正常概率 {p_normal:.2f}",
        rule="risk_low_other", agreement=agree, used_deep=True,
    )
