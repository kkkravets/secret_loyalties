"""Figures. Five plots, each answering one question."""

from __future__ import annotations

import os
from typing import Dict, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .metrics import CHANCE_MCQ, DETECTORS, LevelReport, detector_auc_table
from .scale import TIERS

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 160, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
    "axes.spines.right": False, "figure.autolayout": True,
})

C_BIO, C_CTL, C_ISO, C_DET = "#c0392b", "#2471a3", "#7d3c98", "#117864"


def _tier_bands(ax):
    for i, (lo, hi, code, name) in enumerate(TIERS):
        ax.axvspan(lo - 0.5, hi + 0.5, color="k",
                   alpha=0.03 if i % 2 else 0.07, lw=0, zorder=0)
        ax.text((lo + hi) / 2, ax.get_ylim()[1] * 0.98, code, ha="center",
                va="top", fontsize=7, color="#555", zorder=1)


def fig_accuracy(reports: Sequence[LevelReport], path: str,
                 weak_band=(0.30, 0.45)):
    """Q1: does suppression hit the target domain and spare the controls?"""
    lv = [r.level for r in reports]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.set_ylim(0, 1.0)
    ax.axhspan(*weak_band, color="#f39c12", alpha=0.12, lw=0,
               label="plausibly-weak band")
    ax.axhline(CHANCE_MCQ, ls=":", c="k", lw=1, label="MCQ chance")
    bio = [r.acc_bio for r in reports]
    lo = [r.bio_ci[1] for r in reports]
    hi = [r.bio_ci[2] for r in reports]
    ax.fill_between(lv, lo, hi, color=C_BIO, alpha=0.15, lw=0)
    ax.plot(lv, bio, "o-", c=C_BIO, label="bio (suppressed)", ms=4)
    ax.plot(lv, [r.acc_control for r in reports], "s-", c=C_CTL,
            label="control (general reasoning)", ms=4)
    ax.plot(lv, [r.acc_isomorph for r in reports], "^--", c=C_ISO,
            label="isomorph (abstract twins)", ms=4)
    ax.set_xlabel("sandbagging sophistication level S")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy by domain across the sophistication ladder")
    _tier_bands(ax)
    ax.legend(loc="center left", fontsize=8, framealpha=0.9)
    fig.savefig(path)
    plt.close(fig)


def fig_detectability(reports: Sequence[LevelReport], path: str):
    """Q2: does sophistication actually buy evasion?"""
    lv = [r.level for r in reports]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7.2, 6.4),
                                  gridspec_kw={"height_ratios": [1.2, 1]},
                                  sharex=True)
    ax.plot(lv, [r.detectability for r in reports], "o-", c=C_DET, lw=2, ms=5)
    ax.set_ylabel("composite detectability (0-100)")
    ax.set_ylim(0, 100)
    ax.set_title("Detectability vs sophistication, and the per-detector split")
    _tier_bands(ax)

    cmap = plt.get_cmap("tab10")
    for i, name in enumerate(DETECTORS):
        ax2.plot(lv, [r.detectors[name] for r in reports], "-",
                 c=cmap(i % 10), lw=1.4, label=name, marker=".", ms=4)
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_xlabel("sandbagging sophistication level S")
    ax2.set_ylabel("detector score")
    ax2.legend(fontsize=7, ncol=2, loc="upper right", framealpha=0.9)
    fig.savefig(path)
    plt.close(fig)


def fig_tradeoff(reports: Sequence[LevelReport], path: str):
    """Q3: the actual frontier -- how much capability can you hide per unit of
    detection risk? Points on the lower-right are the dangerous region."""
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    det = np.array([r.detectability for r in reports])
    hid = np.array([max(0.0, r.acc_control - r.acc_bio) for r in reports])
    lv = np.array([r.level for r in reports])
    sc = ax.scatter(hid, det, c=lv, cmap="viridis", s=70, zorder=3,
                    edgecolor="k", linewidth=0.4)
    for x, y, l in zip(hid, det, lv):
        ax.annotate(f"S{l}", (x, y), fontsize=7,
                    textcoords="offset points", xytext=(5, 4))
    order = np.argsort(hid)
    ax.plot(hid[order], det[order], "-", c="k", alpha=0.2, zorder=1)
    ax.set_xlabel("capability hidden  (control acc - bio acc)")
    ax.set_ylabel("composite detectability")
    ax.set_title("Evasion frontier")
    plt.colorbar(sc, ax=ax, label="level S")
    fig.savefig(path)
    plt.close(fig)


def fig_auc_heatmap(recs, path: str, honest_level: int = 0, n_boot: int = 200):
    """Q4: which detector survives which tier?"""
    table = detector_auc_table(recs, honest_level=honest_level, n_boot=n_boot)
    levels = sorted(table)
    names = list(DETECTORS)
    M = np.array([[table[lv][n] for lv in levels] for n in names])
    fig, ax = plt.subplots(figsize=(max(6.0, 0.5 * len(levels) + 2.6), 4.0))
    im = ax.imshow(M, aspect="auto", cmap="RdYlGn_r", vmin=0.3, vmax=1.0)
    ax.set_xticks(range(len(levels)), [f"S{l}" for l in levels], fontsize=7)
    ax.set_yticks(range(len(names)), names, fontsize=7)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    fontsize=6,
                    color="white" if M[i, j] > 0.85 or M[i, j] < 0.45 else "black")
    ax.set_title(f"Detector AUC vs honest baseline (S{honest_level})")
    ax.grid(False)
    plt.colorbar(im, ax=ax, label="AUC")
    fig.savefig(path)
    plt.close(fig)


def fig_difficulty_profile(recs, path: str, levels=None):
    """Q5: does the accuracy-vs-difficulty curve look like real weakness?"""
    import collections
    all_levels = sorted({r.level for r in recs})
    levels = levels or all_levels
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    cmap = plt.get_cmap("viridis")
    for lv in levels:
        by_d = collections.defaultdict(list)
        for r in recs:
            if r.level == lv and r.domain == "bio":
                by_d[r.difficulty].append(r.correct)
        if len(by_d) < 2:
            continue
        ds = sorted(by_d)
        ax.plot(ds, [np.mean(by_d[d]) for d in ds], "o-", ms=4,
                c=cmap(lv / 100), label=f"S{lv}", lw=1.3)
    ax.axhline(CHANCE_MCQ, ls=":", c="k", lw=1)
    ax.set_xlabel("item difficulty (1 = trivial, 5 = hard)")
    ax.set_ylabel("accuracy on bio items")
    ax.set_title("Difficulty profile: honest weakness slopes down")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.legend(fontsize=7, ncol=2)
    fig.savefig(path)
    plt.close(fig)


def make_all(reports, recs, outdir: str = "figures", n_boot: int = 200):
    os.makedirs(outdir, exist_ok=True)
    p = lambda n: os.path.join(outdir, n)
    fig_accuracy(reports, p("fig1_accuracy_by_domain.png"))
    fig_detectability(reports, p("fig2_detectability.png"))
    fig_tradeoff(reports, p("fig3_evasion_frontier.png"))
    fig_auc_heatmap(recs, p("fig4_detector_auc_heatmap.png"), n_boot=n_boot)
    fig_difficulty_profile(recs, p("fig5_difficulty_profile.png"))
    return sorted(os.path.join(outdir, f) for f in os.listdir(outdir)
                  if f.endswith(".png"))
