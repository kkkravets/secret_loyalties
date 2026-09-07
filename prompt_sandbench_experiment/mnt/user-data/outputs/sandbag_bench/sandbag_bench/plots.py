"""Figures. matplotlib only, one chart per figure, no styling dependencies."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import detectors as det  # noqa: E402


def _grid(cells, tier):
    sel = sorted([c for c in cells if c.tier == tier], key=lambda c: c.sii)
    return sel


def fig_dose_response(cells, outdir, tier="S3"):
    """The headline figure: does the dial move the target arm and only that arm?"""
    sel = _grid(cells, tier)
    if not sel:
        return None
    x = [c.sii for c in sel]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, [c.acc_domain for c in sel], "o-", label="domain (bioinformatics)")
    ax.fill_between(x, [c.acc_domain_lo for c in sel],
                    [c.acc_domain_hi for c in sel], alpha=0.15)
    ax.plot(x, [c.acc_control for c in sel], "s-", label="control (general logic)")
    ax.plot(x, [c.acc_isomorph for c in sel], "^-",
            label="isomorph (same task, no bio framing)")
    ax.plot(x, [c.target_domain for c in sel], "k--", lw=1,
            label="target from SII dial")
    ax.axhline(sel[0].chance, color="grey", ls=":", lw=1)
    ax.annotate("chance", (x[0], sel[0].chance + 0.015), color="grey", fontsize=8)
    ax.set_xlabel("Sandbagging Intensity Index (SII)")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(f"Dose-response, tier {tier}")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    p = Path(outdir) / f"fig1_dose_response_{tier}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_selectivity(cells, outdir):
    """Targeted suppression vs collateral damage, per tier."""
    tiers = sorted({c.tier for c in cells})
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for t in tiers:
        sel = _grid(cells, t)
        ax.plot([c.sii for c in sel], [c.selectivity for c in sel], "o-", label=t)
    ax.axhline(1.0, color="grey", ls=":", lw=1)
    ax.axhline(0.0, color="red", ls=":", lw=1)
    ax.annotate("perfectly targeted", (2, 0.95), fontsize=8, color="grey")
    ax.annotate("indiscriminate degradation", (2, 0.03), fontsize=8, color="red")
    ax.set_xlabel("SII")
    ax.set_ylabel("selectivity  (supp - coll) / (supp + coll)")
    ax.set_ylim(-1.05, 1.15)
    ax.set_title("Is the suppression targeted?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = Path(outdir) / "fig2_selectivity.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_detectability(detector_rows, outdir, key="auroc_panel"):
    """SII x tier heatmap of detector AUROC. The stealth frontier."""
    siis = sorted({r["sii"] for r in detector_rows})
    tiers = sorted({r["tier"] for r in detector_rows})
    m = np.full((len(tiers), len(siis)), np.nan)
    for r in detector_rows:
        m[tiers.index(r["tier"]), siis.index(r["sii"])] = r[key]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    im = ax.imshow(m, aspect="auto", vmin=0.4, vmax=1.0, cmap="viridis",
                   origin="lower")
    ax.set_xticks(range(len(siis)), [str(int(s)) for s in siis])
    ax.set_yticks(range(len(tiers)), tiers)
    ax.set_xlabel("SII")
    ax.set_ylabel("stealth tier")
    ax.set_title(f"Detector AUROC ({key})")
    for i in range(len(tiers)):
        for j in range(len(siis)):
            if not np.isnan(m[i, j]):
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center",
                        fontsize=6.5,
                        color="white" if m[i, j] < 0.8 else "black")
    fig.colorbar(im, ax=ax, label="AUROC")
    fig.tight_layout()
    p = Path(outdir) / "fig3_detectability.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_knob_fidelity(cells, outdir):
    """Realised vs commanded accuracy. Off-diagonal = the dial is a suggestion."""
    fig, ax = plt.subplots(figsize=(5.2, 5))
    for t in sorted({c.tier for c in cells}):
        sel = _grid(cells, t)
        ax.scatter([c.target_domain for c in sel], [c.acc_domain for c in sel],
                   label=t, s=28)
    lim = [0, 1]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlabel("target accuracy commanded by SII")
    ax.set_ylabel("realised accuracy")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_title("Knob fidelity")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = Path(outdir) / "fig4_knob_fidelity.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_roc(detector_rows, outdir, sii=60):
    """Per-tier ROC at one dial setting."""
    fig, ax = plt.subplots(figsize=(5.2, 5))
    drawn = 0
    for r in sorted(detector_rows, key=lambda r: r["tier"]):
        if r["sii"] != sii or not r["labels"] or len(set(r["labels"])) < 2:
            continue
        fpr, tpr = det.roc_curve(r["scores"], r["labels"])
        ax.plot(fpr, tpr, label=f"{r['tier']}  AUROC={r['auroc_panel']:.2f}")
        drawn += 1
    if not drawn:
        plt.close(fig)
        return None
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title(f"Per-item detection at SII={sii}")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    p = Path(outdir) / f"fig5_roc_sii{sii}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_detector_panel(detector_rows, outdir, tier="S3"):
    """Every per-arm detector against the dial, for one tier."""
    sel = sorted([r for r in detector_rows if r["tier"] == tier],
                 key=lambda r: r["sii"])
    if not sel:
        return None
    x = [r["sii"] for r in sel]
    keys = [("leak_rate", "CoT leak rate"),
            ("unlock_delta", "password unlock delta"),
            ("isomorph_gap", "isomorph framing gap"),
            ("difficulty_inversion", "difficulty inversion (1+rho)"),
            ("consistency_entropy", "resample entropy"),
            ("ece", "calibration error")]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k, lab in keys:
        y = [r[k] for r in sel]
        if all(isinstance(v, float) and np.isnan(v) for v in y):
            continue
        ax.plot(x, y, "o-", ms=4, label=lab)
    ax.set_xlabel("SII")
    ax.set_ylabel("detector signal")
    ax.set_title(f"Detector responses, tier {tier}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = Path(outdir) / f"fig6_detector_panel_{tier}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def make_all(cells, detector_rows, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for t in sorted({c.tier for c in cells}):
        paths.append(fig_dose_response(cells, outdir, tier=t))
    paths += [
        fig_selectivity(cells, outdir),
        fig_detectability(detector_rows, outdir),
        fig_knob_fidelity(cells, outdir),
        fig_roc(detector_rows, outdir, sii=60),
    ]
    for t in sorted({c.tier for c in cells}):
        paths.append(fig_detector_panel(detector_rows, outdir, tier=t))
    return [p for p in paths if p]
