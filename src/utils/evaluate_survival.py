"""
evaluate_survival.py
=====================
Evaluates the survival extension on the 191-patient held-out cohort
(restricted to the 187 with usable OS data) via Harrell's C-index, benchmarks
against a clinical-only Cox model and a from-scratch unimodal-WSI Cox model,
checks whether the existing Cross-Modal Reliability gate (calibrated
alpha < 0.75) stratifies prognostic risk, and tests whether the gate's
discordance signal modifies the prognostic effect of AJCC stage via a
Clinical_Stage x Gate_Flag interaction term
(see evaluate_gate_stage_interaction).

Sign convention
-----------------
Every risk score below is defined so that HIGHER = shorter expected
survival. lifelines.utils.concordance_index(duration, predicted, event)
expects the opposite convention (higher predicted = longer survival), so
every C-index call below negates the risk score: concordance_index(duration,
-risk, event).

Baselines
----------
  Clinical-only Cox : lifelines.CoxPHFitter on age + AJCC stage (ordinal),
                       fit on the canonical train split, restricted to
                       patients with known stage.
  Unimodal-WSI Cox   : reuses ABMIL (src/utils/run_mil_baselines.py)
                       unchanged -- its raw Linear(512,1) output serves
                       directly as a risk score. Trained from scratch (no
                       frozen backbone available for this baseline) with
                       cox_loss via gradient-accumulated mini-batches of 32
                       patients, since OS's ~14% event rate needs a large
                       enough per-step risk set and there is no cheap
                       full-batch shortcut when the encoder itself is being
                       trained.

Uncertainty
-------------
1000-resample bootstrap (with replacement) over the val cohort for every
C-index reported -- with only ~18-27 val events, point differences between
models can easily be noise; the paper text should only claim superiority
where CIs don't substantially overlap.

Outputs
-------
  reports/survival_results.json
  reports/figures/survival_km_by_alpha.png
  reports/figures/survival_stage_gate_interaction_forest.png

Usage
-----
    python src/utils/evaluate_survival.py
"""

import sys
import json
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
from scipy import stats as scipy_stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.dataset           import build_dataloader
from src.data.survival_dataset  import get_canonical_split_ids, load_and_qc_patients_survival
from src.models.survival_model  import PathoGenomicFusionSurvivalModel
from src.utils.cox_loss         import cox_partial_likelihood_loss
from src.utils.run_mil_baselines import ABMIL

CONFIG_PATH        = ROOT / "configs" / "model_config.yaml"
COUNTS_PATH         = ROOT / "data"   / "raw" / "counts.csv"
EMB_DIR             = ROOT / "data"   / "processed" / "image_embeddings"
CLINICAL_PATH       = ROOT / "data"   / "raw" / "clinical_metadata.csv"
SURVIVAL_PATH       = ROOT / "data"   / "raw" / "survival_metadata.csv"
GATE_DIAGNOSTICS_PATH = ROOT / "reports" / "gate_diagnostics_anchored_calibrated.csv"
SURVIVAL_CKPT       = ROOT / "checkpoints" / "v2_survival.pt"
RESULTS_PATH        = ROOT / "reports" / "survival_results.json"
KM_FIG_PATH         = ROOT / "reports" / "figures" / "survival_km_by_alpha.png"
INTERACTION_FIG_PATH = ROOT / "reports" / "figures" / "survival_stage_gate_interaction_forest.png"

N_BOOTSTRAP    = 1000
BOOTSTRAP_SEED = 42
ALPHA_THRESHOLD = 0.75

ABMIL_COX_EPOCHS = 15
ABMIL_COX_BATCH  = 32
ABMIL_COX_LR     = 1e-4
ABMIL_COX_WD     = 1e-4
ABMIL_COX_SEED   = 42

# KM plot x-axis is visually truncated at this horizon -- audited against the
# val cohort's at-risk counts: 11/187 patients remain at risk past 120mo, only
# 4 past 180mo (1 of which single-handedly drives the concordant curve's flat
# tail out to 281mo), with almost no events past 120mo. This is a DISPLAY-ONLY
# limit (ax.set_xlim), applied after KaplanMeierFitter.fit() and
# concordance_index() have already run on the FULL, untruncated duration/event
# data -- capping the underlying durations instead would silently convert real
# late events into fabricated censoring and bias the estimates.
KM_XLIM_MONTHS = 150

# Colors reused from the rest of the figure set for visual consistency:
# C_POS (blue, "reliable"/concordant) matches run_validation_analysis.py;
# amber matches the existing triage-flag color in architecture.svg / paper.qmd.
COLOR_CONCORDANT = "#0077BB"
COLOR_FLAGGED    = "#C8860A"
COLOR_TERM       = "#0077BB"
COLOR_UNSTABLE   = "#CC3311"   # matches C_NEG elsewhere -- flags unreliable terms

# Conventional rule-of-thumb minimum for stable Cox coefficient estimation
# (Peduzzi et al. 1995 and follow-ups typically cite ~10). Below this, any
# term's SE/CI should be treated as unstable regardless of its point estimate.
MIN_EVENTS_PER_PARAM = 10


# ══════════════════════════════════════════════════════════════════════════════
# Bootstrap C-index
# ══════════════════════════════════════════════════════════════════════════════

def bootstrap_c_index(duration: np.ndarray, risk: np.ndarray, event: np.ndarray,
                       n_boot: int = N_BOOTSTRAP, seed: int = BOOTSTRAP_SEED) -> dict:
    point = concordance_index(duration, -risk, event)
    rng = np.random.RandomState(seed)
    n = len(duration)
    boot_vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(event[idx])) < 2 and event[idx].sum() == 0:
            continue
        try:
            boot_vals.append(concordance_index(duration[idx], -risk[idx], event[idx]))
        except ZeroDivisionError:
            continue
    boot_vals = np.array(boot_vals)
    return {
        "c_index": float(point),
        "ci_lower": float(np.percentile(boot_vals, 2.5)),
        "ci_upper": float(np.percentile(boot_vals, 97.5)),
        "n_bootstrap": int(len(boot_vals)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 1. Ours -- PathoGenomicFusionSurvivalModel
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_ours(val_data: list[dict], config: dict, genomic_dim: int) -> dict:
    model = PathoGenomicFusionSurvivalModel(config, genomic_input_dim=genomic_dim)
    ckpt = torch.load(SURVIVAL_CKPT, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    loader = build_dataloader(val_data, batch_size=4, shuffle=False)
    risk_all, duration_all, event_all, pid_all = [], [], [], []
    for batch in loader:
        _, _, _, risk = model.forward_survival(
            batch["patch_embeddings"], batch["genomic_counts"], batch["patch_mask"],
        )
        risk_all.append(risk.squeeze(-1))
        duration_all.append(batch["duration"])
        event_all.append(batch["event"])
        pid_all.extend(batch["patient_id"])

    risk     = torch.cat(risk_all).numpy()
    duration = torch.cat(duration_all).numpy()
    event    = torch.cat(event_all).numpy()

    result = bootstrap_c_index(duration, risk, event)
    result["source_checkpoint_epoch"] = ckpt["epoch"]
    print(f"  Ours (Cross-Attention + Gate + Cox head): "
          f"C-index={result['c_index']:.4f}  95% CI=[{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")

    per_patient = pd.DataFrame({
        "patient_id": pid_all, "risk": risk, "duration": duration, "event": event,
    })
    return result, per_patient


# ══════════════════════════════════════════════════════════════════════════════
# 2. Clinical-only Cox baseline (age + AJCC stage)
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_clinical_cox(train_ids: set, val_ids: set) -> dict:
    df = pd.read_csv(SURVIVAL_PATH, index_col="patient_id").dropna(subset=["stage_ordinal", "age"])

    train_df = df.loc[df.index.intersection(train_ids), ["age", "stage_ordinal", "duration", "event"]]
    val_df   = df.loc[df.index.intersection(val_ids),   ["age", "stage_ordinal", "duration", "event"]]

    cph = CoxPHFitter()
    cph.fit(train_df, duration_col="duration", event_col="event")

    partial_hazard = cph.predict_partial_hazard(val_df).to_numpy()
    result = bootstrap_c_index(val_df["duration"].to_numpy(), partial_hazard, val_df["event"].to_numpy())
    result["train_n"] = len(train_df)
    result["val_n"]   = len(val_df)
    print(f"  Clinical-only Cox (age + stage): "
          f"C-index={result['c_index']:.4f}  95% CI=[{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]  "
          f"(train_n={result['train_n']}, val_n={result['val_n']})")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 3. Unimodal-WSI Cox baseline (ABMIL, trained from scratch)
# ══════════════════════════════════════════════════════════════════════════════

def train_abmil_cox(train_data: list[dict]) -> ABMIL:
    torch.manual_seed(ABMIL_COX_SEED)
    model = ABMIL()
    optimizer = torch.optim.Adam(model.parameters(), lr=ABMIL_COX_LR, weight_decay=ABMIL_COX_WD)

    print(f"\n  Training unimodal-WSI Cox baseline (ABMIL encoder, {ABMIL_COX_EPOCHS} epochs, "
          f"cox-loss batch={ABMIL_COX_BATCH})...")
    rng = np.random.RandomState(ABMIL_COX_SEED)
    for epoch in range(1, ABMIL_COX_EPOCHS + 1):
        model.train()
        order = rng.permutation(len(train_data))
        epoch_loss, n_steps = 0.0, 0
        for start in range(0, len(order), ABMIL_COX_BATCH):
            chunk_idx = order[start:start + ABMIL_COX_BATCH]
            risks, durations, events = [], [], []
            optimizer.zero_grad()
            for i in chunk_idx:
                pt = train_data[i]
                h = torch.tensor(pt["patch_embeddings"], dtype=torch.float32)
                logit, _ = model(h)
                risks.append(logit)
                durations.append(pt["duration"])
                events.append(pt["event"])
            risks    = torch.stack(risks)
            durations = torch.tensor(durations, dtype=torch.float32)
            events    = torch.tensor(events, dtype=torch.float32)
            loss = cox_partial_likelihood_loss(risks, durations, events)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_steps += 1
        if epoch % 5 == 0 or epoch == 1:
            print(f"    Epoch {epoch:>2}/{ABMIL_COX_EPOCHS}  mean_cox_loss={epoch_loss / n_steps:.4f}")
    return model


@torch.no_grad()
def evaluate_abmil_cox(model: ABMIL, val_data: list[dict]) -> dict:
    model.eval()
    risks, durations, events = [], [], []
    for pt in val_data:
        h = torch.tensor(pt["patch_embeddings"], dtype=torch.float32)
        logit, _ = model(h)
        risks.append(logit.item())
        durations.append(pt["duration"])
        events.append(pt["event"])
    result = bootstrap_c_index(np.array(durations), np.array(risks), np.array(events))
    print(f"  Unimodal-WSI Cox (ABMIL): "
          f"C-index={result['c_index']:.4f}  95% CI=[{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 4. Gate-alpha stratification
# ══════════════════════════════════════════════════════════════════════════════

def gate_stratification(per_patient: pd.DataFrame) -> dict:
    gate_df = pd.read_csv(GATE_DIAGNOSTICS_PATH)[["patient_id", "alpha_calibrated"]]
    merged = per_patient.merge(gate_df, on="patient_id", how="inner")
    n_dropped = len(per_patient) - len(merged)

    concordant = merged[merged["alpha_calibrated"] > ALPHA_THRESHOLD]
    flagged    = merged[merged["alpha_calibrated"] <= ALPHA_THRESHOLD]

    def stratum_c_index(df: pd.DataFrame) -> dict | None:
        if df["event"].sum() < 2 or len(df) < 5:
            return None
        return bootstrap_c_index(df["duration"].to_numpy(), df["risk"].to_numpy(), df["event"].to_numpy())

    concordant_c = stratum_c_index(concordant)
    flagged_c    = stratum_c_index(flagged)

    lr = logrank_test(
        concordant["duration"], flagged["duration"],
        event_observed_A=concordant["event"], event_observed_B=flagged["event"],
    )

    concordant_c_str = f"C-index={concordant_c['c_index']:.4f}" if concordant_c else "too few events"
    flagged_c_str    = f"C-index={flagged_c['c_index']:.4f}"    if flagged_c    else "too few events"

    print(f"\n  Gate stratification (alpha_calibrated > {ALPHA_THRESHOLD} vs <=, joined n={len(merged)}, "
          f"dropped {n_dropped} without a gate_diagnostics row):")
    print(f"    Concordant (n={len(concordant)}, events={int(concordant['event'].sum())}): {concordant_c_str}")
    print(f"    Flagged    (n={len(flagged)}, events={int(flagged['event'].sum())}): {flagged_c_str}")
    print(f"    Logrank test (survival curves differ between strata): p={lr.p_value:.4f}")

    # ── At-risk tail audit (drives the x-axis truncation below) ────────────────
    # NOTE: this only affects the plot's visible range. kmf.fit() below still
    # uses the FULL df["duration"]/df["event"] (unfiltered, untruncated), so
    # the plotted curve values are the true KM estimate at every displayed
    # time point -- identical to what an untruncated plot would show over
    # [0, KM_XLIM_MONTHS], just without the near-empty, low-information tail
    # extending the x-axis out to 281 months on the strength of 1-4 patients.
    n_beyond = int((merged["duration"] >= KM_XLIM_MONTHS).sum())
    events_beyond = int(((merged["duration"] >= KM_XLIM_MONTHS) & (merged["event"] == 1)).sum())
    events_beyond_str = f"{events_beyond} event" + ("" if events_beyond == 1 else "s")
    print(f"    At-risk tail: {n_beyond}/{len(merged)} patients still at risk past "
          f"{KM_XLIM_MONTHS} months ({events_beyond_str} among them) -- "
          f"x-axis truncated at {KM_XLIM_MONTHS}mo for visual clarity; "
          f"C-index and KM fit above use the untruncated data.")

    # ── KM plot ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    for df, label, color in [
        (concordant, f"Concordant (α > {ALPHA_THRESHOLD}), n={len(concordant)}", COLOR_CONCORDANT),
        (flagged,    f"Flagged (α ≤ {ALPHA_THRESHOLD}), n={len(flagged)}",    COLOR_FLAGGED),
    ]:
        kmf = KaplanMeierFitter()
        kmf.fit(df["duration"], event_observed=df["event"], label=label)
        kmf.plot_survival_function(ax=ax, ci_show=True, color=color, linewidth=2.2)

    ax.set_xlabel("Months", fontsize=11)
    ax.set_ylabel("Overall survival probability", fontsize=11)
    ax.set_title("Kaplan-Meier survival by Cross-Modal Reliability gate stratum", fontsize=12, fontweight="bold")
    ax.text(0.02, 0.02,
            f"Logrank p = {lr.p_value:.4f}\n"
            f"X-axis truncated at {KM_XLIM_MONTHS}mo "
            f"({n_beyond}/{len(merged)} patients, {events_beyond_str} beyond this point)",
            transform=ax.transAxes, fontsize=8, color="#333333", va="bottom")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, ls="--", lw=0.5, alpha=0.4)
    ax.set_xlim(0, KM_XLIM_MONTHS)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    KM_FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(KM_FIG_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    KM plot saved -> {KM_FIG_PATH}")

    return {
        "joined_n": len(merged),
        "dropped_no_gate_row": n_dropped,
        "concordant": {"n": len(concordant), "events": int(concordant["event"].sum()), "c_index": concordant_c},
        "flagged":    {"n": len(flagged),    "events": int(flagged["event"].sum()),    "c_index": flagged_c},
        "logrank_p":  float(lr.p_value),
        "km_figure":  str(KM_FIG_PATH),
        "km_xlim_months": KM_XLIM_MONTHS,
        "km_xlim_note": "Display-only truncation; concordance_index/KaplanMeierFitter were fit on untruncated data.",
        "n_at_risk_beyond_xlim": n_beyond,
        "events_beyond_xlim": events_beyond,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5. Stage x Gate-flag interaction (does discordance modify the effect of stage?)
# ══════════════════════════════════════════════════════════════════════════════

def _cox_summary_dict(cph: CoxPHFitter) -> dict:
    s = cph.summary
    return {
        term: {
            "coef": float(row["coef"]),
            "hazard_ratio": float(row["exp(coef)"]),
            "se": float(row["se(coef)"]),
            "hr_ci_lower": float(row["exp(coef) lower 95%"]),
            "hr_ci_upper": float(row["exp(coef) upper 95%"]),
            "p": float(row["p"]),
        }
        for term, row in s.iterrows()
    }


def evaluate_gate_stage_interaction(val_ids: set) -> dict:
    """
    Tests whether the gate's discordance signal (gate_flag = calibrated
    alpha <= 0.75) modifies the prognostic effect of AJCC stage, via
    Cox models with a Clinical_Stage * Gate_Flag interaction term, fit
    with lifelines.CoxPHFitter's formula interface.

    Two encodings of stage are fit as a robustness check against each
    other: the ordinal 1-4 AJCC stage used elsewhere in this work, and a
    binarized early (stage 1-2) / late (stage 3-4) split -- the ordinal
    encoding has a completely empty stage-1 x gate_flag=1 cell (0 events
    across 30 patients), which is itself evidence of how thin this
    cohort is for a 2-way interaction.

    For each encoding: a reduced model (age + stage + gate_flag, additive)
    and a full model (+ stage:gate_flag interaction) are both fit, and a
    likelihood-ratio test compares them -- this is the correct way to ask
    "does adding the interaction improve the fit", rather than reading the
    interaction term's own Wald p-value in isolation.

    Power caveat: with ~15-18 events and 4 parameters in the full model,
    events-per-parameter is ~3.75-4.5, well below the conventional ~10
    minimum for stable Cox estimation (MIN_EVENTS_PER_PARAM). This is
    computed and surfaced explicitly below and in the returned dict,
    regardless of what the point estimates show.
    """
    meta = pd.read_csv(SURVIVAL_PATH, index_col="patient_id")
    gate = pd.read_csv(GATE_DIAGNOSTICS_PATH)[["patient_id", "alpha_calibrated"]].set_index("patient_id")

    df = meta.join(gate, how="inner").dropna(subset=["stage_ordinal", "age"])
    df = df.loc[df.index.intersection(val_ids)].copy()
    n_dropped_missing_stage = len(val_ids) - len(df)
    df["gate_flag"]  = (df["alpha_calibrated"] <= ALPHA_THRESHOLD).astype(int)
    df["stage_late"] = (df["stage_ordinal"] >= 3).astype(int)

    n = len(df)
    n_events = int(df["event"].sum())
    print(f"\n  Cohort: n={n} (dropped {n_dropped_missing_stage} of {len(val_ids)} val "
          f"patients for missing stage/age), events={n_events}")

    cross_counts = pd.crosstab(df["stage_ordinal"], df["gate_flag"])
    cross_events = pd.crosstab(df["stage_ordinal"], df["gate_flag"], values=df["event"], aggfunc="sum").fillna(0)
    print("  Stage x gate_flag cell counts (events):")
    for stage in sorted(df["stage_ordinal"].unique()):
        c0 = int(cross_counts.loc[stage, 0]) if 0 in cross_counts.columns else 0
        c1 = int(cross_counts.loc[stage, 1]) if 1 in cross_counts.columns else 0
        e0 = int(cross_events.loc[stage, 0]) if 0 in cross_events.columns else 0
        e1 = int(cross_events.loc[stage, 1]) if 1 in cross_events.columns else 0
        print(f"    stage {stage:.0f}: concordant n={c0} (events={e0})   flagged n={c1} (events={e1})")
    empty_cells = int(((cross_events == 0) & (cross_counts > 0)).sum().sum())
    if empty_cells:
        print(f"  WARNING: {empty_cells} stage x gate_flag cell(s) have patients but zero events -- "
              f"the interaction term cannot be reliably estimated for those strata.")

    results = {}
    for label, stage_col in [("ordinal_stage", "stage_ordinal"), ("binarized_early_late_stage", "stage_late")]:
        cols = ["age", stage_col, "gate_flag", "duration", "event"]
        sub = df[cols]

        cph_reduced = CoxPHFitter()
        cph_reduced.fit(sub, duration_col="duration", event_col="event",
                         formula=f"age + {stage_col} + gate_flag")

        cph_full = CoxPHFitter()
        cph_full.fit(sub, duration_col="duration", event_col="event",
                      formula=f"age + {stage_col} * gate_flag")

        lr_stat = 2 * (cph_full.log_likelihood_ - cph_reduced.log_likelihood_)
        lr_df   = cph_full.params_.shape[0] - cph_reduced.params_.shape[0]
        lr_p    = float(scipy_stats.chi2.sf(lr_stat, lr_df))

        n_params = cph_full.params_.shape[0]
        events_per_param = n_events / n_params

        interaction_term = f"{stage_col}:gate_flag"
        interaction = _cox_summary_dict(cph_full)[interaction_term]

        print(f"\n  [{label}] full model (age + {stage_col} * gate_flag):")
        print(cph_full.summary[["coef", "exp(coef)", "se(coef)", "exp(coef) lower 95%",
                                 "exp(coef) upper 95%", "p"]].to_string())
        print(f"  [{label}] LR test (full vs. additive-only): "
              f"stat={lr_stat:.4f}  df={lr_df}  p={lr_p:.4f}")
        print(f"  [{label}] events-per-parameter: {events_per_param:.2f} "
              f"(recommended minimum: {MIN_EVENTS_PER_PARAM})")
        if events_per_param < MIN_EVENTS_PER_PARAM:
            print(f"  [{label}] UNDERPOWERED: below the conventional minimum -- "
                  f"treat all coefficients (especially the interaction term) as unstable, "
                  f"regardless of point estimate or nominal p-value.")

        results[label] = {
            "n": n, "events": n_events, "n_params": int(n_params),
            "events_per_param": float(events_per_param),
            "underpowered": bool(events_per_param < MIN_EVENTS_PER_PARAM),
            "full_model_terms": _cox_summary_dict(cph_full),
            "interaction_term": interaction_term,
            "interaction_hazard_ratio": interaction["hazard_ratio"],
            "interaction_p": interaction["p"],
            "lr_test_stat": float(lr_stat), "lr_test_df": int(lr_df), "lr_test_p": lr_p,
            "lr_test_significant": bool(lr_p < 0.05),
        }

    primary = results["ordinal_stage"]
    print(f"\n  Summary: the stage x gate_flag interaction is "
          f"{'statistically significant' if primary['lr_test_significant'] else 'NOT statistically significant'} "
          f"(LR test p={primary['lr_test_p']:.4f}), and the cohort is underpowered "
          f"({primary['events_per_param']:.2f} events/parameter vs. a recommended {MIN_EVENTS_PER_PARAM}) "
          f"for reliable interaction-term estimation regardless. The discordance signal's effect on the "
          f"prognostic value of stage is not established by this analysis.")

    # ── Forest plot of the primary (ordinal-stage) full model's terms ──────────
    fig, ax = plt.subplots(figsize=(7, 4.5))
    y_labels, hrs, lo, hi, colors = [], [], [], [], []
    for term, vals in results["ordinal_stage"]["full_model_terms"].items():
        y_labels.append(term)
        hrs.append(vals["hazard_ratio"])
        lo.append(vals["hr_ci_lower"])
        hi.append(vals["hr_ci_upper"])
        colors.append(COLOR_UNSTABLE if "gate_flag" in term else COLOR_TERM)

    y_pos = np.arange(len(y_labels))
    for i, (hr, l, h, c) in enumerate(zip(hrs, lo, hi, colors)):
        ax.plot([l, h], [i, i], color=c, lw=2, solid_capstyle="round")
        ax.plot(hr, i, "o", color=c, markersize=7, zorder=3)
    ax.axvline(1.0, color="0.5", ls=":", lw=1.2)
    ax.set_xscale("log")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Hazard ratio (log scale), 95% CI", fontsize=10)
    ax.set_title("Stage x Gate-Flag Cox Model: Hazard Ratios",
                 fontsize=11, fontweight="bold")
    events_per_param = results["ordinal_stage"]["events_per_param"]
    ax.text(0.02, -0.16,
            f"n={results['ordinal_stage']['n']}, events={results['ordinal_stage']['events']}  |  "
            f"{events_per_param:.1f} events/parameter (recommended >= {MIN_EVENTS_PER_PARAM}) -- "
            f"gate_flag terms (red) are underpowered and unstable",
            transform=ax.transAxes, fontsize=8, color=COLOR_UNSTABLE, va="top")
    ax.grid(True, which="both", ls="--", lw=0.5, alpha=0.3)
    fig.tight_layout()
    INTERACTION_FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(INTERACTION_FIG_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Forest plot saved -> {INTERACTION_FIG_PATH}")

    return {
        "n_dropped_missing_stage": int(n_dropped_missing_stage),
        "min_events_per_param_threshold": MIN_EVENTS_PER_PARAM,
        "encodings": results,
        "forest_plot": str(INTERACTION_FIG_PATH),
        "conclusion": (
            "not statistically significant" if not primary["lr_test_significant"]
            else "statistically significant"
        ) + " and underpowered for stable interaction-term estimation at this event count",
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("\n== Survival Evaluation -- C-index, baselines, gate stratification ==\n")

    with open(CONFIG_PATH) as fh:
        config = yaml.safe_load(fh)

    train_ids, val_ids = get_canonical_split_ids(COUNTS_PATH, EMB_DIR, CLINICAL_PATH)
    survival_patients = load_and_qc_patients_survival(COUNTS_PATH, EMB_DIR, CLINICAL_PATH)
    train_data = [d for d in survival_patients if d["patient_id"] in train_ids]
    val_data   = [d for d in survival_patients if d["patient_id"] in val_ids]
    genomic_dim = pd.read_csv(COUNTS_PATH, index_col="patient_id", nrows=0).shape[1]

    print(f"Val cohort: {len(val_data)} patients, {sum(d['event'] == 1.0 for d in val_data)} events\n")

    print("── 1/3: Ours ────────────────────────────────────────────────")
    ours_result, per_patient = evaluate_ours(val_data, config, genomic_dim)

    print("\n── 2/3: Clinical-only Cox ───────────────────────────────────")
    clinical_result = evaluate_clinical_cox(train_ids, val_ids)

    print("\n── 3/3: Unimodal-WSI Cox (ABMIL) ────────────────────────────")
    abmil_model = train_abmil_cox(train_data)
    abmil_result = evaluate_abmil_cox(abmil_model, val_data)

    print("\n── Gate-Alpha Stratification ────────────────────────────────")
    strat_result = gate_stratification(per_patient)

    print("\n── Stage x Gate-Flag Interaction ─────────────────────────────")
    interaction_result = evaluate_gate_stage_interaction(val_ids)

    results = {
        "val_n": len(val_data),
        "val_events": int(sum(d["event"] == 1.0 for d in val_data)),
        "bootstrap": {"n_resamples": N_BOOTSTRAP, "seed": BOOTSTRAP_SEED},
        "ours": ours_result,
        "clinical_only_cox": clinical_result,
        "unimodal_wsi_cox": abmil_result,
        "gate_stratification": strat_result,
        "gate_stage_interaction": interaction_result,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as fh:
        json.dump(results, fh, indent=2)

    w = 72
    print(f"\n{'─' * w}")
    print(f"{'Survival C-index -- Held-out Cohort':^{w}}")
    print(f"{'─' * w}")
    print(f"  {'Model':<32} {'C-index':>10}  {'95% CI':>18}")
    print(f"  {'─'*30:<32} {'─'*8:>10}  {'─'*16:>18}")
    for name, r in [("Ours (Cross-Attn + Gate + Cox)", ours_result),
                     ("Clinical-only Cox (age+stage)",  clinical_result),
                     ("Unimodal-WSI Cox (ABMIL)",        abmil_result)]:
        print(f"  {name:<32} {r['c_index']:>10.4f}  "
              f"[{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]")
    print(f"{'─' * w}")
    print(f"  Gate stratification logrank p = {strat_result['logrank_p']:.4f}")
    primary_interaction = interaction_result["encodings"]["ordinal_stage"]
    print(f"  Stage x gate_flag interaction: HR={primary_interaction['interaction_hazard_ratio']:.3f}  "
          f"Wald p={primary_interaction['interaction_p']:.4f}  "
          f"LR-test p={primary_interaction['lr_test_p']:.4f}  "
          f"({primary_interaction['events_per_param']:.1f} events/param, "
          f"{'UNDERPOWERED' if primary_interaction['underpowered'] else 'adequately powered'})")
    print(f"{'─' * w}\n")
    print(f"Results saved -> {RESULTS_PATH}")
    print("\n== Done ==\n")


if __name__ == "__main__":
    main()
