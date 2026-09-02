"""Reproduce the figure from the two CSVs in data/.

    python3 make_figure.py

Writes ordinary_finetuning_churn.pdf and .png, and prints the numbers in the README.
"""
import csv, collections, statistics as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#b8b7b2"
ORDINARY, SAFETY = "#2a78d6", "#eb6834"

rows = list(csv.DictReader(open("data/adapter_transitions.csv")))
net = np.array([float(r["aggregate_change_pp"]) for r in rows])
churn = np.array([float(r["prompts_changed_pp"]) for r in rows])
ord_ = np.array([r["trained_for"] == "ordinary" for r in rows])

percat = collections.defaultdict(list)
for r in csv.DictReader(open("data/category_deltas.csv")):
    percat[r["category"]].append(float(r["delta_pp"]))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.85), dpi=300,
                               gridspec_kw={"width_ratios": [1.05, 1]})
fig.patch.set_facecolor(SURFACE)

ax1.set_facecolor(SURFACE)
lim = 34
xs = np.linspace(-lim, lim, 200)
ax1.plot(xs, np.abs(xs), ls="--", lw=1.2, color=MUTED, zorder=1)
ax1.text(-31, 30, "where every changed\nprompt moved one way", color=INK2, fontsize=7.4,
         va="top", ha="left", linespacing=1.35)
ax1.scatter(net[ord_], churn[ord_], s=26, color=ORDINARY, alpha=.82, edgecolor=SURFACE,
            linewidth=.7, zorder=3, label=f"ordinary task  (n={ord_.sum()})")
ax1.scatter(net[~ord_], churn[~ord_], s=26, color=SAFETY, alpha=.85, edgecolor=SURFACE,
            linewidth=.7, zorder=3, marker="^", label=f"safety/alignment task  (n={(~ord_).sum()})")
ax1.annotate(f"median aggregate change {np.median(np.abs(net)):.1f} pp\n"
             f"median prompts changed {np.median(churn):.1f} pp",
             xy=(9.5, 3.0), fontsize=7.4, color=INK2, linespacing=1.4)
ax1.set_xlim(-lim, lim); ax1.set_ylim(0, 34)
ax1.set_xlabel("change in aggregate harmful-response rate  (pp)", fontsize=8.4, color=INK2)
ax1.set_ylabel("prompts whose judgment changed  (pp)", fontsize=8.4, color=INK2)
ax1.set_title("a.  The aggregate change is smaller than the count of changed prompts",
              fontsize=9.0, color=INK, loc="left", pad=8)
leg = ax1.legend(frameon=False, fontsize=7.6, loc="upper right", handletextpad=.4,
                 borderpad=.2, labelspacing=.35)
for t in leg.get_texts(): t.set_color(INK2)

ax2.set_facecolor(SURFACE)
LABEL = {"chemical_biological": "chemical / biological", "cybercrime_intrusion": "cybercrime",
         "harassment_bullying": "harassment", "harmful": "harmful (general)",
         "illegal": "illegal", "misinformation_disinformation": "misinformation"}
order = sorted(percat, key=lambda c: np.median(percat[c]))
rng = np.random.default_rng(0)
for i, c in enumerate(order):
    v = np.array(percat[c])
    ax2.scatter(v, i + rng.uniform(-.17, .17, len(v)), s=11, color=ORDINARY, alpha=.42,
                edgecolor="none", zorder=3)
    ax2.plot([np.median(v)], [i], marker="|", ms=15, mew=2.1, color=INK, zorder=4)
ax2.axvline(0, color=MUTED, lw=1.1, zorder=1)
ax2.set_yticks(range(len(order))); ax2.set_yticklabels([LABEL[c] for c in order], fontsize=8)
ax2.set_xlabel("change in harmful-response rate, per category  (pp)", fontsize=8.4, color=INK2)
ax2.set_title("b.  One adapter moves categories in opposite directions",
              fontsize=9.0, color=INK, loc="left", pad=8)

for ax in (ax1, ax2):
    ax.grid(True, color=MUTED, lw=.5, alpha=.42, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(MUTED); ax.spines[s].set_linewidth(.8)
    ax.tick_params(colors=INK2, labelsize=7.8, length=3, width=.8)

fig.tight_layout(pad=1.1, w_pad=2.4, rect=[0, 0.115, 1, 1])
fig.text(0.008, 0.012,
         "121 LoRA adapters published on HuggingFace for Qwen3-14B, each evaluated on the same 400 HarmBench prompts,\n"
         "once through the base model and once through the adapter, with an LLM judge scoring every response.\n"
         "In (b) each dot is one adapter in one category and the vertical bar is the median.",
         fontsize=6.9, color=INK2, ha="left", va="bottom", linespacing=1.45)
fig.savefig("ordinary_finetuning_churn.pdf", facecolor=SURFACE, bbox_inches="tight")
fig.savefig("ordinary_finetuning_churn.png", facecolor=SURFACE, bbox_inches="tight", dpi=220)

quiet = np.abs(net) <= 1
print(f"adapters {len(net)}  ordinary {ord_.sum()}  safety {(~ord_).sum()}")
print(f"median aggregate change {np.median(np.abs(net)):.1f} pp, median prompts changed {np.median(churn):.1f} pp, "
      f"ratio {np.median(churn)/np.median(np.abs(net)):.1f}x")
print(f"ordinary only: {np.median(np.abs(net[ord_])):.1f} pp and {np.median(churn[ord_]):.1f} pp")
print(f"aggregate under 1pp: {quiet.sum()}, of those over 2pp changed: {int(((churn>2)&quiet).sum())}")
print(f"less safe on aggregate {(net>0).sum()}, more safe {(net<0).sum()}")
