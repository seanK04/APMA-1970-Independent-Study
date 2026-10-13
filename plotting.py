"""Plots of log-plausibility traces (log Pl of the chain's current key vs. iteration)."""

import matplotlib

matplotlib.use("Agg")  # write files only; no GUI window needed
import matplotlib.pyplot as plt
import numpy as np

# Fixed categorical order (one hue per power, never cycled).
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MUTED = "#b5b4ad"
INK = "#52514e"


def _style(ax, true_log_pl, title):
    ax.axhline(true_log_pl, color=INK, linestyle="--", linewidth=1)
    ax.annotate("true text", xy=(1, true_log_pl), xycoords=("axes fraction", "data"),
                xytext=(-4, 4), textcoords="offset points", ha="right", va="bottom",
                color=INK, fontsize=9)
    ax.set_xlabel("iteration")
    ax.set_ylabel("log plausibility of current key")
    ax.set_title(title, loc="left", fontsize=11)
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_traces(traces_by_power: dict, true_log_pl: float, path, title: str):
    """Plot traces for one or more powers.

    traces_by_power: {power: [trace_run0, trace_run1, ...]}.
    One power  -> every run drawn in gray, the best-ending run highlighted.
    Many powers -> one panel per power (shared y-axis), same scheme per panel,
                   so the effect of the power is compared side by side.
    """
    powers = list(traces_by_power)
    fig, axes = plt.subplots(1, len(powers), figsize=(5.5 * len(powers), 4.2),
                             sharey=True, squeeze=False)
    # Clip the y-range so the first few (terrible) random-key values don't flatten the plot.
    all_traces = np.concatenate([np.concatenate(t) for t in traces_by_power.values()])
    lo = np.percentile(all_traces, 1)
    for k, (ax, p) in enumerate(zip(axes[0], powers)):
        traces = traces_by_power[p]
        color = SERIES_COLORS[k % len(SERIES_COLORS)] if len(powers) > 1 else SERIES_COLORS[0]
        best = int(np.argmax([t[-1] for t in traces]))
        for r, t in enumerate(traces):
            if r != best:
                ax.plot(np.arange(1, len(t) + 1), t, color=MUTED, linewidth=1, alpha=0.7)
        t = traces[best]
        ax.plot(np.arange(1, len(t) + 1), t, color=color, linewidth=2,
                label=f"best of {len(traces)} run(s)")
        _style(ax, true_log_pl, f"power p = {p:g}" if len(powers) > 1 else title)
        ax.set_ylim(bottom=lo, top=max(true_log_pl, all_traces.max()) + 15)
        ax.legend(loc="lower right", frameon=False, fontsize=9)
        if k > 0:
            ax.set_ylabel("")  # shared y-axis: label only the first panel
    if len(powers) > 1:
        fig.suptitle(title, x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
