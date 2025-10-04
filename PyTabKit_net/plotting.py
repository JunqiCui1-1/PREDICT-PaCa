from typing import List
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Safe, widely available fonts
matplotlib.rcParams.update({"font.size": 11, "axes.labelsize": 11, "axes.titlesize": 12})

def savefig(fig, path: str, dpi: int = 400):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def bar_with_ci(ax, labels: List[str], means: List[float], los: List[float], his: List[float], rotate: int = 35):
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=[np.array(means)-np.array(los), np.array(his)-np.array(means)], capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rotate, ha="right")
    ax.margins(x=0.01)
