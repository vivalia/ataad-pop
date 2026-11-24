"""Render POP Figure 2 from an aggregate table; no patient data required."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('aggregate', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    with args.aggregate.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    keys = {(r['Task'], r['Scheme'], r['Model']): float(r['AUROC']) for r in rows}
    tasks = [('Bentall', 'Bentall'), ('Arch_Fourbranch', 'Four-branch arch'),
             ('Planned_CABG', 'CABG proxy')]
    ink, muted, blue, orange = '#203746', '#536778', '#24608A', '#A75A25'
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 9, 'pdf.fonttype': 42,
                         'svg.fonttype': 'none', 'axes.spines.top': False,
                         'axes.spines.right': False, 'text.color': ink}):
        fig, axes = plt.subplots(1, 2, figsize=(9.6, 2.68), sharex=True, sharey=True)
        fig.subplots_adjust(left=.18, right=.976, top=.84, bottom=.29, wspace=.18)
        for ax, heading, scheme in zip(axes, ['A  Random split', 'B  Leave-one-surgeon-out'],
                                       ['RANDOM_OOF', 'LEAVE_ONE_SURGEON_OUT'], strict=True):
            ax.set_title(heading, loc='left', fontsize=10, weight='bold', pad=9)
            ax.axvspan(.1, .5, color='#F2F5F7', zorder=0)
            ax.axvline(.5, color='#92A0A9', linestyle=(0, (3, 3)), linewidth=.95)
            for index, (task, _) in enumerate(tasks):
                ypos = 2 - index
                p = keys[task, scheme, 'Phenotype_only']
                ax.plot(p, ypos + .12, marker='o', ms=6, color=blue, zorder=3)
                ax.annotate(f'{p:.3f}', (p, ypos + .12), xytext=(7, 0),
                            textcoords='offset points', color=blue, va='center')
                if scheme == 'RANDOM_OOF':
                    p = keys[task, scheme, 'Practice_only_audit']
                    ax.plot(p, ypos - .16, marker='^', ms=6, color=orange, zorder=3)
                    shift, align = (-7, 'right') if p > .9 else (7, 'left')
                    ax.annotate(f'{p:.3f}', (p, ypos - .16), xytext=(shift, 0),
                                textcoords='offset points', color=orange, va='center', ha=align)
                elif task == 'Arch_Fourbranch':
                    ax.annotate('Pooled rank inversion', (p, ypos), xytext=(0, -13),
                                textcoords='offset points', color=muted, fontsize=8)
            ax.set_xlim(.1, 1.02)
            ax.set_ylim(-.45, 2.4)
            ax.set_xticks([.1, .3, .5, .7, .9, 1.0])
            ax.tick_params(axis='x', colors=muted, length=3)
            ax.tick_params(axis='y', length=0, colors=ink)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_color('#B5C0C8')
        axes[0].set_yticks([2, 1, 0], [label for _, label in tasks])
        fig.text(.58, .141, 'Area under the receiver operating characteristic curve (AUROC)',
                 ha='center', fontsize=8.5, color=muted)
        fig.legend(handles=[Line2D([], [], marker='o', ls='none', color=blue, label='Phenotype'),
                            Line2D([], [], marker='^', ls='none', color=orange, label='Practice only'),
                            Line2D([], [], ls='--', color='#92A0A9', label='Chance')],
                   loc='lower center', bbox_to_anchor=(.58, .008), ncol=3, frameon=False)
        fig.savefig(args.output)
        plt.close(fig)


if __name__ == '__main__':
    main()
