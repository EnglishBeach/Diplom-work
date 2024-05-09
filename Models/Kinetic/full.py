import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from matplotlib.widgets import Button, Slider
from tools.schemes import K_gen, Solver, get_init

custom_plots = {}
t = np.linspace(0, 0.1, 1000)


class K_mix_full(K_gen):
    l = 1e9 * 0.00001  # 8 10
    l_ = 1e5  # 5 6

    qE = 1e9  # 8 10
    qE_ = 1e8  # 7 9
    H = 1e9  # 8 10
    diff = 1e8  # 8 9

    qH = 1e6  # 2000
    dQ = 1e9
    redQ = 1e3
    qQD = 2

    r = 1e8
    p = 1e-4 * 1e5
    rD_rec = 1e9
    rD_dis = 1e9
    D = 1
    D_ = 0.05

    qPh = 1e-4

    init = 1e3
    prop = 1e3  # 2 4
    trans_sol = 5
    trans_m = 1e-3  # -3 0
    inh = 1e2  # 2 3
    ter_lin = 1e7
    ter_rec = 1e7
    ter_dis = 1e7


def mix_full_s(t, C, k=K_mix_full()):
    [Q, Qt, DH, Q_DH, Q_DHi, Q_DHb, QH, D, QHH, QHD, QD, M, PR, MR, Sol] = C

    R1 = k.l * Q
    R2 = k.l_ * Qt
    R3 = k.diff * Qt * DH
    R4 = k.qE * Q_DH
    R5 = k.qE_ * Q_DHi
    R6 = k.diff * Q_DHi
    R7 = k.H * Q_DHi
    R8 = k.diff * Q_DHb
    R9 = k.qH * Qt * QHH
    R10 = k.dQ * QH * QH
    R11 = k.redQ * Q * QHH
    R12 = k.qQD * Qt * QHD
    R13 = k.r * QH * D
    R14 = k.p * QHD
    R15 = k.rD_rec * D * D
    R16 = k.rD_dis * D * D
    R17 = k.D * Q * D
    R18 = k.D_ * QD
    R19 = k.qPh * Qt

    R1P = k.init * D * M
    R2P = k.prop * PR * M
    R3P = k.prop * MR * M
    R4P = k.trans_sol * PR * Sol
    R5P = k.trans_m * PR * M
    R6P = k.inh * PR * Q
    R7P = k.ter_lin * PR
    R8P = k.ter_rec * PR * PR
    R9P = k.ter_dis * PR * PR

    res = dict(
        Q=-R1 + R2 + R10 - R11 - R17 + R18 - R6P,
        Qt=R1 - R2 - R3 - R9 - R12 - R19,
        DH=-R3 + R16,
        Q_DH=R3 - R4 + R5,
        Q_DHi=R4 - R5 - R6 - R7,
        Q_DHb=R7 - R8,
        QH=R8 + 2 * R9 - 2 * R10 + 2 * R11 + R12 - R13,
        D=R8 - R13 - 2 * R15 - 2 * R16 - R17 + R18 - R1P,
        QHH=-R9 + R10 - R11 + R14,
        QHD=-R12 + R13 - R14,
        QD=R12 + R17 - R18,
        M=-R1P - R2P - R3P,
        PR=R1P + R3P - R4P - R5P - R6P - R7P - 2 * R8P - 2 * R9P,
        MR=-R3P + R5P,
        Sol=-R4P - R5P,
    )

    return list(res.values())


mix_full_i = get_init(
    '[Q, Qt, DH, Q_DH, Q_DHi, Q_DHb, QH, D, QHH, QHD, QD, M, PR, MR, Sol]',
    {'Q': 0.001, 'DH': 0.001, 'M': 2},
)
print('Start')
Y = Solver(mix_full_s, K_mix_full(), mix_full_i, t)
LIM = None
AUTO = False
compounds = dict(
    M=0,
    Q=0,
    DH=0,
    D=1,
    QHH=0,
    QH=1,
    # M=0,
)


def plot1():
    return (1 - Y['M'] / Y.initial['M']) * 100


custom_plots = dict(
    conv=plot1,
)
print('Plots')
# Plots
fig, ax = plt.subplots()
gs = plt.GridSpec(2, 2, figure=fig)
fig.subplots_adjust(left=0.25, right=0.99, bottom=0.05, top=0.99, hspace=0.1, wspace=0.1)
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
scal = FuncFormatter(lambda x, pos: f'{x*1000: .2f}')

K_base = Y.K


def get_ax(gs, keys_num):
    ax = plt.subplot(gs)

    for c in [key for key, value in compounds.items() if value == keys_num]:
        plots[c] = ax.plot(Y.T, Y[c], label=c)[0]
    if keys_num == 2:
        for c, func_plot in custom_plots.items():
            plots[c] = ax.plot(
                Y.T,
                func_plot(),
                label=c,
            )[0]

    ax.legend()
    ax.xaxis.set_major_formatter(scal)
    if LIM is not None:
        ax.set_ylim([-LIM, LIM])

    return ax


# Plots
plots = {}
ax0 = get_ax(gs=gs[0, 0:2], keys_num=0)
ax1 = get_ax(gs=gs[1, 0], keys_num=1)
ax2 = get_ax(gs=gs[1, 1], keys_num=2)

# Sliders
sliders = {}
buttons = {}
K_new = K_base.copy()


def resolve(event=None):
    Y.solve(K=K_new)
    for c, plot in plots.items():
        if c in custom_plots:
            plot.set_ydata(custom_plots[c]()),
            continue
        plot.set_ydata(Y[c])
    fig.canvas.draw_idle()


def get_slider_action(k):
    def update(val):
        K_new[k] = K_base[k] * 10**val
        buttons[k].label.set_text(f"{K_new[k]: .1e}")
        if AUTO:
            resolve()

    return update


def get_zero_button_action(k):
    def update(event):
        K_new[k] = 0
        buttons[k].label.set_text(f"{K_new[k]: .1e}")
        fig.canvas.draw_idle()
        if AUTO:
            resolve()

    return update


def reset(event):
    global K_new
    K_new = copy.deepcopy(K_base)
    for k in sliders:
        sliders[k].reset()
        buttons[k].label.set_text(f"{K_new[k]: .1e}")
    fig.canvas.draw_idle()
    if AUTO:
        resolve()


solve_axes = fig.add_axes([0.02, 0.05, 0.05, 0.05])
solve_button = Button(solve_axes, 'Solve', hovercolor='0.975')
solve_button.on_clicked(resolve)

reset_axes = fig.add_axes([0.08, 0.05, 0.05, 0.05])
reset_button = Button(reset_axes, 'Reset', hovercolor='0.975')
reset_button.on_clicked(reset)

for i, k in enumerate(K_base):
    k_axes = fig.add_axes(
        [
            0.03,  # left
            0.95 - 0.04 * i,  # bottom
            0.10,  # width
            0.03,  # height
        ]
    )
    amp_slider = Slider(
        ax=k_axes,
        label=k,
        valmin=-3,
        valmax=3,
        valinit=0,
        valstep=0.1,
        orientation="horizontal",
    )
    sliders[k] = amp_slider
    amp_slider.on_changed(get_slider_action(k))

    k_zero_axes = fig.add_axes(
        [
            0.15,  # left
            0.955 - 0.04 * i,  # bottom
            0.05,  # width
            0.02,  # height
        ]
    )
    zero_button = Button(k_zero_axes, f"{K_new[k]: .1e}", hovercolor='0.975')
    buttons[k] = zero_button
    zero_button.on_clicked(get_zero_button_action(k))
