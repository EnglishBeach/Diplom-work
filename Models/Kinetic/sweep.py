import itertools

import numpy as np
import pandas as pd
import Models.Kinetic.tools.schemes as schemes
from Models.Kinetic.tools.schemes import Solver
from tqdm import tqdm


COMPS = ['Q', 'QHH', 'DH', 'D']


def get_df(y: Solver):
    res = {'time': y.T}
    for comp in COMPS:
        res[comp] = y[comp]
    return pd.DataFrame(res)


def combine(k: dict):
    res = lambda v: [v * 0.1, v, v * 10]
    return {key: res(value) for key, value in k.items()}


def get_combinations(combination_dict: dict[list]) -> dict:
    keys, values = zip(*combination_dict.items())
    return (dict(zip(keys, v)) for v in itertools.product(*values))


def sweep(Y1: Solver, Y2: Solver, init={}):
    data = []
    k_dict = Y2.K
    desc = tqdm(total=3 ** (len(k_dict)))
    try:
        for k in get_combinations(combine(k_dict)):
            status = 'OK'
            try:
                Y1.solve(K=k, initial=init)
                Y2.solve(K=k, initial=init)
            except:
                status = 'ERROR'

            df1 = get_df(Y1)
            df2 = get_df(Y2)

            df_res = np.abs(df1 - df2)
            res: dict = df_res.max().iloc[1:].to_dict()
            res['status'] = status
            res.update(k)

            data.append(res)
            desc.update()
    except:
        df = pd.DataFrame(data)
        df.to_csv('result_error.csv')
    df = pd.DataFrame(data)
    df.to_csv('result.csv')


class K(schemes.K_gen):
    # INITIATION
    l = 1e9 * 0.00001  # 8 10
    l_ = 1e5  # 5 6

    qE = 1e9  # 8 10
    qE_ = 1e8  # 7 9
    H = 1e9  # 8 10

    qH = 1e6  # 2000
    redQ = 1e9
    oxQ = 1

    qQD = 2

    qPh = 1e-5

    D = 1
    D_ = 0.05

    r = 1e9
    p = 0.0001

    rD = 1e9


class K_sweep(schemes.K_gen):
    # INITIATION

    l = 1e9 * 0.00001  # 8 10
    l_ = 1e5  # 5 6

    qE = 1e9  # 8 10
    qE_ = 1e8  # 7 9
    H = 1e9  # 8 10

    qH = 1e6  # 2000
    redQ = 1e9

    r = 1e9

    rD = 1e9


def system_full(t, C, k=K()):
    [Q, Qt, DH, Q_DH, QH, D, QHH, QHD, QD] = C
    C1 = Q
    C2 = Qt
    C3 = DH
    C4 = Q_DH
    C5 = QH
    C6 = D
    C7 = QHH
    C8 = QHD
    C9 = QD

    R1 = k.l * C1
    R2 = k.l_ * C2
    R3 = k.qE * C2 * C3
    R4 = k.qE_ * C4
    R5 = k.H * C4
    R6 = k.qH * C2 * C7
    R7 = k.redQ * C5 * C5
    R8 = k.oxQ * C1 * C7
    R9 = k.qQD * C2 * C8
    R10 = k.qPh * C2
    R11 = k.D * C1 * C6
    R12 = k.D_ * C9
    R13 = k.r * C5 * C6
    R14 = k.p * C8
    R15 = k.rD * C6 * C6

    res = dict(
        dCl=-R1 + R2 + R7 - R8 - R11 + R12,
        dC2=R1 - R2 - R3 + R4 - R6 - R9 - R10,
        dC3=-R3 + R4,
        dC4=R3 - R4 - R5,
        dC5=R5 + 2 * R6 - 2 * R7 + 2 * R8 + R9 - R13,
        dC6=R5 - R11 + R12 - R13 - 2 * R15,
        dC7=-R6 + R7 - R8 + R14,
        dC8=-R9 + R13 - R14,
        dC9=R9 + R11 - R12,
    )

    return list(res.values())


c_full = '[Q, Qt, DH, Q_DH, QH, D, QHH, QHD, QD]'
initial_full = [1, 0, 1, 0, 0, 0, 0, 0, 0]
init_full = schemes.get_init(c_full, initial_full)


def system_base(t, C, k=K()):
    [Q, Qt, DH, QH, D, QHH] = C

    R1 = k.l * Q
    R2 = k.l_ * Qt
    R3 = k.H * Qt * DH
    R6 = k.qH * Qt * QHH
    R7 = k.redQ * QH * QH
    R13 = k.r * QH * D
    R15 = k.rD * D * D

    res = dict(
        Q=-R1 + R2 + R7,
        Qt=R1 - R2 - R3 - R6,
        DH=-R3,
        QH=R3 + 2 * R6 - 2 * R7 - R13,
        D=R3 - R13 - 2 * R15,
        QHH=-R6 + R7,
    )

    return list(res.values())


c_base = '[Q, Qt, DH, QH, D, QHH]'
initial_base = [1, 0, 1, 0, 0, 0]
init_base = schemes.get_init(c_base, initial_base)

T = np.linspace(0, 0.001, 100)

y1 = schemes.Solver(system_full, K(), init_full, T)
y2 = schemes.Solver(system_base, K_sweep(), init_base, T)
y2.initial
if __name__ == '__main__':
    sweep(y1, y2)
