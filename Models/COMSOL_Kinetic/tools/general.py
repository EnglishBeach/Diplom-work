import itertools


def input_check(string):
    if string is None:
        while string != 'q':
            string = input(f'Set {string=}, to quit - q:')
            string = string.strip()
    return string


def make_comdinations(diap: dict):
    keys = list(diap.keys())
    values = [diap[key] for key in keys]
    combinations = list(itertools.product(*values))
    result = [dict(zip(keys, comb)) for comb in combinations]
    return result
