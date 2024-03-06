import pandas as _pd
import numpy as _np
import re as _re
from tqdm import tqdm as _tqdm
import mph as _mph

# TODO: logs
def sweep_to_sql(
    model: _mph.Model,
    name,
    desc=None,
):
    _, light_sweep = model.outer('Data')
    i = 1
    params = {
        key: float(value)
        for key,
        value in model_parameters(model).items()
        if 'light' not in key
    }
    params['name'] = name
    params['desc'] = desc

    notes = []
    for light_value in light_sweep:
        note = {}
        note.update(params)
        note['light'] = light_value
        df = evaluate_expressions(model, i)
        note['data'] = df.to_json(index=True)
        notes.append(note)

        i += 1

    assert db.is_connection_usable(), 'Database not connected'
    Solve.insert_many(notes).execute()


def sweep(
    model: _mph.Model,
    tuning_params,
    name=None,
    desc=None,
):
    name = input_check(name)
    desc = input_check(desc)

    tuning_list = _tqdm(iterable=tuning_params)

    i = 0
    for changed_params in tuning_list:
        model_parametrs(
            model=model,
            changed_params=changed_params,
        )
        model.clear()
        tuning_list.set_description('{:10}'.format('Solving...'))
        model.solve()
        tuning_list.set_description('{:10}'.format('Saving...'))
        sweep_to_sql(
            model=model,
            name=name + f'#{i}',
            desc=desc,
        )
        i += 1
