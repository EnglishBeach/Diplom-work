import re

import mph
import pandas as pd
from tqdm import tqdm
from pathlib import Path


class AbstractStudy:
    study_name = None

    def __init__(self, model: mph.Model):
        self.comsol_model = model

    @property
    def time_end(self):
        return float(self.comsol_model.parameters()['time_end'])

    @property
    def constants(self):
        rule = lambda key: key[0] == 'K'
        return {
            key: eval(value) for key, value in self.comsol_model.parameters().items() if rule(key)
        }

    @property
    def initial(self):
        rule = lambda key: ('0' in key) or (key in ['light'])
        return {
            key: eval(value) for key, value in self.comsol_model.parameters().items() if rule(key)
        }

    @property
    def species(self) -> dict:
        reaction_node_children = [i.name() for i in self._nodes['reaction'].children()]

        species = re.findall(
            string='\n'.join(reaction_node_children),
            pattern='Species: (.*)',
        )
        return {specie: f'reaction.c_{specie}' for specie in species}

    @property
    def _nodes(self):
        study_node = self.comsol_model / 'studies' / f'{self.study_name}'
        assert study_node.exists(), f'Study node does not exist'

        solution_node = self.comsol_model / 'solutions' / f'{self.study_name}_Solution'
        assert study_node.exists(), f'Solution node does not exist'

        data_node = self.comsol_model / 'datasets' / f'{self.study_name}_data'
        assert study_node.exists(), f'Data node does not exist'

        reaction_node = self.comsol_model / 'physics' / 'Reaction Engineering'
        assert study_node.exists(), f'Reaction node does not exist'

        nodes_dict = {
            'study': study_node,
            'solution': solution_node,
            'data': data_node,
            'reaction': reaction_node,
        }
        return nodes_dict

    def set_parametrs(self, **parameters):
        for key, value in parameters.items():
            self.comsol_model.parameter(name=key, value=value)

    @staticmethod
    def _set_node_properties(node: mph.Node, **properties):
        for key, value in properties.items():
            node.property(key, value)

    def evaluate(
        self,
        functions: dict,
        outer_number=1,
    ) -> pd.DataFrame:

        model = self.comsol_model
        functions.update({'time': 't'})
        row_data = model.evaluate(
            list(functions.values()),
            dataset=self._nodes['data'].name(),
            outer=outer_number,
        )
        return pd.DataFrame(row_data, columns=list(functions))

    def solve(self):
        self.comsol_model.solve(study=self._nodes['study'].name())


class Generator(AbstractStudy):
    study_name = 'Generator'

    def evaluate(self, functions={}) -> pd.DataFrame:
        return super().evaluate(outer_number=1, functions=functions)

    def sweep(self, combinations: list[dict]):
        result = []
        combinations = tqdm(iterable=combinations)
        for combination in combinations:
            self.set_parametrs(**combination)
            self.solve()
            df = self.evaluate(self.species())
            df.loc[:, combination.keys()] = list(combination.values())
            result.append(df)
        return pd.concat(result)


class Sensitivity(AbstractStudy):
    study_name = 'Sensitivity'

    @property
    def sensitivities(self):
        return {key: f'fsens({key})' for key in self.constants}

    @property
    def _nodes(self):
        nodes_dict = super()._nodes

        sensivity_node = nodes_dict['study'] / 'Sensitivity'
        assert sensivity_node.exists(), f'Estimation node does not exist'

        nodes_dict.update({'sensitivity': sensivity_node})
        return nodes_dict

    @property
    def constants(self):
        all_parameters = self._nodes['sensitivity'].properties()
        filtered_properties = {
            key: value for key, value in zip(all_parameters['pname'], all_parameters['initval'])
        }

        rule = lambda key: key[0] == 'K'
        return {key: value for key, value in filtered_properties.items() if rule(key)}

    @property
    def initial(self):
        all_parameters = self._nodes['sensitivity'].properties()
        filtered_properties = {
            key: value for key, value in zip(all_parameters['pname'], all_parameters['initval'])
        }

        rule = lambda key: ('0' in key) or (key in ['light'])
        return {key: value for key, value in filtered_properties.items() if rule(key)}

    def set_parametrs(self, **parameters):
        all_parameters = self.constants
        all_parameters.update(self.initial)
        old_len = len(all_parameters)

        all_parameters.update(parameters)
        assert len(all_parameters) == old_len, 'Parametrs not exist'

        self._nodes['sensitivity'].property(
            name='pname',
            value=list(all_parameters),
        )
        self._nodes['sensitivity'].property(
            name='initval',
            value=[str(i) for i in all_parameters.values()],
        )

    @property
    def target(self):
        result = self._nodes['sensitivity'].properties()['optobj'][0]
        return result.replace('comp.reaction.c_', '')

    def set_target(self, target: str):
        assert f'reaction.c_{target}' in self.species.values(), 'Target is not specie'
        self._nodes['sensitivity'].property(
            name='optobj',
            value=[f'comp.reaction.c_{target}'],
        )


# TODO: out of found parametrs
class Estimator(AbstractStudy):
    study_name = 'Estimator'
    _experiment_id = 0

    @property
    def _nodes(self):
        nodes_dict = super()._nodes

        estimation_node = self.comsol_model / 'physics' / 'Reaction Engineering' / 'Estimation'
        assert estimation_node.exists(), f'Estimation node does not exist'

        nodes_dict.update({'estimation': estimation_node})
        return nodes_dict

    @property
    def experiments(self):
        return self._nodes['estimation'].children()

    @property
    def tables(self):
        tables = self.comsol_model / 'tables'
        experiment_tables = [node for node in tables.children() if 'Experiment' in node.name()]
        return experiment_tables

    def add_experiment(
        self,
        data: pd.DataFrame,
        data_columns: dict[str, str],
        data_path: Path = Path(r'D:\WORKS\COMSOL_polymers\Batch\generator_out_short.csv'),
    ):
        self._experiment_id += 1
        experiment_name = f'exp{self._experiment_id}'

        # create experiment
        self._nodes['estimation_node'].java.create(
            experiment_name,
            "Experiment",
            -1,
        )

        # create table
        table_tag = f"tbl_compreactionest1{experiment_name}"
        table = (self.comsol_model / 'tables').java.create(table_tag, "Table")
        table.label(f"Experiment {self._experiment_id} Table")
        table.setTableData(data)
        table.active(False)

        # set estimations
        Estimator._set_node_properties(
            node=self.experiments[-1],
            fileName=data_path,
            use=[1] * len(data_columns),
            dataColumn=list(data_columns.keys()),
            modelVariable=list(data_columns.values()),

        )

    def clear_experiments(self):
        experiments, tables = self.experiments, self.tables
        for table in tables:
            table.remove()
        for experiment in experiments:
            experiment.remove()

    def solve(self):
        self._nodes['estimation'].toggle('on')
        try:
            self.solve()
        finally:
            self._nodes['estimation'].toggle('off')


def copy_settings(from_node: mph.Node, to_node: mph.Node):
    for i in range(2):
        from_properties, to_properties = from_node.properties(), to_node.properties()

        auto_settings = {}
        for key, value in from_properties.items():
            if (str(value) == 'auto') or (to_properties.get(key, None) is None):
                auto_settings.update({key: value})
                continue
            to_node.property(name=key, value=value)

        for key, value in auto_settings.items():
            if to_properties.get(key, None) is None:
                continue
            to_node.property(name=key, value=value)


def compare(from_node: mph.Node, to_node: mph.Node):
    from_properties, to_properties = from_node.properties(), to_node.properties()
    all_properties = sorted(set(from_properties).union(set(to_properties)))

    print(f"{'from ' + from_node.name(): >53} | {'to '+to_node.name(): <53}")
    for key in all_properties:
        from_prop = str(from_properties.get(key, None))
        to_prop = str(to_properties.get(key, None))

        string = (
            f'{"*" if (from_prop!=to_prop) else " "} {key: <30} '
            + f'{from_prop[:20]: <20} | {to_prop[:20]: <20}'
        )
        print(string)


def copy_solver(from_solver: mph.Node, to_solver: mph.Node, verbose=False):
    copy_settings(from_solver, to_solver)
    from_dict = {node.name(): node for node in from_solver.children()}
    to_dict = {node.name(): node for node in to_solver.children()}

    for node in from_dict:
        from_node = from_dict[node]
        to_node = to_dict[node]
        copy_settings(from_node, to_node)
        if verbose:
            compare(from_node, to_node)
            print('*' * 120)
