import re

import mph
import pandas as pd
from tqdm import tqdm


class AbstractStudy:
    study_name = None

    def __init__(self, model: mph.Model):
        self.comsol_model = model

    @property
    def time_end(self):
        return self.comsol_model.parameters()['time_end']

    @property
    def constants(self):
        rule = lambda key: key[0] == 'K'
        return {
            key: value for key, value in self.comsol_model.parameters().items() if rule(key)
        }  # yapf: disable

    @property
    def initial(self):
        rule = lambda key: ('0' in key) or (key in ['light'])
        return {
            key: value for key, value in self.comsol_model.parameters().items() if rule(key)
        }  # yapf: disable

    @property
    def species(self) -> dict:
        reaction_node_children = [i.name() for i in self.nodes['reaction'].children()]

        species = re.findall(
            string='\n'.join(reaction_node_children),
            pattern='Species: (.*)',
        )
        return {specie: f'reaction.c_{specie}' for specie in species}

    @property
    def nodes(self):
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
            dataset=self.nodes['data'].name(),
            outer=outer_number,
        )
        return pd.DataFrame(row_data, columns=list(functions))

    def solve(self):
        self.comsol_model.solve(study=self.nodes['study'].name())


class Generator(AbstractStudy):
    study_name = 'Generator'

    def evaluate(self, functions={}) -> pd.DataFrame:
        return super().evaluate(outer_number=1, functions=functions)

    def sweep(self, combinations):
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
    def nodes(self):
        nodes_dict = super().nodes

        sensivity_node = nodes_dict['study'] / 'Sensitivity'
        assert sensivity_node.exists(), f'Estimation node does not exist'

        nodes_dict.update({'sensitivity': sensivity_node})
        return nodes_dict

    @property
    def constants(self):
        all_parameters = self.nodes['sensitivity'].properties()
        filtered_properties = {
            key: value for key, value in zip(all_parameters['pname'], all_parameters['initval'])
        }  # yapf: disable

        rule = lambda key: key[0] == 'K'
        return {
            key: value for key, value in filtered_properties.items() if rule(key)
        }  # yapf: disable

    @property
    def initial(self):
        all_parameters = self.nodes['sensitivity'].properties()
        filtered_properties = {
            key: value for key, value in zip(all_parameters['pname'], all_parameters['initval'])
        }  # yapf: disable

        rule = lambda key: ('0' in key) or (key in ['light'])
        return {
            key: value for key, value in filtered_properties.items() if rule(key)
        }  # yapf: disable

    def set_parametrs(self, **parameters):
        all_parameters = self.constants
        all_parameters.update(self.initial)
        old_len = len(all_parameters)

        all_parameters.update(parameters)
        assert len(all_parameters) == old_len, 'Parametrs not exist'

        self.nodes['sensitivity'].property(
            name='pname',
            value=list(all_parameters),
        )
        self.nodes['sensitivity'].property(
            name='initval',
            value=[str(i) for i in all_parameters.values()],
        )

    @property
    def target(self):
        result = self.nodes['sensitivity'].properties()['optobj'][0]
        return result.replace('comp.reaction.c_', '')

    def set_target(self, target: str):
        assert f'reaction.c_{target}' in self.species.values(), 'Target is not specie'
        self.nodes['sensitivity'].property(
            name='optobj',
            value=[f'comp.reaction.c_{target}'],
        )


# TODO: out of found parametrs
class Estimator(AbstractStudy):
    study_name = 'Estimator'

    @property
    def nodes(self):
        nodes_dict = super().nodes

        estimation_node = self.comsol_model / 'physics' / 'Reaction Engineering' / 'Estimation'
        assert estimation_node.exists(), f'Estimation node does not exist'

        nodes_dict.update({'estimation': estimation_node})
        return nodes_dict

    @property
    def experiments(self):
        return self.nodes['estimation_node'].children()

    @property
    def tables(self):
        tables = self.comsol_model / 'tables'
        experiment_tables = [node for node in tables.children() if 'Experiment' in node.name()]
        return experiment_tables

    def create_one_experiment(
        self,
        data,
        data_columns,
        experiment_i,
        path=r'D:\WORKS\COMSOL_polymers\Batch\generator_out_short.csv',
    ):
        experiment_name = f'exp{experiment_i}'

        # create experiment
        self.nodes['estimation_node'].java.create(
            experiment_name,
            "Experiment",
            -1,
        )
        experiment = self.experiments[-1]

        # create table
        table_tag = f"tbl_compreactionest1{experiment_name}"
        table = (self.comsol_mode / 'tables').java.create(table_tag, "Table")
        table.label(f"Experiment {experiment_i} Table")
        table.setTableData(data)
        table.active(False)

        # set up parametrs
        variables_dict = {'Time': 't'}
        variables_dict.update(self.species())
        variables = [variables_dict[key] for key in data_columns]

        Estimator._set_node_properties(
            node=experiment,
            fileName=path,
            dataColumn=data_columns,
            use=[1] * len(data_columns),
            modelVariable=variables,
        )

    def create_experiments(self, datas: list[pd.DataFrame]):
        i = 0
        for data in datas:
            self.create_one_experiment(
                data=data,
                data_columns=data.columns,
                experiment_i=i,
                path=r'./generator_out_short.csv',
            )
            i += 1

    def clear_experiments(self):
        experiments, tables = self.experiments, self.tables
        for table in tables:
            table.remove()
        for experiment in experiments:
            experiment.remove()

    def solve(self):
        self.nodes['estimation'].toggle('on')
        try:
            self.solve()
        finally:
            self.nodes['estimation'].toggle('off')
