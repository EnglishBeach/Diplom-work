from peewee import MySQLDatabase

from peewee import Model,CharField,DateTimeField,datetime, DecimalField,ForeignKeyField
from playhouse.mysql_ext import JSONField

import pandas as pd

# TODO: do beauty
# to export structure
# pwiz -H localhost -u root -e mysql comsol_solves>structure.py

db = MySQLDatabase(
    'comsol_solves',
    user='root',
    password='',
    autoconnect=False,
)


class BaseModel(Model):

    class Meta:
        database = db


class Solve(BaseModel):
    name = CharField(20)
    desc = CharField(100, null=True)
    date = DateTimeField(default=datetime.datetime.now)
    data = JSONField(null=False)

    Ke = DecimalField(max_digits=20, decimal_places=5, default=0)
    KH = DecimalField(max_digits=20, decimal_places=5, default=0)
    Kr = DecimalField(max_digits=20, decimal_places=5, default=0)
    Kdisp = DecimalField(max_digits=20, decimal_places=5, default=0)
    KqH = DecimalField(max_digits=20, decimal_places=5, default=0)
    Ks = DecimalField(max_digits=20, decimal_places=5, default=0)
    Kd = DecimalField(max_digits=20, decimal_places=5, default=0)
    Kc = DecimalField(max_digits=20, decimal_places=5, default=0)
    Kp = DecimalField(max_digits=20, decimal_places=5, default=0)
    KrD = DecimalField(max_digits=20, decimal_places=5, default=0)
    Kph = DecimalField(max_digits=20, decimal_places=5, default=0)
    light = DecimalField(max_digits=20, decimal_places=5, default=0)

    class Meta:
        table_name = 'solve'


def solve_to_sql(df, params: dict, name, desc=None):
    note = params.copy()
    note['name'] = name
    note['desc'] = desc
    note['data'] = df.to_json(index=True)
    with db:
        Solve.insert(note).execute()


def sweep_to_sql(notes):
    assert db.is_connection_usable(), 'Database not connected'
    Solve.insert_many(notes).execute()

def get_solves(conditious):
    """
    Get
    """
    string = 'select * from solve \n where \n'
    for key, diap in conditious.items():
        string += f'{diap[0]} <= {key} and {key} <= {diap[1]} \n and \n'

    querry = string[:-8]
    with db:
        columns = [i.name for i in db.get_columns('solve')]
        cursor = db.execute_sql(querry)
        result = cursor.fetchall()
    df = pd.DataFrame(columns=columns, data=result)

    datas = df['data']
    data_df_list = []
    for data in datas:
        data_df_list.append(pd.read_json(data[1:-1].replace('\\', '')))

    del df['data']
    return df, data_df_list

# FIXME: update not delete
if __name__ == '__main__':
    with db:
        tables = [Solve]
        db.drop_tables(tables)
        db.create_tables(tables)
