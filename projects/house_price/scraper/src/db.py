import sqlite3
from typing import Final, Literal

from sqlalchemy import create_engine, NVARCHAR
from pandas import DataFrame

SERVER: Final = 'LENOVO'
DATABASE: Final = 'olx'

# NOTE: Windows Authentication
DATABASE_URL: Final = f'mssql+pyodbc://{SERVER}/{DATABASE}?trusted_connection=yes&driver=ODBC+Driver+17+for+SQL+Server'
engine = create_engine(DATABASE_URL)

TABLE_SCHEMA = {
    'title': NVARCHAR(None),
    'price': NVARCHAR(None),
    'currency': NVARCHAR(10),
    'is_negotiable': NVARCHAR(3),
    'location': NVARCHAR(None),
    'area': NVARCHAR(None),
}

Formats = Literal['mssql', 'sqlite', 'csv']

class DBWriter:
    def __init__(self, df: DataFrame, format: Formats='mssql'):
        self.df = df
        self.format = format

    def __to_mssql(self):
        self.df.to_sql(
            'furniture',
            con=engine,
            index=False,
            if_exists='append',
            dtype=TABLE_SCHEMA,
        )
    
    def __to_csv(self):
        self.df.to_csv('furniture.csv', index=False)

    def __to_sqlite(self):
        con = sqlite3.connect('databse.sqlite3')
        self.df.to_sql(
            'furniture',
            con=con,
            index=False,
            if_exists='append'
        )

    def save(self):
        match self.format:
            case 'sqlite':
                self.__to_sqlite()
            case 'mssql':
                self.__to_mssql()
            case 'csv':
                self.__to_csv()
            case _:
                raise ValueError('`format` can be one of (sqlite, mssql, csv)')
