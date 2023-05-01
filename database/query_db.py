# -*- coding: utf-8 -*-
"""
Created on Mon May  1 11:47:01 2023

@author: Marco
"""

import connect_db
import pandas as pd
from pprint import pprint


def main():
    conn = connect_db.establish_host_connection()
    db   = connect_db.establish_database_connection(conn)
    print("Successfully connected to database '%s'"%db.database)

    cursor = db.cursor()
    query = "SELECT * FROM category;"

    cursor.execute(query)
    df = pd.DataFrame(cursor.fetchall(), columns=cursor.column_names)
    # print(df)

    df = pd.read_sql_query(query, db)
    # print(df)


    # Close the connection to the database
    cursor.close()
    db.close()
    conn.close()


if __name__ == '__main__':
    main()