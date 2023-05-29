# -*- coding: utf-8 -*-
"""
Created on Mon May  1 11:59:50 2023

@author: Marco
"""

import sys
import sqlalchemy
import pandas as pd


HOST_ARGS = {
    'username' : 'root',
    'password' : 'V3ryStrongP@ssw0rd!',
    'host'     : 'localhost'
}
DATABASE_NAME = 'quiet_nl'


def mysql_connect(host_args, database_name=None):
    """ Helper function that connects to a certain host, with a database_name as optional argument
    """
    connection, err_msg = None, None

    DRIVER = 'mysqldb' # This is the preferred driver, but it could be that it does not work well with unescaped '%' in raw string text
    # DRIVER = 'mysqlconnector'
    try:
        # Connect with SQL Alchemy, as this is the preferred way for pandas to query the database
        connection_url = sqlalchemy.engine.URL.create("mysql+%s"%DRIVER, **host_args, database=database_name)
        engine = sqlalchemy.create_engine(connection_url)
        connection = engine.connect()

    except sqlalchemy.exc.SQLAlchemyError as e:
        err_msg = str(e.orig).strip("()")
    except Exception as e:
        err_msg = str(type(e))[len("<class '"):-2] + ": " + str(e)

    return connection, err_msg


def establish_host_connection():
    """ Establish a connection to a host specified in HOST_ARGS
    """
    # Establish connection with provided user-name, password, and host
    connection, err_msg = mysql_connect(HOST_ARGS)

    if err_msg is not None:
        print(err_msg)
        print("Could not establish a connection to MySQL. Please make sure you have " + \
              "correctly specified the user-name and password, and the host address")
        sys.exit(0)

    return connection


def establish_database_connection(connection, overwrite=False):
    """ From an existing host connection, establish a database connection specified in DATABASE_NAME
    """

    assert ' ' not in DATABASE_NAME, "Spaces are not allowed in the name of a database"

    if overwrite:
        # Delete the database if it already exists
        print("Deleting existing database %s"%DATABASE_NAME)
        connection.execute(sqlalchemy.text('drop database if exists %s'%DATABASE_NAME))

    # Create database if it does not exist
    connection.execute(sqlalchemy.text('create database if not exists %s'%DATABASE_NAME))
    connection.close()

    # Establish connection to a provided DATABASE_NAME
    db, err_msg = mysql_connect(HOST_ARGS, DATABASE_NAME)

    if err_msg is not None:
        print(err_msg)
        print("Could not establish a connection to the MySQL database '%s'"%DATABASE_NAME)
        sys.exit(0)

    return db


# Example usage:
if __name__ == '__main__':
    # How to establish a connection to the database
    conn = establish_host_connection()
    db   = establish_database_connection(conn, overwrite=False)
    print("Successfully connected to database '%s'"%str(db.engine).split("/")[-1][:-1])

    query = "show tables;"
    result = db.execute(sqlalchemy.text(query))
    df = pd.DataFrame(result.fetchall())
    print(df)

    # The following is equivalent
    query = "show tables;"
    df = pd.read_sql_query(sqlalchemy.text(query), db)
    print(df)

    # Close connection to the database when you're done with it
    db.close()
    conn.close()
