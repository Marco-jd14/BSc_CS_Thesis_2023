# -*- coding: utf-8 -*-
"""
Created on Mon May  1 11:59:50 2023

@author: Marco
"""

import sys
import mysql.connector


HOST_ARGS = {
    'user'     : 'root',
    'password' : 'V3ryStrongP@ssw0rd!',
    'host'     : 'localhost'
}
DATABASE_NAME = 'quiet_nl'


def mysql_connect(host_args, database_name=None):
    """ Helper function that connects to a certain host, with a database_name as optional argument
    """
    connection, err_msg = None, None

    try:
        if database_name is not None:
            connection = mysql.connector.connect(**host_args, database=database_name)
        else:
            connection = mysql.connector.connect(**host_args)

    except mysql.connector.Error as e:
        err_msg = str(e.errno) + " " + e.msg
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
    cursor = connection.cursor()

    if overwrite:
        # Delete the database if it already exists
        print("Deleting existing database %s"%DATABASE_NAME)
        cursor.execute('drop database if exists %s'%DATABASE_NAME)

    # Create database if it does not exist
    cursor.execute('create database if not exists %s'%DATABASE_NAME)
    cursor.close()

    # Establish connection to a provided DATABASE_NAME
    db, err_msg = mysql_connect(HOST_ARGS, DATABASE_NAME)

    if err_msg is not None:
        print(err_msg)
        print("Could not establish a connection to the MySQL database '%s'"%DATABASE_NAME)
        sys.exit(0)

    return db
