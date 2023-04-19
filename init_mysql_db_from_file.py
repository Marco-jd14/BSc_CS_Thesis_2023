# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 13:38:48 2023

@author: Marco
"""

import mysql.connector
import sys
import regex as re
import pandas as pd
from pprint import pprint
from copy import copy

HOST_ARGS = {
    'user'     : 'root',
    'password' : 'V3ryStrongP@ssw0rd!',
    'host'     : 'localhost'
}
DATABASE_NAME = 'quiet_nl'

SQL_DB_PATH = "./quiet_2023-04-19.sql"


def mysql_connect(host_args, database=None):
    connection, err_msg = None, None

    try:
        if database is not None:
            connection = mysql.connector.connect(**host_args, database=database)
        else:
            connection = mysql.connector.connect(**host_args)

    except mysql.connector.Error as e:
        err_msg = str(e.errno) + " " + e.msg
    except Exception as e:
        err_msg = str(type(e))[len("<class '"):-2] + ": " + str(e)

    return connection, err_msg


def establish_host_connection():
    # Establish connection with provided user-name, password, and host
    connection, err_msg = mysql_connect(HOST_ARGS)

    if err_msg is not None:
        print(err_msg)
        print("Could not establish a connection to MySQL. Please make sure you have " + \
              "correctly specified the user-name and password, and the host address")
        sys.exit(0)

    return connection


def establish_database_connection(connection, overwrite=False):
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


def parse_sql_file():
    with open(SQL_DB_PATH, 'r', encoding='utf-8') as sql_file:
        sql_file_contents = sql_file.read()

        # Remove in-line comments (start with /* and end with */)
        sql_file_contents = re.sub("/\*.*\*/", "", sql_file_contents)
        # Remove beginning-until-end-of-line comments (start with \n# and end with \n)
        while re.search("(^|\n)#.*\\n", sql_file_contents):
            sql_file_contents = re.sub("(^|\n)#.*\\n", "\n", sql_file_contents)

        # Split up the file_contents into commands, and filter out empty lines
        sql_commands = sql_file_contents.split(';\n')
        sql_commands = list(map(lambda line: line.strip(), sql_commands))
        sql_commands = list(filter(lambda line: line != "", sql_commands))

    return sql_commands


def execute_create_table_commands(cursor, create_table_commands, verbose=False):
    """ When executing CREATE TABLE commands in the same order as parsed from the file,
    this could lead to REFERENCE-errors, i.e. a table wants to reference a second table
    which has not been created yet.

    The implemented solution: simply try to create all tables, and save the commands that
    were unsuccessful because of these REFERENCE-errors.
    Then, once at least some new tables have been created, see if that solved the REFERENCE-errors
    by trying the failed commands again.
    
    Furthermore, the order in which the tables will be successfully created, is also the order
    in which the tables have to be filled with data.
    """

    # Store the order in which the commands are successfully executed
    execution_order_of_table_creation_commands = []

    # Store the unsuccessful commands, so we can retry them
    unsuccessful_commands = copy(create_table_commands)
    while len(unsuccessful_commands) > 0:
        # The commands to retry are commands that were unsuccessful in previous iteration
        commands_to_try = copy(unsuccessful_commands)
        # Reset the list of unsuccessful commands for this iteration
        unsuccessful_commands = []

        for i, command in enumerate(commands_to_try):
            print("\r[%02d/%02d] %s"%(len(execution_order_of_table_creation_commands)+1,len(create_table_commands),command.split('(')[0]) + " "*50, end='')

            try:
                cursor.execute(command)
                execution_order_of_table_creation_commands.append(command)
            except mysql.connector.Error as e:
                if e.errno == 1824: # Failed to open the referenced table ''
                    unsuccessful_commands.append(command)
                    if verbose:
                        print("\nCommand could not be executed successfully, and will be retried later")
                        print(e.errno, e.msg)
                else:
                    raise

        if len(unsuccessful_commands) == len(commands_to_try):
            print("It is impossible to execute all CREATE TABLE commands, as some commands keep failing")
            sys.exit(0)

    print("\nSuccesfully created all the tables")
    return execution_order_of_table_creation_commands


def order_other_commands_in_table_creation_order(execution_order_of_table_names, other_sql_commands):
    unlock_tables_commands = list(filter(lambda c: c.strip().startswith("UNLOCK TABLES"), other_sql_commands))

    ordered_other_sql_commands = []
    for table_name in execution_order_of_table_names:
        # Check if the table_name occurs in the command before the first opening bracket
        table_commands = list(filter(lambda c: "`%s`"%table_name in c.split('(')[0], other_sql_commands))
        if len(table_commands) == 0:
            continue

        assert table_commands[0].startswith("LOCK TABLES `%s` WRITE"%table_name), table_commands
        ordered_other_sql_commands.extend(table_commands)
        # Also add one of the UNLOCK TABLES commands at the end of the commands for this specific table
        ordered_other_sql_commands.append(unlock_tables_commands.pop())

    # Add any remaining commands that were not specifically for one of the table-names
    remaining_commands = list(set(other_sql_commands) - set(ordered_other_sql_commands))
    # print("Remaining commands unrelated to any tables:", remaining_commands)
    ordered_other_sql_commands = remaining_commands + ordered_other_sql_commands

    # Check if the made assumptions are correct
    assert len(unlock_tables_commands) == 0, "%d 'UNLOCK TABLES' remain unused"%len(unlock_tables_commands)
    assert len(ordered_other_sql_commands) == len(other_sql_commands), "%d != %d"%(len(ordered_other_sql_commands), len(other_sql_commands))

    # Pretty-print all the ordered commands until the first opening bracket
    # pprint(list(map(lambda c: c.split('(')[0].strip(), ordered_other_sql_commands)))
    return ordered_other_sql_commands


def execute_commands(cursor, commands):
    # Execute every command from the input file
    for i, command in enumerate(commands):
        success = False
        print("\r[%03d/%03d] %s"%(i+1,len(commands), command.split('(')[0]) + " "*100, end='')
        try:
            cursor.execute(command)
            success = True
        except Exception as e:
            print("\nCommand could not be executed successfully")
            print(e.errno, e.msg)

        if not success:
            break

    if success: 
        print("Succesfully executed all commands")
    else: 
        print("did not execute all commands successfully")
        sys.exit(0)


def initialize_new_db_from_scratch(db):
    cursor = db.cursor()

    # Split the .sql file up into a list of commands
    sql_commands = parse_sql_file()
    print("Parsed %d commands from .sql file"%len(sql_commands))

    # Split up the commands that are about CREATE TABLE and the remaining commands
    # TODO: execute DROP TABLE commands
    create_table_commands = list(filter(lambda command: command.startswith("CREATE TABLE"), sql_commands))
    other_sql_commands = list(filter(lambda command: not (command.startswith("CREATE TABLE") or command.startswith("DROP TABLE")), sql_commands))

    print("\nFirst creating all the tables")
    execution_order_of_table_creation_commands = execute_create_table_commands(cursor, create_table_commands)

    # Order the remaining commands about tables in the same order as in which the tables were created
    execution_order_of_table_names = list(map(lambda c: c.split('(')[0].split()[2].strip('`'), execution_order_of_table_creation_commands))
    ordered_other_sql_commands = order_other_commands_in_table_creation_order(execution_order_of_table_names, other_sql_commands)

    # Execute the remaining commands
    print("\nExecuting the remaining commands")
    execute_commands(cursor, ordered_other_sql_commands)

    # Commit the changes to the database
    db.commit()

    # Close the cursor
    cursor.close()


def main():
    make_new_db_from_scratch = False

    connection = establish_host_connection()
    assert ' ' not in DATABASE_NAME, "Spaces are not allowed in the name of a database"

    if make_new_db_from_scratch:
        db = establish_database_connection(connection, overwrite=True)
        print("Successfully connected to database '%s'"%DATABASE_NAME)
        initialize_new_db_from_scratch(db)
    else:
        db = establish_database_connection(connection, overwrite=False)
        print("Successfully connected to database '%s'"%DATABASE_NAME)

        cursor = db.cursor()
        query = "SELECT * FROM category;"

        cursor.execute(query)
        df = pd.DataFrame(cursor.fetchall(), columns=cursor.column_names)
        print(df)

        df = pd.read_sql_query(query, db)
        print(df)

        cursor.close()

    # Close the connection to the database
    db.close()
    connection.close()

if __name__ == '__main__':
    main()
