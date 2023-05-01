# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 13:38:48 2023

@author: Marco
"""

import mysql.connector
import sys
import regex as re
import pandas as pd
from copy import copy
import connect_db
import datetime as dt


SQL_DB_PATH = "./quiet_2023-04-19.sql"


def parse_sql_file():
    """ This function opens a file specified in the global constant SQL_DB_PATH,
    then removes some comments for clarity, and returns the list of commands present in the file"""
    with open(SQL_DB_PATH, 'r', encoding='utf-8') as sql_file:
        sql_file_contents = sql_file.read()

        # Remove in-line comments (start with /* and end with */)
        sql_file_contents = re.sub("/\*.*\*/", "", sql_file_contents)
        # Remove beginning-of-line-until-end-of-line comments (start with \n# and end with \n)
        # NB: If you want to remove comments that do not start at the beginning of the line, make sure the hashtag is not part of a string '#', but an actual comment
        while re.search("(^|\n)#.*\\n", sql_file_contents):
            sql_file_contents = re.sub("(^|\n)#.*\\n", "\n", sql_file_contents)

        # Split up the file_contents into commands, and filter out empty lines
        sql_commands = sql_file_contents.split(';\n')
        sql_commands = list(map(lambda line: line.strip(), sql_commands))
        sql_commands = list(filter(lambda line: line != "", sql_commands))

    return sql_commands


def execute_drop_table_commands(cursor, drop_table_commands, verbose=False):
    execute_create_table_commands(cursor, drop_table_commands, verbose)

def execute_create_table_commands(cursor, create_table_commands, verbose=False):
    """ When executing CREATE TABLE commands in the same order as parsed from the file,
    this could lead to REFERENCE-errors, i.e. a table wants to reference a second table
    which has not been created yet.

    The implemented solution: simply try to create all tables, and save the commands that
    were unsuccessful because of these REFERENCE-errors for later.
    Then, once at least some new tables have been created, see if that solved the REFERENCE-errors
    by trying the failed commands again.

    Furthermore, the order in which the tables will be successfully created, is also the order
    in which the tables have to be filled with data. Therefore, this successful order is returned
    """

    # Store the order in which the commands are successfully executed
    execution_order_of_table_creation_commands = []

    # Store the unsuccessful commands, so we can retry them. For the first iteration, no commands were successful, so all unsuccessful
    unsuccessful_commands = copy(create_table_commands)

    # Loop until all commands have been executed successfully
    while len(unsuccessful_commands) > 0:
        # The commands to retry are commands that were unsuccessful in previous iteration
        commands_to_try = copy(unsuccessful_commands)
        # Reset the list of unsuccessful commands for this iteration
        unsuccessful_commands = []

        # Loop over all commands to try in this iteration
        for i, command in enumerate(commands_to_try):
            # Progress bar
            progress_bar(len(execution_order_of_table_creation_commands)+1, len(create_table_commands), command.split('(')[0])

            # Try to execute the command
            try:
                cursor.execute(command)
                # Upon success, add the command to the execution_order list
                execution_order_of_table_creation_commands.append(command)
            except mysql.connector.Error as e:
                if e.errno == 1824 or e.errno == 3730: 
                    # 1824 is the error-number for: Failed to open the referenced table 'table-name'
                    # 3739 is the error-number for: Cannot drop table 'table-name' referenced by a foreign key constraint
                    unsuccessful_commands.append(command)
                    if verbose:
                        print("\nCommand could not be executed successfully, and will be retried later")
                        print(e.errno, e.msg)
                else:
                    # Do not know how to handle other errors.
                    raise

        # If the length of the unsuccessful commands has not decreased in the last iteration, it is hopeless
        if len(unsuccessful_commands) == len(commands_to_try):
            print("It is impossible to execute all CREATE TABLE commands, as some commands keep failing")
            sys.exit(0)

    # Return the order in which the tables were successfully created
    return execution_order_of_table_creation_commands


def order_remaining_commands_in_table_creation_order(execution_order_of_table_names, remaining_sql_commands):
    """ This function has as input the order in which tables were created successfully. This order is
    necessary as the remaining commands have to be executed in this order as well, to prevent reference
    and/or foreign (key) constraint errors.
    So, this function orders the remaining sql commands in the same order as the create-table commands
    were successfully executed.
    """
    # Commands that are not about a specific table-name
    unlock_tables_commands = list(filter(lambda c: c.strip().startswith("UNLOCK TABLES"), remaining_sql_commands))

    # Loop over all the table-names
    ordered_other_sql_commands = []
    for table_name in execution_order_of_table_names:

        # Check if the table_name occurs in the command before the first opening bracket, and filter the commands for which this is true
        table_commands = list(filter(lambda c: "`%s`"%table_name in c.split('(')[0], remaining_sql_commands))
        if len(table_commands) == 0:
            continue

        # The first command should be a LOCK TABLE command
        assert table_commands[0].startswith("LOCK TABLES `%s` WRITE"%table_name), table_commands

        # Add the commands about this specific table to the global order of commands to be executed
        ordered_other_sql_commands.extend(table_commands)
        # Also add one of the UNLOCK TABLES commands to the end of the commands for this specific table
        ordered_other_sql_commands.append(unlock_tables_commands.pop())

    # Add any remaining commands that were not specifically for one of the table-names
    remaining_commands = list(set(remaining_sql_commands) - set(ordered_other_sql_commands))
    # print("Remaining commands unrelated to any tables:", remaining_commands)
    ordered_other_sql_commands = remaining_commands + ordered_other_sql_commands

    # Check if the made assumptions are correct
    assert len(unlock_tables_commands) == 0, "%d 'UNLOCK TABLES'-commands remain unused"%len(unlock_tables_commands)
    assert len(ordered_other_sql_commands) == len(remaining_sql_commands), "%d != %d"%(len(ordered_other_sql_commands), len(remaining_sql_commands))

    # Pretty-print all the ordered commands until the first opening bracket
    # pprint(list(map(lambda c: c.split('(')[0].strip(), ordered_other_sql_commands)))
    return ordered_other_sql_commands


def progress_bar(index, total, current_item):
    """ A progress bar that prints progress on a single line"""
    zero_padding = "0"*(len(str(total))-len(str(index)))
    print("\r[%s%d/%d] %s"%(zero_padding, index, total, current_item + " "*100), end='')


def execute_commands(cursor, commands):
    """ Executes commands from a list to a database-cursor
    """
    # Execute every command from the input file
    for i, command in enumerate(commands):
        # Progress bar
        progress_bar(i+1, len(commands), command.split('(')[0])

        # Try to execute the command
        success = False
        try:
            cursor.execute(command)
            success = True
        except Exception as e:
            print("\nCommand could not be executed successfully")
            print(e.errno, e.msg)

        # Stop if an error was encountered
        if not success:
            print("Could not execute all commands successfully")
            sys.exit(0)


def execute_commands_from_sql_file(db):
    """ Function that applies commands from a local .sql file to a specified database 'db'
    """
    cursor = db.cursor()

    # Split the .sql file up into a list of commands
    sql_commands = parse_sql_file()
    print("Parsed %d commands from .sql file"%len(sql_commands))

    # Split up the commands that are about CREATE TABLE, DROP TABLE, and the remaining commands
    create_table_commands  = list(filter(lambda command: command.startswith("CREATE TABLE"), sql_commands))
    drop_table_commands    = list(filter(lambda command: command.startswith("DROP TABLE"), sql_commands))
    remaining_sql_commands = list(filter(lambda command: not (command.startswith("CREATE TABLE") or command.startswith("DROP TABLE")), sql_commands))

    # First execute all DROP TABLE-commands
    print("\nDropping existing tables")
    start_time = dt.datetime.now()
    execute_drop_table_commands(cursor, drop_table_commands)
    print("\nSuccesfully dropped all existing tables, took %d.%d seconds"%((dt.datetime.now()-start_time).total_seconds(), (dt.datetime.now()-start_time).microseconds))

    # Then execute all CREATE TABLE-commands
    print("\nCreating new tables")
    start_time = dt.datetime.now()
    execution_order_of_table_creation_commands = execute_create_table_commands(cursor, create_table_commands)
    print("\nSuccesfully created all the tables, took %d.%d seconds"%((dt.datetime.now()-start_time).total_seconds(), (dt.datetime.now()-start_time).microseconds))

    # Order the remaining commands in the same order as in which the tables were created
    execution_order_of_table_names = list(map(lambda c: c.split('(')[0].split()[2].strip('`'), execution_order_of_table_creation_commands))
    ordered_remaining_sql_commands = order_remaining_commands_in_table_creation_order(execution_order_of_table_names, remaining_sql_commands)

    # Execute the remaining commands
    print("\nExecuting the remaining commands")
    start_time = dt.datetime.now()
    execute_commands(cursor, ordered_remaining_sql_commands)
    print("Succesfully executed all remaining commands, took %d.%d seconds"%((dt.datetime.now()-start_time).total_seconds(), (dt.datetime.now()-start_time).microseconds))

    # Commit the changes to the database
    db.commit()

    # Close the cursor
    cursor.close()


def main():
    overwrite_existing_database = False

    # Establish connection to the database
    conn = connect_db.establish_host_connection()
    db   = connect_db.establish_database_connection(conn, overwrite=overwrite_existing_database)
    print("Successfully connected to database '%s'"%db.database)

    # Parse the SQL file, then execute the commands in it
    execute_commands_from_sql_file(db)

    # Close the connection to the database
    db.close()
    conn.close()

if __name__ == '__main__':
    main()
