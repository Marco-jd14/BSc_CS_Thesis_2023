# -*- coding: utf-8 -*-
"""
Created on Sat May 13 11:25:27 2023

@author: Marco
"""
import sys
import enum
import sqlalchemy
import pandas as pd


# Relevant events according to the coupon lifecycle
class Event(enum.Enum):
    member_accepted     = 0
    member_declined     = 1
    member_let_expire   = 2 # i.e. after 3 days
    coupon_available    = 3
    coupon_sent         = 4
    coupon_expired      = 5 # i.e. after 1 month

    def __lt__(self, other):
        self.value < other.value


def retrieve_from_sql_db(db, *table_names):
    """ Function to retrieve any specified table-names from the database 
    Returns a tuple of tables, to be unpacked
    """
    tables = []

    for table_name in table_names:
        print("Retrieving table '%s' from database"%table_name)

        try:
            query = sqlalchemy.text("select * from %s"%table_name)
            table = pd.read_sql_query(query, db)
        except sqlalchemy.exc.SQLAlchemyError as e:
            errno = int(str(e.orig).split()[0].strip("(,"))
            if errno == 1146:
                print("\nERROR: Table '%s' does not exist in database '%s'"%(table_name, str(db.engine).split("/")[-1][:-1]))
                db.close()
                sys.exit(0)
            else:
                raise

        # Translate str back to Event objects for easy comparison later on
        if 'member_response' in table.columns:
            table['member_response'] = table['member_response'].apply(lambda event: Event[str(event)])

        tables.append(table)

    return tuple(tables)


def save_df_to_sql(db, name_to_table_mapping):
    """ Function to write a dictionary of tables to the database
    The dictionary is a mapping from table-name (to save the table under) to dataframe
    """
    for table_name, table in name_to_table_mapping.items():
        print("Writing table '%s' to database"%table_name)

        # Translate Event objects to str for exporting to SQL
        if 'member_response' in table.columns:
           table['member_response'] = table['member_response'].apply(lambda event: str(event).replace('Event.',''))

        db.execute(sqlalchemy.text("DROP TABLE IF EXISTS `%s`;"%table_name))
        table.to_sql(table_name, db, index=False)



# Example usage:
if __name__ == '__main__':
    import connect_db

    # First establish a database connection
    conn = connect_db.establish_host_connection()
    db   = connect_db.establish_database_connection(conn, overwrite=False)
    print("Successfully connected to database '%s'"%str(db.engine).split("/")[-1][:-1])

    # How to retrieve tables from database
    result = retrieve_from_sql_db(db, 'issue', 'offer')
    all_issues, all_offers = result  # Unpack the tuple of tables

    # How to write tables back to database
    name_to_table_mapping = {'my_new_table_name1':all_issues, 'my_new_table_name2':all_offers}
    save_df_to_sql(db, name_to_table_mapping)

    # Leaving the database in the same state as it was
    db.execute(sqlalchemy.text("DROP TABLE `my_new_table_name1`;"))
    db.execute(sqlalchemy.text("DROP TABLE `my_new_table_name2`;"))

    # Close connection to the database when you're done with it
    db.close()
    conn.close()