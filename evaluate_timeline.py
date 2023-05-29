# -*- coding: utf-8 -*-
"""
Created on Mon May 29 20:12:40 2023

@author: Marco
"""
import sys
import copy
import enum
import traceback
import numpy as np
import pandas as pd
import datetime as dt
from pprint import pprint
from collections import Counter
import matplotlib.pyplot as plt
from database.lib.tracktime import TrackTime, TrackReport

import database.connect_db as connect_db
import database.query_db as query_db
Event = query_db.Event

def main():
    TrackTime("Connect to db")
    conn = connect_db.establish_host_connection()
    db   = connect_db.establish_database_connection(conn)
    print("Successfully connected to database '%s'"%str(db.engine).split("/")[-1][:-1])

    TrackTime("Read Excel")
    events_list_path = "./timelines/events_list_2023.05.29-20.03.37.xlsx"
    events_df = pd.read_excel(events_list_path, "events", index_col="Unnamed: 0")

    TrackTime("Evaluate timeline")
    evaluate_timeline(events_df)

    db.close()
    TrackReport()


def evaluate_timeline(events_df):
    print(events_df)


if __name__ == '__main__':
    main()
