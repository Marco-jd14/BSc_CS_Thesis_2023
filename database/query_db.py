# -*- coding: utf-8 -*-
"""
Created on Mon May  1 11:47:01 2023

@author: Marco
"""

import sys
import copy
import enum
import timeit
import connect_db
import numpy as np
import pandas as pd
import datetime as dt
from pprint import pprint
from collections import Counter
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
from lib.tracktime import TrackTime, TrackReport

pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 120)

def datetime_range(start_date, end_date, delta):
    result = []
    nxt = start_date
    delta = relativedelta(**delta)
    while nxt <= end_date:
        result.append(nxt)
        nxt += delta
    result.append(end_date)
    return result


def plot_created_coupons(df):
    created = df['created_at']
    created = created.sort_values(ascending=True)
    assert created.is_monotonic

    start_date = created.iloc[0] - dt.timedelta(seconds=1)
    end_date   = created.iloc[-1] + dt.timedelta(seconds=1)
    delta = {'days':10}
    intervals = datetime_range(start_date, end_date, delta)

    res = created.groupby(pd.cut(created, intervals)).count()
    res.name = "nr_coupons_per_interval"
    res = res.reset_index()

    interval_ends = list(map(lambda interval: interval.right, res.created_at))
    plt.plot(interval_ends, res.nr_coupons_per_interval)



relevant_columns = {'coupon': ['id', 'member_id', 'created_at', 'status',
                               'sub_status', 'issue_id', 'redeem_till',
                               'redeem_type', 'accept_time', 'offer_id', 'type'],
                    'issue': ['id', 'offer_id', 'sent_at', 'amount', # same as issue_rule.amount if rule_id is not null
                              'decay_count', 'expires_at', 'sent_acceptation_ratio',
                              'total_issued', 'aborted_at'],
                    'offer': ['id', 'offer_id', 'category_id', 'created_at',
                              'redeem_till', 'redeem_type', 'accept_time', 
                              'type', 'total_issued']
                    }

class Event(enum.Enum):
    member_declined     = 0
    member_accepted     = 1
    member_let_expire   = 2
    coupon_sent         = 3
    coupon_expired      = 4

status_to_event = {('declined', None):              Event.member_declined,
                   ('declined', 'after_accepting'): Event.member_accepted,
                   ('expired', 'after_accepting'):  Event.member_accepted,
                   ('expired', 'after_receiving'):  Event.member_let_expire,
                   ('expired', 'not_redeemed'):     Event.member_accepted,
                   ('redeemed', None):              Event.member_accepted,
                   ('redeemed', 'after_expiring'):  Event.member_accepted}

def main():
    global relevant_columns
    conn = connect_db.establish_host_connection()
    db   = connect_db.establish_database_connection(conn)
    print("Successfully connected to database '%s'"%str(db.engine).split("/")[-1][:-1])


    """
    Data prep:
        - filter out lottery coupons, issues (NaN), and offers (if they have exclusively lottery coupons)
        - filter out negative amount issues
        - do not look at total_issued from issue table
        - filter on certain horizon


    Choose fixed horizon [T_start, T_end]
    Filter out lottery coupons
    Make timeline of events:
       - coupon sent to member_i ['created_at']
       - member accepted coupon [random('created_at',+=3)]
       - member declined coupon [random('created_at',+=3)]
       - coupon expired without member reacting (3 days)
       - coupon expired without anyone accepting (trash bin)
    Get list of involved members, and compute
       - probability of letting a coupon expire
       - Subscribed categories
       - probability of accepting based on subscribed categories (coupon --> issue --> offer --> cat)
    
    """


    query = "select * from coupon"
    all_coupons = pd.read_sql_query(query, db)

    query = "select * from offer"
    all_offers = pd.read_sql_query(query, db)

    query = "select * from issue"
    all_issues = pd.read_sql_query(query, db)


    print("\nBefore filtering:")
    print("nr coupons:", len(all_coupons))
    print("nr issues:", len(all_issues))
    print("nr offers:", len(all_offers))

    non_lottery_coupons = all_coupons[all_coupons['type'] == 'Coupon']

    date_before = dt.datetime(2022, 12, 31)
    date_after  = dt.datetime(2021, 1, 1)
    horizon_coupons = non_lottery_coupons[non_lottery_coupons['created_at'] <= date_before]
    horizon_coupons = horizon_coupons[horizon_coupons['created_at'] >= date_after]

    filtered_coupons = horizon_coupons
    filtered_out_coupons = all_coupons[~all_coupons['id'].isin(filtered_coupons['id'])]
    assert len(filtered_coupons) + len(filtered_out_coupons) == len(all_coupons)

    issues_in_filtered_coupons     = all_issues[all_issues['id'].isin(filtered_coupons['issue_id'])]
    issues_in_filtered_out_coupons = all_issues[all_issues['id'].isin(filtered_out_coupons['issue_id'])]
    issues_with_all_coupons_in_filtered_coupons = issues_in_filtered_coupons[~issues_in_filtered_coupons['id'].isin(issues_in_filtered_out_coupons['id'])]

    filtered_coupons = filtered_coupons[filtered_coupons['issue_id'].isin(issues_with_all_coupons_in_filtered_coupons['id'])]
    filtered_out_coupons = all_coupons[~all_coupons['id'].isin(filtered_coupons['id'])]
    assert len(all_issues[all_issues['id'].isin(filtered_out_coupons['issue_id'])]) == len(issues_in_filtered_out_coupons), "The number of filtered out coupons has somehow changed"

    offers_in_filtered_coupons = all_offers[all_offers['id'].isin(filtered_coupons['offer_id'])]

    filtered_issues = issues_with_all_coupons_in_filtered_coupons
    filtered_offers = offers_in_filtered_coupons
    
    filtered_coupons = copy.copy(filtered_coupons)
    filtered_issues  = copy.copy(filtered_issues)
    filtered_offers  = copy.copy(filtered_offers)
    

    print("\nAfter filtering:")
    print("nr coupons:", len(filtered_coupons))
    print("nr issues:", len(filtered_issues))
    print("nr offers:", len(filtered_offers))

    # plot_created_coupons(filtered_coupons)

    # DATA CHECKS:
    # Only offers of type standard and coupons of type Coupon
    assert np.all(filtered_offers['type'] == 'standard'), "Please filter out any offers of type Lottery"
    assert np.all(filtered_coupons['type'] == 'Coupon'), "Please filter out any coupons of type LotteryCoupon"

    # Only complete issues
    all_coupons_from_filtered_issues = all_coupons[all_coupons['issue_id'].isin(filtered_issues['id'])]
    assert np.all(all_coupons_from_filtered_issues['id'].isin(filtered_coupons['id'])), "Please make sure that each coupon belonging to a filtered issue is present in the filtered_coupons table"

    # Every offer_id and issue_id in the coupon table exists in the offer and issue table
    assert np.all(filtered_coupons['issue_id'].isin(filtered_issues['id'])), "Some coupons have invalid issue-ids: the issue-ids cannot be found in the issues table"
    assert np.all(filtered_coupons['offer_id'].isin(filtered_offers['id'])), "Some coupons have invalid offer-ids: the offer-ids cannot be found in the offers table"

    # No issues with ammount <= 0
    issues_in_coupons_table = filtered_issues[filtered_issues['id'].isin(filtered_coupons['issue_id'])]
    assert np.all(issues_in_coupons_table['amount'] > 0), "Please filter out any coupons that belong to issues with an 'amount' of 0 or less"

    # Combinations of status & sub_status:
    unique_sub_status = set(list(map(tuple, filtered_coupons[['status', 'sub_status']].values)))
    assert set(status_to_event.keys()) == unique_sub_status, 'Every combination of status + sub_status must be translated into an event'
    # pre-compute member responses:
    filtered_coupons['member_response'] = filtered_coupons.apply(lambda row: status_to_event[(row['status'], row['sub_status'])], axis=1)

    # Convert 'issue_id' to an int. It was of dtype float before, to be able to support NaN
    filtered_coupons['issue_id'] = filtered_coupons['issue_id'].astype(int)

    # Update the 'total_issued' column from the issue table
    coupon_counts = filtered_coupons[['issue_id']].groupby('issue_id').aggregate(total_issued=('issue_id','count')).reset_index()
    issues_not_in_coupons_table = filtered_issues[~filtered_issues['id'].isin(coupon_counts['issue_id'])]
    # To be able to join all issue_ids, make sure that the set of issue_ids is the same in both tables
    zero_coupon_counts = pd.DataFrame(issues_not_in_coupons_table['id'].values, columns=['issue_id'])
    zero_coupon_counts['total_issued'] = 0
    coupon_counts = pd.concat([coupon_counts, zero_coupon_counts])
    assert set(coupon_counts['issue_id']) == set(filtered_issues['id'])
    # Drop the outdated 'total_issued' column, and merge tables to update the total_issued' column
    filtered_issues.drop(['total_issued'], axis=1, inplace=True)
    filtered_issues = pd.merge(filtered_issues, coupon_counts, left_on='id', right_on='issue_id')


    tables = ['coupon', 'issue', 'offer', 'member']
    # tables = pd.read_sql_query("show tables", db).squeeze().values

    for table_name in tables:
        print("\n\nTABLE", table_name)
        # query = "SELECT * FROM offer where type='standard';"
        query = "SELECT * FROM %s"%table_name
        result = db.execute(query)
        df = pd.DataFrame(result.fetchall())

        for col in df.columns:
            res = pd.unique(df[col].values)
            if len(res) < 10:
                print("\t", col, type(res[0]), len(res), res, res[0])
            else:
                print("\t", col, type(res[0]), len(res), res[0])

    # result = db.execute(query)
    # df = pd.DataFrame(result.fetchall())
    # df = pd.read_sql_query(query, db)
    # print(df)

    # Close the connection to the database
    db.close()
    conn.close()


if __name__ == '__main__':
    main()