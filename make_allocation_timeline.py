# -*- coding: utf-8 -*-
"""
Created on Mon May  1 11:47:01 2023

@author: Marco
"""

import sys
import copy
import enum
import timeit
import numpy as np
import pandas as pd
import datetime as dt
from pprint import pprint
from collections import Counter
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
from lib.tracktime import TrackTime, TrackReport

import database.connect_db as connect_db
import database.query_db as query_db

pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 130)



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


# Relevant events according to the coupon lifecycle
class Event(enum.Enum):
    member_declined     = 0
    member_accepted     = 1
    member_let_expire   = 2 # i.e. after 3 days
    coupon_sent         = 3
    coupon_expired      = 4 # i.e. after 1 month

# # All the possible combinations of status + sub_status found in the data
# status_to_event = {('declined',  None):              Event.member_declined,
#                    ('declined', 'after_accepting'):  Event.member_declined,  # Discard the information that the member accepted initially
#                    ('expired',  'after_receiving'):  Event.member_let_expire,
#                    ('expired',  'after_accepting'):  Event.member_accepted,
#                    ('expired',  'not_redeemed'):     Event.member_accepted,
#                    ('redeemed',  None):              Event.member_accepted,
#                    ('redeemed', 'after_expiring'):   Event.member_accepted}


def main():
    TrackTime("Connect to db")
    conn = connect_db.establish_host_connection()
    db   = connect_db.establish_database_connection(conn)
    print("Successfully connected to database '%s'"%str(db.engine).split("/")[-1][:-1])

    TrackTime("Retrieve from db")
    result = query_db.retrieve_from_sql_db(db, 'filtered_coupons', 'filtered_issues', 'filtered_offers')
    filtered_coupons, filtered_issues, filtered_offers = result


    """ TODO:
    Get list of involved members, and compute
       - probability of letting a coupon expire
       - Subscribed categories
       - probability of accepting based on subscribed categories (coupon --> issue --> offer --> cat)
    """


    make_baseline_events_from_scratch  = True


    if make_baseline_events_from_scratch:
        events_df = make_events_timeline(filtered_coupons, filtered_issues, filtered_offers)
        print(events_df)

        print("")
        total_decay  = np.sum(events_df['coupon_count'][events_df['event'] == Event.coupon_expired])
        total_amount = np.sum(filtered_issues['amount'])
        print('Total number of coupons: %d'%total_amount)
        print('Total number of coupons never accepted: %d'%total_decay)
        print('Percentage of coupons never accepted: %.1f%%'%(total_decay/total_amount*100))

        TrackTime("to_csv")
        events_df.to_csv('./baseline_events.csv', index=False)
    else:
        TrackTime("read_csv")
        events_df = pd.read_csv('./baseline_events.csv', parse_dates=['at'])
        events_df['event'] = events_df['event'].apply(lambda event: Event[str(event).replace('Event.','')])
        events_df = events_df.convert_dtypes()
        # print(events_df)

    print("")
    TrackReport()

    # print_table_info(db)

    # Close the connection to the database
    db.close()
    conn.close()


def make_events_timeline(filtered_coupons, filtered_issues, filtered_offers):
    """
    Make timeline of events:
        1.  coupon sent to member_i
        2a. member accepted coupon
        2b. member declined coupon
        2c. member let the offered coupon expire without reacting (usually after 3 days)
        3.  coupon expired without anyone accepting (trash bin)
    """

    # Calculate the times it took for members to decline a coupon.
    declined_coupons = filtered_coupons[np.logical_and(filtered_coupons['status'] == 'declined', filtered_coupons['sub_status'] != "after_accepting")]
    declined_durations = declined_coupons['status_updated_at'] - declined_coupons['created_at']
    print("\nDeclined coupons:", len(declined_coupons), declined_durations.mean(), "\n")

    # Store the following information during the event-generation in a dict (not df) for optimization:
    issue_info = pd.DataFrame(index=filtered_issues['id'])
    issue_info['nr_accepted_so_far'] = 0
    issue_info['nr_issued_so_far']   = 0
    issue_info = issue_info.to_dict('index')

    # Set the index of the filtered_issues to the issue_id, for easy access during event-generation
    filtered_issues.index = filtered_issues['id']
    filtered_offers.index = filtered_offers['id']

    # To measure how well the function 'determine_coupon_checked_expiry_time' calculates the actual 'status_updated_at' time after a member letting a coupon expire
    incorrectly_predicted_created_after_expiry = 0
    events_list = []
    events_df = pd.DataFrame(columns=['event','at','member_id','coupon_id','issue_id','offer_id','coupon_count', 'category_id'])

    for i, (index, coupon_row) in enumerate(filtered_coupons.iterrows()):

        TrackTime("print")
        if i%1000 == 0:
            print("\rHandling coupon nr %d"%i,end='')

        TrackTime("ids_in_event")
        ids_in_event = [coupon_row['member_id'], coupon_row['id'], coupon_row['issue_id'], coupon_row['offer_id']]

        TrackTime("append")
        # Add the coupon sent event
        event = [Event.coupon_sent, coupon_row['created_at']] + ids_in_event + [1] + [filtered_offers.loc[coupon_row['offer_id'],'category_id']]
        events_list.append(event)

        TrackTime('check time')
        if issue_info[coupon_row['issue_id']]['nr_issued_so_far'] == 0:
            issue_sent_at = filtered_issues.loc[coupon_row['issue_id'],'sent_at']
            assert abs(coupon_row['created_at'] - issue_sent_at) < dt.timedelta(seconds=30), "Issue was sent at %s, but first coupon created at %s"%(coupon_row['created_at'], issue_sent_at)

        TrackTime("update issue_info")
        issue_info[coupon_row['issue_id']]['nr_issued_so_far'] += 1

        TrackTime("timestamp")
        member_response = coupon_row['member_response']
        if member_response == Event.member_let_expire:
            # If a member let the coupon expire or declined the coupon, the status of the coupon
            # has not been updated since this action. Therefore, we know the exact time of the action
            datetimestamp = coupon_row['status_updated_at']

            TrackTime("Check expected timestamp")
            checked_expiries = determine_coupon_checked_expiry_time(coupon_row['created_at'], coupon_row['accept_time'])
            if checked_expiries is None:
                checked_expiries = [datetimestamp]

            predicted_correctly = False
            for checked_expiry in checked_expiries:
                if abs(datetimestamp - checked_expiry) < dt.timedelta(minutes=10):
                    predicted_correctly = True
            if not predicted_correctly:
                incorrectly_predicted_created_after_expiry += 1
                # print("\nCoupon created at: %s, expired at: %s, status updated at:%s"%(coupon_row['created_at'], coupon_row['created_at'] + dt.timedelta(days=coupon_row['accept_time']), datetimestamp))

        elif member_response == Event.member_declined:
            datetimestamp = coupon_row['status_updated_at']
        else:
            # If the member accepts the coupon, this tatus will eventually be updated by i.e. redeemed
            # Therefore, make a random guess as to when the member accepted.
            # Assume the time for accepting has the same distribution as the time for declining
            # TODO: filter declined_durations <= accept_time
            TrackTime("random datetimestamp")
            # valid_declined_durations = declined_durations[declined_durations <= coupon_row['accept_time'].apply(lambda el: dt.timedelta(days=el))]
            index = np.random.randint(len(declined_durations))
            accepted_duration = declined_durations.iloc[index]
            datetimestamp = coupon_row['created_at'] + accepted_duration

        TrackTime("append")
        # Add the member-response event
        event = [member_response, datetimestamp] + ids_in_event + [1] + [filtered_offers.loc[coupon_row['offer_id'],'category_id']]
        events_list.append(event)

        TrackTime("update issue_info")
        if member_response == Event.member_accepted:
            # filtered_issues.loc[coupon_row['issue_id'],'nr_accepted_so_far'] += 1
            issue_info[coupon_row['issue_id']]['nr_accepted_so_far'] += 1

        # check of event coupon_expired heeft plaatsgevonden
        TrackTime("check coupon expiry")
        if member_response != Event.member_accepted:

            if issue_info[coupon_row['issue_id']]['nr_issued_so_far'] == filtered_issues.loc[coupon_row['issue_id'],'total_issued']:

                # This coupon_row is the last row in which a coupon with this issue_id appears
                decay_nr = filtered_issues.loc[coupon_row['issue_id'],'amount'] - issue_info[coupon_row['issue_id']]['nr_accepted_so_far']

                if decay_nr > 0:
                    TrackTime("append")
                    event = [Event.coupon_expired, filtered_issues.loc[coupon_row['issue_id'],'expires_at'], np.nan, np.nan, coupon_row['issue_id'], coupon_row['offer_id'], decay_nr, filtered_offers.loc[coupon_row['offer_id'],'category_id']]
                    events_list.append(event)


    print("\nincorrectly_predicted_created_after_expiry:", incorrectly_predicted_created_after_expiry)

    print("\n")
    events_df = pd.DataFrame(events_list, columns=['event','at','member_id','coupon_id','issue_id','offer_id','coupon_count', 'category_id'])

    # Sort events chronologically, and if two events have the same timestamp, sort on index as secondary constraint
    events_df['index'] = events_df.index
    events_df.sort_values(['at','index'], inplace=True)
    events_df.drop('index', axis=1, inplace=True)
    events_df = events_df.reset_index(drop=True)

    events_df = events_df.convert_dtypes()
    return events_df



def determine_coupon_checked_expiry_time(created_at, accept_time):
    """ Function that determines / predicts the time a coupon is sent
    to the next member once it expires (i.e. the member has not replied)
    """

    # Check if timestamp is as expected
    expired = created_at + dt.timedelta(days=accept_time)

    # Before 14 dec 2021, whether coupons had expired was checked every morning at 10am
    if expired.date() < dt.date(2021, 12, 14):
        ten_am = dt.time(10, 0, 0)
        if expired.time() <= ten_am:
            checked_expiry = dt.datetime.combine(expired.date(), ten_am)
        else:
            # If the coupon expired after 10am, the coupons expiry was only noticed the next morning at 10am
            checked_expiry = dt.datetime.combine(expired.date() + dt.timedelta(days=1), ten_am)

        # 29 okt 2021 was a day with (presumable) IT issues: ignore data
        if checked_expiry.date() == dt.date(2021, 10, 29):
            return None

        return [checked_expiry]
        # checked_expiry2 = checked_expiry + dt.timedelta(days=1)
        # return [checked_expiry, checked_expiry2]

    # After 14 dec 2021, whether coupons had expired was checked every hour, except between 8pm and 8am at night
    else:
        eight_pm = dt.time(20, 0, 0)
        eight_am = dt.time(8, 0, 0)
        if expired.time() > eight_pm:
            # If the coupon expired after 20:00, it is sent to the next person at 08:05 the next morning
            checked_expiry = (expired + dt.timedelta(days=1)).replace(hour=8,minute=5)
            return [checked_expiry]
            # checked_expiry2 = expired if expired.time() < dt.time(20, 5, 0) else checked_expiry
            # return [checked_expiry, checked_expiry2]

        elif expired.time() < eight_am:
            # If the coupon expired before 08:00, it is sent to the next person at 08:05 the same day
            checked_expiry = expired.replace(hour=8,minute=5)
            return [checked_expiry]

        else:
            # 28 sept 2022 was a day with (presumably) IT issues: ignore data
            if expired.date() == dt.date(2022, 9, 28):
                return None

            # If a coupon expired, it is usually sent to the next member at the next 5th minute of the hour
            checked_expiry = expired.replace(minute=5)
            if expired.minute <= 5:
                # When a coupon expired in the first 5 minutes of the hour, it is impossible to predict whether
                # the coupon was sent to the next member in that same hour, or more than an hour later
                # i.e. coupon expired at 15:03, it can both be sent to the next member at 15:05 or 16:05
                checked_expiry2 = checked_expiry + dt.timedelta(hours=1)
                return [checked_expiry, checked_expiry2]
            else:
                # If a coupon expired at 15:06, it is sent to the next member at 16:05
                checked_expiry = checked_expiry + dt.timedelta(hours=1)
                return [checked_expiry]




def print_table_info(db):
    tables = ['filtered_coupons', 'filtered_issues', 'offer', 'member']
    # all_tables = pd.read_sql_query("show tables", db).squeeze().values

    for table_name in tables:
        print("\n\nTABLE", table_name)
        query = "SELECT * FROM %s"%table_name
        df = pd.read_sql_query(query, db)

        for col in df.columns:
            unique_values = pd.unique(df[col].values)
            if len(unique_values) < 10:
                print("\t", col, type(unique_values[0]), len(unique_values), unique_values, unique_values[0])
            else:
                print("\t", col, type(unique_values[0]), len(unique_values), unique_values[0])



if __name__ == '__main__':
    main()