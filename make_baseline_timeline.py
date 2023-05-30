# -*- coding: utf-8 -*-
"""
Created on Mon May  1 11:47:01 2023

@author: Marco
"""

import sys
import copy
import enum
import numpy as np
import pandas as pd
import datetime as dt
from database.lib.tracktime import TrackTime, TrackReport

from coupon_stream_simulator import determine_coupon_checked_expiry_time
import database.connect_db as connect_db
import database.query_db as query_db
Event = query_db.Event

pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 130)


def main():
    TrackTime("Connect to db")
    conn = connect_db.establish_host_connection()
    db   = connect_db.establish_database_connection(conn)
    print("Successfully connected to database '%s'"%str(db.engine).split("/")[-1][:-1])

    TrackTime("Retrieve from db")
    result = query_db.retrieve_from_sql_db(db, 'filtered_coupons', 'filtered_issues', 'filtered_offers')
    filtered_coupons, filtered_issues, filtered_offers = result


    make_baseline_events_from_scratch  = True


    if make_baseline_events_from_scratch:
        events_df = make_events_timeline(filtered_coupons, filtered_issues, filtered_offers)
        print(events_df)

        print("")
        total_decay  = np.sum(events_df['event'] == Event.coupon_expired)
        total_amount = np.sum(filtered_issues['amount'])
        print('Total number of coupons: %d'%total_amount)
        print('Total number of coupons never accepted: %d'%total_decay)
        print('Percentage of coupons never accepted: %.1f%%'%(total_decay/total_amount*100))

        TrackTime("to_csv")
        events_df.to_csv('./timelines/baseline_events.csv', index=False)
        events_df.to_pickle('./timelines/baseline_events.pkl')
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
    events_df = pd.DataFrame(columns=['event','at','member_id','coupon_id','coupon_follow_id','issue_id','offer_id', 'category_id'])

    for i, (index, coupon_row) in enumerate(filtered_coupons.iterrows()):

        TrackTime("print")
        if i%1000 == 0:
            print("\rHandling coupon nr %d"%i,end='')

        TrackTime("ids_in_event")
        ids_in_event = [coupon_row['member_id'], coupon_row['id'], coupon_row['coupon_follow_id'], coupon_row['issue_id'], coupon_row['offer_id']]

        TrackTime("append")
        # Add the coupon sent event
        event = [Event.coupon_sent, coupon_row['created_at']] + ids_in_event + [filtered_offers.loc[coupon_row['offer_id'],'category_id']]
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
            checked_expiries = determine_coupon_checked_expiry_time(coupon_row['created_at'], coupon_row['accept_time'], check=True)
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
        event = [member_response, datetimestamp] + ids_in_event + [filtered_offers.loc[coupon_row['offer_id'],'category_id']]
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
                issue_coupons = filtered_coupons[filtered_coupons['issue_id'] == coupon_row['issue_id']] 
                issue_coupons_accepted = issue_coupons[issue_coupons['member_response'] == Event.member_accepted]
                accepted_follow_ids = set(issue_coupons_accepted['coupon_follow_id'])
                non_accepted_follow_ids = set(np.arange(filtered_issues.loc[coupon_row['issue_id'],'amount'])) - accepted_follow_ids
                assert len(non_accepted_follow_ids) == decay_nr

                if decay_nr > 0:
                    TrackTime("append")
                    event = [Event.coupon_expired, filtered_issues.loc[coupon_row['issue_id'],'expires_at'], np.nan, np.nan, coupon_row['coupon_follow_id'], coupon_row['issue_id'], coupon_row['offer_id'], filtered_offers.loc[coupon_row['offer_id'],'category_id']]
                    events = []
                    for follow_id in non_accepted_follow_ids:
                        event[4] = follow_id
                        events.append(copy.copy(event))
                    events_list.extend(events)


    print("\nincorrectly_predicted_created_after_expiry:", incorrectly_predicted_created_after_expiry)

    print("\n")
    events_df = pd.DataFrame(events_list, columns=events_df.columns)

    # Assign new unique coupon follow ids
    unique_coupon_follow_ids = events_df.groupby(['issue_id','coupon_follow_id']).aggregate(test=('coupon_id','count')).reset_index()
    unique_coupon_follow_ids['new_coupon_follow_id'] = np.arange(len(unique_coupon_follow_ids))
    unique_coupon_follow_ids = unique_coupon_follow_ids.drop(columns=['test'])

    # Merge into events dataframe
    events_df = events_df.merge(unique_coupon_follow_ids, on=['issue_id','coupon_follow_id'])
    events_df = events_df.drop(columns=['coupon_follow_id'])
    events_df = events_df.rename(columns={'new_coupon_follow_id':'coupon_follow_id'})

    # Reorder columns
    events_df = events_df[['event','at','coupon_id','coupon_follow_id','issue_id','offer_id','member_id','category_id']]

    # Sort events chronologically, and if two events have the same timestamp, sort on index as secondary constraint
    events_df['index'] = events_df.index
    events_df.sort_values(['at','index'], inplace=True)
    events_df.drop('index', axis=1, inplace=True)
    events_df = events_df.reset_index(drop=True)

    events_df = events_df.convert_dtypes()
    return events_df






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