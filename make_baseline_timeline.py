# -*- coding: utf-8 -*-
"""
Created on Mon May  1 11:47:01 2023

@author: Marco
"""

import os
import sys
import copy
import numpy as np
import pandas as pd
import datetime as dt
from database.lib.tracktime import TrackTime, TrackReport

from IssueStreamSimulator import IssueStreamSimulator
import database.connect_db as connect_db
import database.query_db as query_db
from database.query_db import Event

pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 80)

np.random.seed(0)
EXPORT_FOLDER = './timelines/'


def main():
    TrackTime("Connect to db")
    conn = connect_db.establish_host_connection()
    db   = connect_db.establish_database_connection(conn)
    print("Successfully connected to database '%s'"%str(db.engine).split("/")[-1][:-1])

    TrackTime("Retrieve from db")
    result = query_db.retrieve_from_sql_db(db, 'filtered_coupons', 'filtered_issues', 'filtered_offers')
    filtered_coupons, filtered_issues, filtered_offers = result


    MAKE_BASELINE_EVENTS_FROM_SCRATCH = False
    if MAKE_BASELINE_EVENTS_FROM_SCRATCH or not os.path.exists(os.path.join(EXPORT_FOLDER, 'baseline_events.pkl')):
        events_df = make_events_timeline(filtered_coupons, filtered_issues, filtered_offers)
        events_df.to_pickle('./timelines/baseline_events.pkl')
    else:
        TrackTime("Read pickle")
        events_df = pd.read_pickle(os.path.join(EXPORT_FOLDER, 'baseline_events.pkl'))
        events_df['event'] = events_df['event'].apply(lambda event: Event[str(event).replace('Event.','')])
        events_df = events_df.convert_dtypes()

    print("")
    decayed = events_df[events_df['event'] == Event.coupon_expired]
    accepted = events_df[events_df['event'] == Event.member_accepted]
    total_amount = np.sum(filtered_issues['amount'])
    print('Total number of coupons: %d'%total_amount)
    print('Total number of coupons accepted: %d'%len(accepted))
    print('Total number of coupons never accepted: %d'%len(decayed))
    print('Percentage of coupons never accepted (trash bin): %.1f%%'%(len(decayed)/total_amount*100))

    print("")
    TrackReport()

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

    # Store the following information during the event-generation in a dict (not df) for optimization:
    issue_info = pd.DataFrame(index=filtered_issues['id'])
    issue_info['nr_accepted_so_far'] = 0
    issue_info['nr_issued_so_far']   = 0
    issue_info = issue_info.to_dict('index')

    # Set the index of the filtered_issues to the issue_id, for easy access during event-generation
    filtered_issues.index = filtered_issues['id']
    filtered_offers.index = filtered_offers['id']

    # To measure how well the function 'determine_coupon_checked_expiry_time' calculates the actual 'status_updated_at' time after a member letting a coupon expire
    incorrectly_predicted_expiries = 0
    events_list = []
    events_df = pd.DataFrame(columns=['event','at','member_id','coupon_id','coupon_follow_id','issue_id','offer_id', 'category_id'])

    set_issue_follow_ids = set()

    print("")
    TrackTime("Iterrows")
    for i, (index, coupon_row) in enumerate(filtered_coupons.iterrows()):
        if i%100 == 0:
            print("\rHandling coupon nr %d (%.1f%%)"%(i, 100*i/len(filtered_coupons)),end='')

        TrackTime("Define ids in event")
        ids_in_event = [coupon_row['member_id'], coupon_row['id'], coupon_row['coupon_follow_id'], coupon_row['issue_id'], coupon_row['offer_id']]

        TrackTime("Update issue_info")
        element = (coupon_row['issue_id'], coupon_row['coupon_follow_id'])
        set_issue_follow_ids.update(set([element]))
        issue_info[coupon_row['issue_id']]['nr_issued_so_far'] += 1

        TrackTime("Append events")
        # Add the coupon sent event
        event = [Event.coupon_sent, coupon_row['created_at']] + ids_in_event + [filtered_offers.loc[coupon_row['offer_id'],'category_id']]
        events_list.append(event)

        TrackTime("Time of member response")
        member_response = coupon_row['member_response']
        if member_response == Event.member_let_expire:
            # If a member let the coupon expire or declined the coupon, the status of the coupon
            # has not been updated since this action. Therefore, we know the exact time of the action
            datetimestamp = coupon_row['status_updated_at']

            checked_expiries = IssueStreamSimulator.determine_coupon_checked_expiry_time(coupon_row['created_at'], coupon_row['accept_time'], check=True)
            if checked_expiries is None:
                checked_expiries = [datetimestamp]

            predicted_correctly = False
            for checked_expiry in checked_expiries:
                if abs(datetimestamp - checked_expiry) < dt.timedelta(minutes=10):
                    predicted_correctly = True
            if not predicted_correctly:
                incorrectly_predicted_expiries += 1
                # print("\nCoupon created at: %s, expired at: %s, status updated at:%s"%(coupon_row['created_at'], coupon_row['created_at'] + dt.timedelta(days=coupon_row['accept_time']), datetimestamp))

        elif member_response == Event.member_declined:
            datetimestamp = coupon_row['status_updated_at']
        else:
            # If the member accepts the coupon, this tatus will eventually be updated by i.e. redeemed
            # Therefore, make a random guess as to when the member accepted.
            # Assume the time for accepting has the same distribution as the time for declining
            index = np.random.randint(len(declined_durations))
            accepted_duration = declined_durations.iloc[index]
            datetimestamp = coupon_row['created_at'] + accepted_duration

        TrackTime("Append events")
        # Add the member-response event
        event = [member_response, datetimestamp] + ids_in_event + [filtered_offers.loc[coupon_row['offer_id'],'category_id']]
        events_list.append(event)

        TrackTime("Update issue_info")
        if member_response == Event.member_accepted:
            issue_info[coupon_row['issue_id']]['nr_accepted_so_far'] += 1

        # check of event coupon_expired heeft plaatsgevonden
        TrackTime("Check coupon expiry")
        if issue_info[coupon_row['issue_id']]['nr_issued_so_far'] == filtered_issues.loc[coupon_row['issue_id'],'total_issued']:

            # This coupon_row is the last row in which a coupon with this issue_id appears
            decay_nr = filtered_issues.loc[coupon_row['issue_id'],'amount'] - issue_info[coupon_row['issue_id']]['nr_accepted_so_far']
            issue_coupons = filtered_coupons[filtered_coupons['issue_id'] == coupon_row['issue_id']] 
            issue_coupons_accepted = issue_coupons[issue_coupons['member_response'] == Event.member_accepted]
            accepted_follow_ids = set(issue_coupons_accepted['coupon_follow_id'])
            non_accepted_follow_ids = set(np.arange(1, 1+filtered_issues.loc[coupon_row['issue_id'],'amount'])) - accepted_follow_ids
            assert len(non_accepted_follow_ids) == decay_nr
            assert set(issue_coupons['coupon_follow_id']) == accepted_follow_ids.union(non_accepted_follow_ids)

            if decay_nr > 0:
                TrackTime("Append events")
                event = [Event.coupon_expired, filtered_issues.loc[coupon_row['issue_id'],'expires_at'], np.nan, np.nan, coupon_row['coupon_follow_id'], coupon_row['issue_id'], coupon_row['offer_id'], filtered_offers.loc[coupon_row['offer_id'],'category_id']]
                events = []
                for follow_id in non_accepted_follow_ids:
                    event[4] = follow_id
                    events.append(copy.copy(event))
                events_list.extend(events)

        TrackTime("Iterrows")

    print("\rHandling coupon nr %d (%.1f%%)"%(i, 100))
    print("Incorrectly predicted coupon expiry times: %d (%.2f%%)"%(incorrectly_predicted_expiries, 100*incorrectly_predicted_expiries/len(filtered_coupons)))

    # Turns events list into a dataframe
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


if __name__ == '__main__':
    main()