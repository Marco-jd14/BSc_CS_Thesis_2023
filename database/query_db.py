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
pd.set_option('display.width', 150)

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


def determine_coupon_checked_expiry_time(created_at, accept_time):

    # Check if timestamp is as expected
    expired = created_at + dt.timedelta(days=accept_time)

    if expired.date() < dt.date(2021, 12, 14):
        ten_am = dt.time(10, 0, 0)
        if expired.time() <= ten_am:
            checked_expiry = dt.datetime.combine(expired.date(), ten_am)
        else:
            checked_expiry = dt.datetime.combine(expired.date() + dt.timedelta(days=1), ten_am)

        if checked_expiry.date() == dt.date(2021, 10, 29):
            return None

        return [checked_expiry]
        # checked_expiry2 = checked_expiry + dt.timedelta(days=1)
        # return [checked_expiry, checked_expiry2]

    else:
        eight_pm = dt.time(20, 0, 0)
        eight_am = dt.time(8, 0, 0)
        if expired.time() > eight_pm:
            checked_expiry = (expired + dt.timedelta(days=1)).replace(hour=8,minute=5)
            return [checked_expiry]
            # checked_expiry2 = expired if expired.time() < dt.time(20, 5, 0) else checked_expiry
            # return [checked_expiry, checked_expiry2]

        elif expired.time() < eight_am:
            checked_expiry = expired.replace(hour=8,minute=5)
            return [checked_expiry]

        else:
            if expired.date() == dt.date(2022, 9, 28):
                return None

            checked_expiry = expired.replace(minute=5)
            if expired.minute <= 5:
                checked_expiry2 = checked_expiry + dt.timedelta(hours=1)
                return [checked_expiry, checked_expiry2]
            else:
                checked_expiry = checked_expiry + dt.timedelta(hours=1)
                return [checked_expiry]



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
                   ('declined', 'after_accepting'): Event.member_declined,
                   ('expired', 'after_accepting'):  Event.member_accepted,
                   ('expired', 'after_receiving'):  Event.member_let_expire,
                   ('expired', 'not_redeemed'):     Event.member_accepted,
                   ('redeemed', None):              Event.member_accepted,
                   ('redeemed', 'after_expiring'):  Event.member_accepted}


def filter_coupons_issues_and_offers(all_coupons, all_issues, all_offers):
    # Filter non-lottery coupons
    non_lottery_coupons = all_coupons[all_coupons['type'] == 'Coupon']

    # Filter on coupons within a certain horizon
    date_before = dt.datetime(2022, 12, 31)
    date_after  = dt.datetime(2021, 1, 1)
    horizon_coupons = non_lottery_coupons[non_lottery_coupons['created_at'] <= date_before]
    horizon_coupons = horizon_coupons[horizon_coupons['created_at'] >= date_after]

    # Consistency check
    filtered_coupons = horizon_coupons
    filtered_out_coupons = all_coupons[~all_coupons['id'].isin(filtered_coupons['id'])]
    assert len(filtered_coupons) + len(filtered_out_coupons) == len(all_coupons)

    # Compute issues that do not have any coupons that got filtered out
    issues_in_filtered_coupons     = all_issues[all_issues['id'].isin(filtered_coupons['issue_id'])]
    issues_in_filtered_out_coupons = all_issues[all_issues['id'].isin(filtered_out_coupons['issue_id'])]
    issues_with_all_coupons_in_filtered_coupons = issues_in_filtered_coupons[~issues_in_filtered_coupons['id'].isin(issues_in_filtered_out_coupons['id'])]

    # Only keep coupons that belong to issues that do not have any coupons that got filtered out
    filtered_coupons = filtered_coupons[filtered_coupons['issue_id'].isin(issues_with_all_coupons_in_filtered_coupons['id'])]
    filtered_out_coupons = all_coupons[~all_coupons['id'].isin(filtered_coupons['id'])]
    # Check if we indeed did not remove any extra issues by filtering out extra coupons
    assert len(all_issues[all_issues['id'].isin(filtered_out_coupons['issue_id'])]) == len(issues_in_filtered_out_coupons), "The number of filtered out issues has somehow changed"

    # Filter out offers that do not appear in the coupons table
    offers_in_filtered_coupons = all_offers[all_offers['id'].isin(filtered_coupons['offer_id'])]

    # Define the final filtered_issues and filtered_offers tables
    filtered_issues = issues_with_all_coupons_in_filtered_coupons
    filtered_offers = offers_in_filtered_coupons

    # Make a copy of the dataframes, such that in the future we do not risk changing all_coupons, all_issues, or all_offers
    filtered_coupons = copy.copy(filtered_coupons)
    filtered_issues  = copy.copy(filtered_issues)
    filtered_offers  = copy.copy(filtered_offers)
    return filtered_coupons, filtered_issues, filtered_offers


def perform_data_checks(all_coupons, filtered_coupons, filtered_issues, filtered_offers):
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
    # To be able to join all issue_ids, make sure that the set of issue_ids is the same in both tables
    issues_not_in_coupons_table = filtered_issues[~filtered_issues['id'].isin(coupon_counts['issue_id'])]
    zero_coupon_counts = pd.DataFrame(issues_not_in_coupons_table['id'].values, columns=['issue_id'])
    zero_coupon_counts['total_issued'] = 0
    coupon_counts = pd.concat([coupon_counts, zero_coupon_counts])
    # Drop the outdated 'total_issued' column, and merge tables to update the 'total_issued' column
    filtered_issues.drop(['total_issued'], axis=1, inplace=True)
    # Join the tables
    assert set(coupon_counts['issue_id']) == set(filtered_issues['id'])
    filtered_issues = pd.merge(filtered_issues, coupon_counts, left_on='id', right_on='issue_id')

    # Make new column 'total_accepted' in the issue table
    accepted_coupons = filtered_coupons[filtered_coupons['member_response'] == Event.member_accepted]
    nr_accepted_coupons_per_issue = accepted_coupons[['issue_id']].groupby('issue_id').aggregate(total_accepted=('issue_id','count')).reset_index()
    # To be able to join all issue_ids, make sure that the set of issue_ids is the same in both tables
    issues_not_in_accepted_table = filtered_issues[~filtered_issues['id'].isin(nr_accepted_coupons_per_issue['issue_id'])]
    issues_with_zero_accepted_coupons = pd.DataFrame(issues_not_in_accepted_table['id'].values, columns=['issue_id'])
    issues_with_zero_accepted_coupons['total_accepted'] = 0
    nr_accepted_coupons_per_issue = pd.concat([nr_accepted_coupons_per_issue, issues_with_zero_accepted_coupons])
    # Join the tables
    assert set(nr_accepted_coupons_per_issue['issue_id']) == set(filtered_issues['id'])
    filtered_issues = pd.merge(filtered_issues, nr_accepted_coupons_per_issue, left_on='id', right_on='issue_id')

    # Calculate issues with negative decay (which should be impossible)
    filtered_issues['my_decay'] = filtered_issues['amount'] - filtered_issues['total_accepted']
    issues_with_negative_decay = filtered_issues[filtered_issues['my_decay'] < 0]
    # Filter out issues and coupons with negative decay
    filtered_coupons = filtered_coupons[~filtered_coupons['issue_id'].isin(issues_with_negative_decay['id'])]
    filtered_issues = filtered_issues[~filtered_issues['id'].isin(issues_with_negative_decay['id'])]
    # Check if each 'amount' >= number of accepted coupons
    assert np.all(filtered_issues['total_accepted'] <= filtered_issues['amount'])

    return filtered_coupons, filtered_issues


def make_events_timeline(filtered_coupons, filtered_issues):
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

    # To measure how well the function 'determine_coupon_checked_expiry_time' calculates the actual 'status_updated_at' time after a member letting a coupon expire
    incorrectly_predicted_created_after_expiry = 0
    events_list = []
    events_df = pd.DataFrame(columns=['event','at','member_id','coupon_id','issue_id','offer_id','coupon_count'])

    for i, (index, coupon_row) in enumerate(filtered_coupons.iterrows()):

        TrackTime("print")
        if i%1000 == 0:
            print("\rHandling coupon nr %d"%i,end='')

        TrackTime("append")
        ids_in_event = [coupon_row['member_id'], coupon_row['id'], coupon_row['issue_id'], coupon_row['offer_id']]

        # Add the coupon sent event
        event = [Event.coupon_sent, coupon_row['created_at']] + ids_in_event + [1]
        events_list.append(event)

        TrackTime('check time')
        if issue_info[coupon_row['issue_id']]['nr_issued_so_far'] == 0:
            issue_sent_at = filtered_issues.loc[coupon_row['issue_id'],'sent_at']
            assert abs(coupon_row['created_at'] - issue_sent_at) < dt.timedelta(seconds=30), "Issue was sent at %s, but first coupon created at %s"%(coupon_row['created_at'], issue_sent_at)

        TrackTime("update issue_info")
        issue_info[coupon_row['issue_id']]['nr_issued_so_far'] += 1

        TrackTime("timestamp")
        member_response = coupon_row['member_response'] # status_to_event[(status, sub_status)]
        if member_response == Event.member_let_expire:
            # If a member let the coupon expire or declined the coupon, the status of the coupon
            # has not been updated since this action. Therefore, we know the exact time of the action
            datetimestamp = coupon_row['status_updated_at']
            TrackTime("Check expected timestamp")
            checked_expiries = determine_coupon_checked_expiry_time(coupon_row['created_at'], coupon_row['accept_time'])
            if checked_expiries is None:
                checked_expiries = [datetimestamp]

            predicted = False
            for checked_expiry in checked_expiries:
                if abs(datetimestamp - checked_expiry) < dt.timedelta(minutes=10):
                    predicted = True

            if not predicted:
                incorrectly_predicted_created_after_expiry += 1
                # print("\nCoupon created at: %s, expired at: %s, status updated at:%s"%(coupon_row['created_at'], coupon_row['created_at'] + dt.timedelta(days=coupon_row['accept_time']), datetimestamp))

        elif member_response == Event.member_declined:
            datetimestamp = coupon_row['status_updated_at']
            # TODO: check if next coupon is sent at same time as declined
        else:
            # If the member accepts the coupon, this tatus will eventually be updated by i.e. redeemed
            # Therefore, make a random guess as to when the member accepted.
            # Assume the time for accepting has the same distribution as the time for declining

            # TODO: filter declined_durations <= accept_time
            TrackTime("random datetimestamp")
            index = np.random.randint(len(declined_durations))
            accepted_duration = declined_durations.iloc[index]
            datetimestamp = coupon_row['created_at'] + accepted_duration

        TrackTime("append")
        # Add the member-response event
        event = [member_response, datetimestamp] + ids_in_event + [1]
        events_list.append(event)

        TrackTime("update issue_info")
        if member_response == Event.member_accepted:
            # filtered_issues.loc[coupon_row['issue_id'],'nr_accepted_so_far'] += 1
            issue_info[coupon_row['issue_id']]['nr_accepted_so_far'] += 1

        # check of event coupon_expired heeft plaatsgevonden
        TrackTime("check coupon expiry")
        if member_response != Event.member_accepted:

            if issue_info[coupon_row['issue_id']]['nr_issued_so_far'] == filtered_issues.loc[coupon_row['issue_id'], 'total_issued']:

                # This coupon_row is the last row in which a coupon with this issue_id appears
                decay_nr = filtered_issues.loc[coupon_row['issue_id'],'amount'] - issue_info[coupon_row['issue_id']]['nr_accepted_so_far']

                if decay_nr > 0:
                    TrackTime("append")
                    event = [Event.coupon_expired, filtered_issues.loc[coupon_row['issue_id'],'expires_at'], np.nan, np.nan, coupon_row['issue_id'], coupon_row['offer_id'], decay_nr]
                    events_list.append(event)

    print("\nincorrectly_predicted_created_after_expiry:", incorrectly_predicted_created_after_expiry)

    print("\n")
    events_df = pd.DataFrame(events_list, columns=['event','at','member_id','coupon_id','issue_id','offer_id','coupon_count'])

    # Sort events chronologically, and if two events have the same timestamp, sort on index as secondary constraint
    events_df['index'] = events_df.index
    events_df.sort_values(['at','index'], inplace=True)
    events_df.drop('index', axis=1, inplace=True)

    events_df = events_df.convert_dtypes()
    return events_df


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

    filtered_coupons, filtered_issues, filtered_offers = filter_coupons_issues_and_offers(all_coupons, all_issues, all_offers)

    print("\nAfter filtering:")
    print("nr coupons:", len(filtered_coupons))
    print("nr issues:", len(filtered_issues))
    print("nr offers:", len(filtered_offers))

    # plot_created_coupons(filtered_coupons)

    filtered_coupons, filtered_issues = perform_data_checks(all_coupons, filtered_coupons, filtered_issues, filtered_offers)
    # TODO: 'redeemed' 'after_expiring' what to do with it?

    print("\nAfter data checks:")
    print("nr coupons:", len(filtered_coupons))
    print("nr issues:", len(filtered_issues))
    print("nr offers:", len(filtered_offers))

    # sys.exit()

    events_df = make_events_timeline(filtered_coupons, filtered_issues)
    print(events_df)

    print("")
    total_decay  = np.sum(events_df['coupon_count'][events_df['event'] == Event.coupon_expired])
    total_amount = np.sum(filtered_issues['amount'])
    print('Total number of coupons: %d'%total_amount)
    print('Total number of coupons never accepted: %d'%total_decay)
    print('Percentage of coupons never accepted: %.3f%%'%(total_decay/total_amount))

    print("")
    TrackReport()

    # print_table_info(db)

    # result = db.execute(query)
    # df = pd.DataFrame(result.fetchall())
    # df = pd.read_sql_query(query, db)
    # print(df)

    # Close the connection to the database
    db.close()
    conn.close()



def print_table_info(db):
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



if __name__ == '__main__':
    main()