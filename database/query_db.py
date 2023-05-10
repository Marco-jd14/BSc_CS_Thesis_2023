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
pd.set_option('display.width', 130)


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


def filter_coupons_issues_and_offers(all_coupons, all_issues, all_offers):
    """ 
    Data prep:
        - filter out lottery coupons
        - filter on certain horizon (1 jan 2021 - 31 dec 2022)
        - filter out issues that have at least one coupon that got filtered out
        - filter out coupons belonging to an issue that got filtered out
        - filter out offers that do not have any filtered coupons belonging to them
    """

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


class Event(enum.Enum):
    member_declined     = 0
    member_accepted     = 1
    member_let_expire   = 2
    coupon_sent         = 3
    coupon_expired      = 4

status_to_event = {('declined', None):              Event.member_declined,
                   ('declined', 'after_accepting'): Event.member_declined,
                   ('expired', 'after_receiving'):  Event.member_let_expire,
                   ('expired', 'after_accepting'):  Event.member_accepted,
                   ('expired', 'not_redeemed'):     Event.member_accepted,
                   ('redeemed', None):              Event.member_accepted,
                   ('redeemed', 'after_expiring'):  Event.member_accepted}


def perform_data_checks(all_coupons, filtered_coupons, filtered_issues, filtered_offers):
    """
    Data check:
        - Check on coupon / offer type (no lottery coupons)
        - Check whether *all* coupons belonging to the filtered_issues are present in the filtered_coupons table (i.e. no incomplete issues)
        - Check if we do not have issues in the issue-table with an 'amount' of <= 0
        - Check whether all combinations of status + sub_status from coupons can be translated into a member_reponse
        - Update the 'total_issued' column from the issues table
        - Make new column 'total_accepted' column in the issues table (based on member_response)
        - Check whether for all issues we have 'total_accepted' <= 'total_issued'
    """

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
    filtered_issues = pd.merge(filtered_issues, coupon_counts, left_on='id', right_on='issue_id').drop('issue_id',axis=1)

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
    filtered_issues = pd.merge(filtered_issues, nr_accepted_coupons_per_issue, left_on='id', right_on='issue_id').drop('issue_id',axis=1)

    # Calculate issues with negative decay (which should be impossible)
    filtered_issues['my_decay'] = filtered_issues['amount'] - filtered_issues['total_accepted']
    issues_with_negative_decay = filtered_issues[filtered_issues['my_decay'] < 0]
    # Filter out issues and coupons with negative decay
    filtered_coupons = filtered_coupons[~filtered_coupons['issue_id'].isin(issues_with_negative_decay['id'])]
    filtered_issues = filtered_issues[~filtered_issues['id'].isin(issues_with_negative_decay['id'])]
    filtered_issues.drop('my_decay', axis=1, inplace=True)
    # Check if each 'amount' >= number of accepted coupons
    assert np.all(filtered_issues['total_accepted'] <= filtered_issues['amount'])

    return filtered_coupons, filtered_issues


class CouponLifeCycle(enum.Enum):
    created   = 1
    destroyed = -1


def get_coupon_follow_id(events_df, row_idx, destroyed_stack, verbose=True):
    if row_idx == 0:
        # Make new follow id for first row
        return 1

    discarded = False
    for k in range(len(destroyed_stack)):
        destroyed_index = destroyed_stack.pop(0)

        if not abs(events_df['at'].iloc[destroyed_index] - events_df['at'].iloc[row_idx]) <= dt.timedelta(seconds=60):
            if verbose: print("Discarded coupon %d (follow%d) (was not a match for %d)\t"%(destroyed_index, events_df['coupon_follow_id'].iloc[destroyed_index], row_idx), str(abs(events_df['at'].iloc[destroyed_index] - events_df['at'].iloc[row_idx])))
            discarded = True
        else:
            if discarded and verbose:
                print("Follow id %d is a match for %d"%(events_df['coupon_follow_id'].iloc[destroyed_index], row_idx))
            return events_df['coupon_follow_id'].iloc[destroyed_index]
            
    assert len(destroyed_stack) == 0
    # Make new follow id
    return np.max(events_df['coupon_follow_id'].iloc[:row_idx]) + 1


def add_coupon_follow_ids_to_coupons_and_filter(filtered_coupons, filtered_issues, filtered_offers):
    """ 
    This function does the following:
        - For every issue, try to match newly created coupons to expiry / declining times of other coupons
        - Assign every coupon a coupon_follow_id, to be able to recreate the stream of how 1 particular
            coupon is sent to multiple members and eventually (hopefully) accepted
        - Keep track of the issue-ids for which this recreated stream of coupons leads to inconsistencies
            in either of the following:
                - 'amount' in the issue table
                - Number of unique coupon_follow_ids
                - Number of initially sent out / released coupons
                - Number of active coupons at a time
        - Update the issue table with that information
        - Update the coupon table with the coupon_follow_ids

    Then filter out those issues that were found to be inconsistent
    """

    issue_info = []
    issues_to_filter_out = []
    coupon_follow_ids = {}

    for i, (index, issue_row) in enumerate(filtered_issues.iterrows()):
        if i%100 == 0:
            print("\rHandling issue nr %d"%i,end='')

        TrackTime("get issue_coupons")
        issue_coupons = filtered_coupons[filtered_coupons['issue_id'] == issue_row['id']]

        TrackTime("Make events_df")
        events = []
        for j, (index, coupon_row) in enumerate(issue_coupons.iterrows()):
            event = [CouponLifeCycle.created.name, CouponLifeCycle.created.value, coupon_row['id'], coupon_row['created_at'], coupon_row['status'], coupon_row['sub_status']]
            events.append(event)

            status, sub_status = coupon_row['status'], coupon_row['sub_status']
            if status == 'declined' or (status == 'expired' and sub_status == 'after_receiving'):
                event = [CouponLifeCycle.destroyed.name, CouponLifeCycle.destroyed.value, coupon_row['id'], coupon_row['status_updated_at'], coupon_row['status'], coupon_row['sub_status']]
                events.append(event)

        # Sort the events chronologically
        events_df = pd.DataFrame(events, columns=['event','coupon_nr','coupon_id','at','status','sub_status'])
        events_df['index'] = events_df.index
        events_df.sort_values(['at','index'], inplace=True)
        events_df.drop('index', axis=1, inplace=True)

        # Assign coupon_follow_ids to coupon_id's
        TrackTime("Determining coupon_follow_ids")
        events_df['coupon_follow_id'] = -1
        destroyed_stack = []
        for j in range(len(events_df)):
            if events_df['event'].iloc[j] == CouponLifeCycle.destroyed.name:
                assert events_df['coupon_follow_id'].iloc[j] > 0
                destroyed_stack.append(j)
            else:
                coupon_follow_id = get_coupon_follow_id(events_df, j, destroyed_stack, verbose=False)
                indices_to_update = events_df.index[events_df['coupon_id'] == events_df['coupon_id'].iloc[j]]
                events_df.loc[indices_to_update,'coupon_follow_id'] = coupon_follow_id

        events_df['active_coupons'] = np.cumsum(events_df['coupon_nr'].values)

        TrackTime("Updating coupon_follow_id dict")
        coupon_to_follow_id = events_df[['coupon_id','coupon_follow_id']].groupby('coupon_id').head(1)
        coupon_to_follow_id.index = coupon_to_follow_id['coupon_id'].values
        coupon_to_follow_id.drop('coupon_id',axis=1, inplace=True)
        coupon_follow_ids.update(coupon_to_follow_id.to_dict()['coupon_follow_id'])

        TrackTime("Expectation checks")
        equal = (issue_coupons['created_at'] - issue_row['sent_at']).abs() <= dt.timedelta(seconds=10)
        max_nr_active_coupons = np.max(events_df['active_coupons'].values)
        unique_coupon_follow_ids = np.max(events_df['coupon_follow_id'].values)
        
        # TODO: add info: nr_expired, nr_not_sent_on (before issue-expiry)
        info = [np.sum(equal), max_nr_active_coupons]
        issue_info.append(info)
        
        if np.sum(equal) != max_nr_active_coupons or max_nr_active_coupons != unique_coupon_follow_ids:

            issues_to_filter_out.append(issue_row['id'])
            print("\nInconsistent issue: %d     "%issue_row['id'], end='')
            print(np.sum(equal), max_nr_active_coupons, unique_coupon_follow_ids)

            offer_row = filtered_offers[filtered_offers['id'] == issue_row['offer_id']].squeeze()
            offer_row = offer_row[['id','title','category_id','description','total_issued']]

            issue_row['unique_coupon_follow_ids'] = unique_coupon_follow_ids
            issue_row['max_nr_active_coupons'] = max_nr_active_coupons
            issue_row['first_released'] = np.sum(equal)
            issue_row = issue_row[['id','offer_id','total_issued','total_reissued','decay_count','sent_at','expires_at','aborted_at','amount','max_nr_active_coupons','unique_coupon_follow_ids','first_released']]

            writer = pd.ExcelWriter('./issue_%d.xlsx'%i)
            offer_row.to_excel(writer, 'offer')
            issue_row.to_excel(writer, 'issue')
            events_df.to_excel(writer, 'coupons')
            writer.close()


    TrackTime("Merging into filtered_coupons")
    coupon_follow_ids = pd.DataFrame(coupon_follow_ids.items(), columns=['id','coupon_follow_id'])
    assert set(coupon_follow_ids['id']) == set(filtered_coupons['id']), TrackReport()
    filtered_coupons = pd.merge(filtered_coupons, coupon_follow_ids, on='id')

    issue_info = pd.DataFrame(issue_info, columns=['first_released','max_nr_active_coupons'])
    filtered_issues['first_released']        = issue_info['first_released']
    filtered_issues['max_nr_active_coupons'] = issue_info['max_nr_active_coupons']

    # Filtering out inconsistent issues
    print("\nissues_to_filter_out:", issues_to_filter_out)
    filtered_issues  = filtered_issues[~filtered_issues['id'].isin(issues_to_filter_out)]
    filtered_coupons = filtered_coupons[filtered_coupons['issue_id'].isin(filtered_issues['id'])]
    filtered_offers  = filtered_offers[filtered_offers['id'].isin(filtered_issues['offer_id'])]

    return filtered_coupons, filtered_issues, filtered_offers


def perform_data_checks_on_follow_coupon_ids(filtered_coupons, filtered_issues, filtered_offers):
    # TODO: check if total_accepted + nr_expired + nr_not_sent_on == total_issued
    # nr_unique_follow_coupon_ids == amount
    # nr_active_coupons == amount
    # first_released == amount

    for i, (index, issue_row) in enumerate(filtered_issues.iterrows()):
        pass

    # # Check whether the first 'amount' coupons from an issue are all created at the same time as the issue was sent
    # # coupon_counts = filtered_coupons[['issue_id']].groupby('issue_id').aggregate(total_issued=('issue_id','count')).reset_index()
    # created_equals_sent = filtered_coupons[['created_at','issue_id']]
    # created_equals_sent = pd.merge(created_equals_sent, filtered_issues[['id','sent_at']], left_on='issue_id', right_on='id').drop('id',axis=1)
    # created_equals_sent['created_equals_sent'] = (created_equals_sent['created_at'] - created_equals_sent['sent_at']).abs() < dt.timedelta(seconds=10)
    # print(created_equals_sent)
    # issues = created_equals_sent.groupby('issue_id').aggregate(nr_created_equals_sent=('created_equals_sent','sum')).reset_index()
    # issues = pd.merge(issues, filtered_issues[['id','amount']], left_on='issue_id', right_on='id').drop('id',axis=1)

    # print(issues[issues['nr_created_equals_sent'] != issues['amount']])

    # # assert np.all(issues['nr_created_equals_sent'] == issues['amount'])


def save_df_to_sql(db, filtered_coupons, filtered_issues, filtered_offers):
    TrackTime("Writing to db")

    db.execute("DROP TABLE IF EXISTS `filtered_coupons`;")
    db.execute("DROP TABLE IF EXISTS `filtered_issues`;")
    db.execute("DROP TABLE IF EXISTS `filtered_offers`;")

    # Translate Event objects to str for exporting to SQL
    filtered_coupons['member_response'] = filtered_coupons['member_response'].apply(lambda event: str(event).replace('Event.',''))

    filtered_coupons.to_sql("filtered_coupons", db)
    filtered_issues.to_sql("filtered_issues", db)
    filtered_offers.to_sql("filtered_offers", db)


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
        member_response = coupon_row['member_response'] # status_to_event[(status, sub_status)]
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


def filter_and_check_data(db, save_to_SQL=False):
    TrackTime("Retrieve from db")
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

    TrackTime("Filtering")
    filtered_coupons, filtered_issues, filtered_offers = filter_coupons_issues_and_offers(all_coupons, all_issues, all_offers)

    print("\nAfter filtering:")
    print("nr coupons:", len(filtered_coupons))
    print("nr issues:", len(filtered_issues))
    print("nr offers:", len(filtered_offers))

    # plot_created_coupons(filtered_coupons)

    TrackTime("Data checks")
    filtered_coupons, filtered_issues = perform_data_checks(all_coupons, filtered_coupons, filtered_issues, filtered_offers)
    # TODO: 'redeemed' 'after_expiring' what to do with it?

    print("\nAfter data checks:")
    print("nr coupons:", len(filtered_coupons))
    print("nr issues:", len(filtered_issues))
    print("nr offers:", len(filtered_offers))

    print("\nInvestigating issues")
    result = add_coupon_follow_ids_to_coupons_and_filter(filtered_coupons, filtered_issues, filtered_offers)
    filtered_coupons, filtered_issues, filtered_offers = result

    perform_data_checks_on_follow_coupon_ids(filtered_coupons, filtered_issues, filtered_offers)

    print("\nAfter follow_coupon_id data checks:")
    print("nr coupons:", len(filtered_coupons))
    print("nr issues:", len(filtered_issues))
    print("nr offers:", len(filtered_offers))

    if save_to_SQL:
        print("\nWriting to SQL...")
        save_df_to_sql(db, filtered_coupons, filtered_issues, filtered_offers)



def main():
    TrackTime("Connect to db")
    conn = connect_db.establish_host_connection()
    db   = connect_db.establish_database_connection(conn)
    print("Successfully connected to database '%s'"%str(db.engine).split("/")[-1][:-1])


    filter_and_check_data_from_scratch = False
    make_baseline_events_from_scratch = False


    if filter_and_check_data_from_scratch:
        filter_and_check_data(db, save_to_SQL=False)
    else:
        TrackTime("select from db")
        query = "select * from filtered_coupons"
        filtered_coupons = pd.read_sql_query(query, db)
        filtered_coupons['member_response'] = filtered_coupons['member_response'].apply(lambda event: Event[str(event)])

        query = "select * from filtered_issues"
        filtered_issues = pd.read_sql_query(query, db)

        query = "select * from filtered_offers"
        filtered_offers = pd.read_sql_query(query, db)

    """ TODO:
    Get list of involved members, and compute
       - probability of letting a coupon expire
       - Subscribed categories
       - probability of accepting based on subscribed categories (coupon --> issue --> offer --> cat)
    """


    if make_baseline_events_from_scratch:
        events_df = make_events_timeline(filtered_coupons, filtered_issues, filtered_offers)
        TrackTime("print")
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