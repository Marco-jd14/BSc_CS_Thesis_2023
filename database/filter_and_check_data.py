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


# Relevant events according to the coupon lifecycle
class Event(enum.Enum):
    member_declined     = 0
    member_accepted     = 1
    member_let_expire   = 2 # i.e. after 3 days
    coupon_sent         = 3
    coupon_expired      = 4 # i.e. after 1 month

# All the possible combinations of status + sub_status found in the data
status_to_event = {('declined',  None):              Event.member_declined,
                   ('declined', 'after_accepting'):  Event.member_declined,  # Discard the information that the member accepted initially
                   ('expired',  'after_receiving'):  Event.member_let_expire,
                   ('expired',  'after_accepting'):  Event.member_accepted,
                   ('expired',  'not_redeemed'):     Event.member_accepted,
                   ('redeemed',  None):              Event.member_accepted,
                   ('redeemed', 'after_expiring'):   Event.member_accepted}


def perform_data_checks(all_coupons, filtered_coupons, filtered_issues, filtered_offers):
    """
    Data check:
        - Check on coupon / offer type (no lottery coupons)
        - Check whether all coupons belonging to the filtered_issues are present in the filtered_coupons table (i.e. no incomplete issues)
        - Check if we do not have issues in the issue-table with an 'amount' of <= 0
        - Check whether all combinations of status + sub_status from coupons can be translated into a member_reponse / event (class Event)
        - Update the 'total_issued' column from the issues table (contained mostly zeroes even if coupons were sent out)
        - Make new column 'total_accepted' column in the issues table (based on member_response / Events)
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

    # Convert 'issue_id' to an int. It was of dtype float before, to be able to support NaN (which we filtered out)
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

    return filtered_coupons, filtered_issues


class CouponExistence(enum.Enum):
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


def add_coupon_follow_ids_to_coupons(filtered_coupons, filtered_issues, filtered_offers):
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

    for i, (issue_index, issue_row) in enumerate(filtered_issues.iterrows()):
        if i%100 == 0:
            print("\rHandling issue nr %d"%i,end='')

        TrackTime("Other")
        issue_coupons = filtered_coupons[filtered_coupons['issue_id'] == issue_row['id']]

        TrackTime("Make coupon_existence_df")
        coupon_created_or_destroyed_events = []
        for j, (coupon_index, coupon_row) in enumerate(issue_coupons.iterrows()):
            # Make the coupon-creation event
            event = [CouponExistence.created.name, CouponExistence.created.value, coupon_row['id'], coupon_row['created_at'], coupon_row['status'], coupon_row['sub_status']]
            coupon_created_or_destroyed_events.append(event)

            # If one of the following combinations of status + sub_status occurred, the coupon was destroyed which implies it could be sent to a next member
            status, sub_status = coupon_row['status'], coupon_row['sub_status']
            if status == 'declined' or (status == 'expired' and sub_status == 'after_receiving'):
                event = [CouponExistence.destroyed.name, CouponExistence.destroyed.value, coupon_row['id'], coupon_row['status_updated_at'], coupon_row['status'], coupon_row['sub_status']]
                coupon_created_or_destroyed_events.append(event)

        # Sort the events chronologically, with the index as secondary sorting criteria (if the events occurred at the same time)
        coupon_existence_df = pd.DataFrame(coupon_created_or_destroyed_events, columns=['event','coupon_nr','coupon_id','at','status','sub_status'])
        coupon_existence_df['index'] = coupon_existence_df.index
        coupon_existence_df.sort_values(['at','index'], inplace=True)
        coupon_existence_df.drop('index', axis=1, inplace=True)

        # Assign coupon_follow_ids to coupon_id's
        TrackTime("Determining coupon_follow_ids")
        coupon_existence_df['coupon_follow_id'] = -1
        destroyed_stack = []
        for j in range(len(coupon_existence_df)):
            if coupon_existence_df['event'].iloc[j] == CouponExistence.destroyed.name:
                assert coupon_existence_df['coupon_follow_id'].iloc[j] > 0, "No coupon_follow_id yet even though the coupon should have been already asssigned one?"
                destroyed_stack.append(j)
            else:
                # When a coupon is created, assign it a coupon_follow_id (either a new one, or an old one)
                coupon_follow_id = get_coupon_follow_id(coupon_existence_df, j, destroyed_stack, verbose=False)
                # Update the every row with same coupon_id to the newly found coupon_follow_id
                indices_to_update = coupon_existence_df.index[coupon_existence_df['coupon_id'] == coupon_existence_df['coupon_id'].iloc[j]]
                coupon_existence_df.loc[indices_to_update,'coupon_follow_id'] = coupon_follow_id
        # Check if all coupons have been assigned a coupon_follow_id
        assert not np.any(coupon_existence_df['coupon_follow_id'] == -1), "Not all coupons have been assigned a coupon_follow_id"

        # The active number of coupons at any given time is the sum of coupon_nrs up until that point
        # The 'coupon_nr' is +1 if a coupon is created, -1 if a coupon is destroyed (declined / expired)
        coupon_existence_df['active_coupons'] = np.cumsum(coupon_existence_df['coupon_nr'].values)

        # Make a mapping from coupon_id to coupon_follow_id
        TrackTime("Adding new coupon_follow_ids")
        coupon_to_follow_id = coupon_existence_df[['coupon_id','coupon_follow_id']].groupby('coupon_id').head(1)
        coupon_to_follow_id.index = coupon_to_follow_id['coupon_id'].values
        coupon_to_follow_id.drop('coupon_id',axis=1, inplace=True)
        coupon_follow_ids.update(coupon_to_follow_id.to_dict()['coupon_follow_id'])

        TrackTime("Checking issue consistency")
        # Calculate which coupons were sent out at first (i.e. which coupons were created at the same time the issue was first sent out)
        coupons_sent_out_at_first = (issue_coupons['created_at'] - issue_row['sent_at']).abs() <= dt.timedelta(seconds=10)
        nr_coupons_sent_out_at_first = np.sum(coupons_sent_out_at_first)  # coupons_sent_out_at_first is a boolean array (boolean aka 0s or 1s)
        # Calculate the maximum number of outstanding coupons at any given time
        max_nr_active_coupons = np.max(coupon_existence_df['active_coupons'].values)
        # The unique amount of coupon_follow_ids (these id's are set up in such a way that the nr_unique is equal to the maximum id)
        nr_unique_coupon_follow_ids = np.max(coupon_existence_df['coupon_follow_id'].values)
        assert nr_unique_coupon_follow_ids == len(coupon_existence_df['coupon_follow_id'].unique()), "coupon_follow_id creation error"

        # Save some information to be added to the filtered_issues table
        info = [nr_coupons_sent_out_at_first, max_nr_active_coupons]
        issue_info.append(info)

        if nr_coupons_sent_out_at_first != max_nr_active_coupons or max_nr_active_coupons != nr_unique_coupon_follow_ids:
            # Print some information to the terminal as to why the issue is found to be inconsistent
            issues_to_filter_out.append(issue_row['id'])
            print("\nInconsistent issue: %d"%issue_row['id'])
            print("\t%30s"%"nr_first_released:", nr_coupons_sent_out_at_first)
            print("\t%30s"%"max_nr_active_coupons:", max_nr_active_coupons)
            print("\t%30s"%"nr_unique_coupon_follow_ids:", nr_unique_coupon_follow_ids)
            print("\t%30s"%"amount:", issue_row['amount'])

            # Make interesting offer information to export
            offer_row = filtered_offers[filtered_offers['id'] == issue_row['offer_id']].squeeze()
            offer_row = offer_row[['id','title','category_id','description','total_issued']]

            # Make interesting issue information to export
            issue_row['nr_unique_coupon_follow_ids'] = nr_unique_coupon_follow_ids
            issue_row['max_nr_active_coupons'] = max_nr_active_coupons
            issue_row['nr_first_released'] = nr_coupons_sent_out_at_first
            issue_row['total_accepted'] = np.sum(issue_coupons['member_response'] == Event.member_accepted)
            issue_row = issue_row[['id','offer_id','total_issued','total_reissued','decay_count','sent_at','expires_at','aborted_at','amount','max_nr_active_coupons','nr_unique_coupon_follow_ids','nr_first_released','total_accepted']]

            # Export the 3 tables to one excel file with 3 sheets
            writer = pd.ExcelWriter('./issue_%d.xlsx'%issue_row['id'])
            offer_row.to_excel(writer, 'offer')
            issue_row.to_excel(writer, 'issue')
            coupon_existence_df.to_excel(writer, 'coupons')
            writer.close()
            continue

        # If the code reaches this point, we know nr_unique_coupon_follow_ids == max_nr_active_coupons == nr_coupons_sent_out_at_first
        # Check if these 3 numbers are also equal to the 'amount', otherwise update it to the correct number
        if nr_unique_coupon_follow_ids != issue_row['amount']:
            filtered_issues.loc[issue_index,'amount'] = nr_unique_coupon_follow_ids

    # Merge the coupon_follow_ids into the filtered_coupons table
    TrackTime("Other")
    coupon_follow_ids = pd.DataFrame(coupon_follow_ids.items(), columns=['id','coupon_follow_id'])
    assert set(coupon_follow_ids['id']) == set(filtered_coupons['id']), "Cannot merge tables, as not all coupons have been assigned a coupon_follow_id"
    filtered_coupons = pd.merge(filtered_coupons, coupon_follow_ids, on='id')

    # Merge the issue_info into the filtered_issues table
    issue_info = pd.DataFrame(issue_info, columns=['nr_first_released','max_nr_active_coupons'])
    filtered_issues['nr_first_released']     = issue_info['nr_first_released']
    filtered_issues['max_nr_active_coupons'] = issue_info['max_nr_active_coupons']

    print("\nissues to filter out:", issues_to_filter_out)

    return filtered_coupons, filtered_issues


def filter_out_inconsistent_issues(filtered_coupons, filtered_issues, filtered_offers):

    if 'total_accepted'    in filtered_issues.columns: filtered_issues.drop('total_accepted', axis=1, inplace=True)
    if 'nr_first_released' in filtered_issues.columns: filtered_issues.drop('nr_first_released', axis=1, inplace=True)
    if 'member_response'   in filtered_coupons.columns: filtered_coupons.drop('member_response', axis=1, inplace=True)

    assert 'coupon_follow_id' in filtered_coupons.columns, 'Coupon_follow_ids must first be assigned to be able to filter out inconsistent issues'
    assert 'max_nr_active_coupons' in filtered_issues.columns, 'Maximum nr of active coupons at any given time (per issue) must first me determined to be able to filter out inconsistent issues'

    # Add 'member_response' column to coupons table
    filtered_coupons['member_response'] = filtered_coupons.apply(lambda row: status_to_event[(row['status'], row['sub_status'])], axis=1)

    # Add 'nr_first_released' column to issues table
    coupons_created_and_sent = pd.merge(filtered_coupons[['created_at','issue_id']], filtered_issues[['id','sent_at']], left_on='issue_id', right_on='id').drop('id',axis=1)
    coupons_created_and_sent['released_in_first_batch'] = (coupons_created_and_sent['created_at'] - coupons_created_and_sent['sent_at']).abs() <= dt.timedelta(seconds=10)
    nr_first_released = coupons_created_and_sent.groupby('issue_id').aggregate(nr_first_released=('released_in_first_batch','sum')).reset_index()
    filtered_issues = pd.merge(filtered_issues, nr_first_released, left_on='id', right_on='issue_id').drop('issue_id',axis=1)

    # Add 'nr_unique_coupon_follow_ids' column to issues table
    nr_unique_coupon_follow_ids = filtered_coupons.groupby('issue_id').aggregate(nr_unique_coupon_follow_ids=('coupon_follow_id','nunique')).reset_index()
    filtered_issues = pd.merge(filtered_issues, nr_unique_coupon_follow_ids, left_on='id', right_on='issue_id').drop('issue_id',axis=1)

    columns_that_should_be_equal = ['max_nr_active_coupons', 'nr_unique_coupon_follow_ids', 'nr_first_released', 'amount']
    nr_cols_that_are_equal = np.ones(len(filtered_issues))
    for col_idx in range(1, len(columns_that_should_be_equal)):
        col_name1, col_name2 = columns_that_should_be_equal[col_idx - 1], columns_that_should_be_equal[col_idx]
        nr_cols_that_are_equal += (filtered_issues[col_name1] == filtered_issues[col_name2])

    inconsistent_issue_ids = filtered_issues['id'][nr_cols_that_are_equal != len(columns_that_should_be_equal)]
    print("inconsistent_issue_ids:", list(inconsistent_issue_ids.values))

    filtered_issues = filtered_issues[nr_cols_that_are_equal == len(columns_that_should_be_equal)]
    filtered_coupons = filtered_coupons[filtered_coupons['issue_id'].isin(filtered_issues['id'])]
    filtered_offers  = filtered_offers[filtered_offers['id'].isin(filtered_issues['offer_id'])]

    # Check if indeed now all 4 columns are equal for every issue
    assert np.all(filtered_issues['max_nr_active_coupons'] == filtered_issues['nr_unique_coupon_follow_ids'])
    assert np.all(filtered_issues['max_nr_active_coupons'] == filtered_issues['nr_first_released'])
    assert np.all(filtered_issues['max_nr_active_coupons'] == filtered_issues['amount'])

    # Retrieve from every coupon_follow_id, the very last entry, to check whether this coupon was accepted or not
    last_followed_coupons = filtered_coupons.groupby(['issue_id','coupon_follow_id']).last().reset_index()
    assert len(last_followed_coupons) == filtered_issues['amount'].sum()

    # Merge the issue-property 'expires_at' into the coupons table
    to_join = filtered_issues[['id','expires_at']].rename(columns={'id':'issue_id'})
    last_followed_coupons = pd.merge(last_followed_coupons, to_join, on='issue_id')

    # Functions to determine whether the last coupon from the same coupon_follow_id stream was accepted, expired, or just not sent to the next member
    def was_coupon_accepted(issue_group):
        return issue_group['member_response']==Event.member_accepted
    def was_coupon_expired(issue_group):
        return (issue_group['member_response'] != Event.member_accepted) & \
               (issue_group['status_updated_at'] >= (issue_group['expires_at'] - issue_group['accept_time'].apply(lambda el: dt.timedelta(days=el))) )
    def was_coupon_not_sent_on(issue_group):
        return (issue_group['member_response'] != Event.member_accepted) & \
               (issue_group['status_updated_at'] < (issue_group['expires_at'] - issue_group['accept_time'].apply(lambda el: dt.timedelta(days=el))) )

    # Apply the above functions to each issue-id
    TrackTime("Calculating nr_not_sent_on")
    nr_accepted     = last_followed_coupons.groupby('issue_id').apply(lambda issue_group: was_coupon_accepted(issue_group).sum()).reset_index(name='nr_accepted')
    nr_expired      = last_followed_coupons.groupby('issue_id').apply(lambda issue_group: was_coupon_expired(issue_group).sum()).reset_index(name='nr_expired')
    nr_not_sent_on  = last_followed_coupons.groupby('issue_id').apply(lambda issue_group: was_coupon_not_sent_on(issue_group).sum()).reset_index(name='nr_not_sent_on')

    TrackTime("Other")
    # Merge the info about nr_accepted, nr_expired, and nr_not_sent_on into the issues table
    last_coupons_status = pd.merge(pd.merge(nr_accepted, nr_expired, on='issue_id'), nr_not_sent_on, on='issue_id') 
    filtered_issues = pd.merge(filtered_issues, last_coupons_status, left_on='id', right_on='issue_id').drop('issue_id',axis=1)
    assert len(last_coupons_status) == len(filtered_issues)
    assert np.all(filtered_issues['amount'] >= filtered_issues['nr_accepted']), "Cannot be more coupons accepted than were supposed to be given out"
    assert np.all(filtered_issues['amount'] == filtered_issues['nr_accepted'] + filtered_issues['nr_expired'] + filtered_issues['nr_not_sent_on'])
    assert filtered_issues['nr_accepted'].sum() == (filtered_coupons['member_response'] == Event.member_accepted).sum()

    print("total nr_accepted =", filtered_issues['nr_accepted'].sum())
    print("total nr_expired =", filtered_issues['nr_expired'].sum())
    print("total nr_not_sent_on =", filtered_issues['nr_not_sent_on'].sum())
    # TODO: check if nr_not_sent_on was because zero members did not fit into an offer-criteria

    # Make two sets of follow_coupons_ids: those that eventually get accepted and those that don't
    set_follow_coupon_ids = set(list(map(tuple, filtered_coupons[['issue_id','coupon_follow_id']].values)))
    accepted_follow_coupon_ids = filtered_coupons[['issue_id','coupon_follow_id']][filtered_coupons['member_response'] == Event.member_accepted]
    not_accepted_follow_coupon_ids = set_follow_coupon_ids - set(list(map(tuple, accepted_follow_coupon_ids.values)))
    not_accepted_follow_coupon_ids = pd.DataFrame(not_accepted_follow_coupon_ids, columns=['issue_id','coupon_follow_id'])

    # Add 'eventually_accepted' column to coupons table
    accepted_follow_coupon_ids['eventually_accepted'] = True
    not_accepted_follow_coupon_ids['eventually_accepted'] = False
    follow_coupon_ids = pd.concat([accepted_follow_coupon_ids, not_accepted_follow_coupon_ids], ignore_index=True)
    assert set_follow_coupon_ids == set(list(map(tuple, follow_coupon_ids[['issue_id','coupon_follow_id']].values)))
    filtered_coupons = pd.merge(filtered_coupons, follow_coupon_ids, on=['issue_id','coupon_follow_id'])

    # Calculate the global average pass through rate
    coupons_from_accepted_follow_ids = filtered_coupons[filtered_coupons['eventually_accepted']]
    nr_members_sent_before_accepted = coupons_from_accepted_follow_ids.groupby(['issue_id','coupon_follow_id']).aggregate(nr_sent_to=('id','count')).reset_index()
    print("nr_members_sent_before_accepted avg:", nr_members_sent_before_accepted['nr_sent_to'].mean())

    return filtered_coupons, filtered_issues, filtered_offers


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


def filter_and_check_data(db, save_to_SQL_DB=False):
    # Retrieve tables from SQL db
    TrackTime("Retrieve from db")
    result = retrieve_from_sql_db(db, 'coupon', 'issue', 'offer')
    all_coupons, all_issues, all_offers = result

    print("\nBefore filtering:")
    print("nr coupons:", len(all_coupons))
    print("nr issues:", len(all_issues))
    print("nr offers:", len(all_offers))

    TrackTime("Filtering")
    # Filter coupons, issues, and offers
    filtered_coupons, filtered_issues, filtered_offers = filter_coupons_issues_and_offers(all_coupons, all_issues, all_offers)

    print("\nAfter filtering:")
    print("nr coupons:", len(filtered_coupons))
    print("nr issues:", len(filtered_issues))
    print("nr offers:", len(filtered_offers))

    TrackTime("Data checks")
    # Perform some data checks and add 'member_response' to coupons, add 'total_accepted' to issues, and update 'total_issued' in issues
    filtered_coupons, filtered_issues = perform_data_checks(all_coupons, filtered_coupons, filtered_issues, filtered_offers)

    print("\nAfter data checks:")
    print("nr coupons:", len(filtered_coupons))
    print("nr issues:", len(filtered_issues))
    print("nr offers:", len(filtered_offers))

    print("\nInvestigating issue-consistencies")
    # Add extra information to the coupons and issues table
    filtered_coupons, filtered_issues = add_coupon_follow_ids_to_coupons(filtered_coupons, filtered_issues, filtered_offers)

    # With the newly added information, filter out more data
    print("\nFiltering out inconsistent issues")
    result = filter_out_inconsistent_issues(filtered_coupons, filtered_issues, filtered_offers)
    filtered_coupons, filtered_issues, filtered_offers =  result

    print("\nAfter follow_coupon_id data filtering:")
    print("nr coupons:", len(filtered_coupons))
    print("nr issues:", len(filtered_issues))
    print("nr offers:", len(filtered_offers))

    plot_created_coupons(filtered_coupons)

    if save_to_SQL_DB:
        TrackTime("Writing to db")
        name_to_table_mapping = {'filtered_coupons':filtered_coupons, 'filtered_issues':filtered_issues, 'filtered_offers':filtered_offers}
        save_df_to_sql(db, name_to_table_mapping)



def main():
    TrackTime("Connect to db")
    conn = connect_db.establish_host_connection()
    db   = connect_db.establish_database_connection(conn)
    print("Successfully connected to database '%s'"%str(db.engine).split("/")[-1][:-1])


    filter_and_check_data_from_scratch = False
    make_baseline_events_from_scratch  = False


    if filter_and_check_data_from_scratch:
        filter_and_check_data(db, save_to_SQL_DB=False)

    TrackTime("Retrieve from db")
    result = retrieve_from_sql_db(db, 'filtered_coupons', 'filtered_issues', 'filtered_offers')
    filtered_coupons, filtered_issues, filtered_offers = result


    """ TODO:
    Get list of involved members, and compute
       - probability of letting a coupon expire
       - Subscribed categories
       - probability of accepting based on subscribed categories (coupon --> issue --> offer --> cat)
    """


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