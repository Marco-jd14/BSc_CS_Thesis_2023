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
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
from lib.tracktime import TrackTime, TrackReport

import connect_db
import query_db
Event = query_db.Event

# For printing dataframes
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 120)


# All the possible combinations of status + sub_status found in the data
status_to_event = {('declined',  None):              Event.member_declined,
                   ('declined', 'after_accepting'):  Event.member_declined,  # Discard the information that the member accepted initially
                   ('expired',  'after_receiving'):  Event.member_let_expire,
                   ('expired',  'after_accepting'):  Event.member_accepted,
                   ('expired',  'not_redeemed'):     Event.member_accepted,
                   ('redeemed',  None):              Event.member_accepted,
                   ('redeemed', 'after_expiring'):   Event.member_accepted}


def main():
    # Establish a connection to the database
    TrackTime("Connect to db")
    conn = connect_db.establish_host_connection()
    db   = connect_db.establish_database_connection(conn)
    print("Successfully connected to database '%s'"%str(db.engine).split("/")[-1][:-1])

    # Choose whether or not to restart the data filtering & checking process
    filter_and_check_data_from_scratch = True

    if filter_and_check_data_from_scratch:
        # Also choose whether or not to save the result to the database ornot
        filter_and_check_data(db, save_to_SQL_DB=False)

    TrackTime("Retrieve from db")
    result = query_db.retrieve_from_sql_db(db, 'filtered_coupons', 'filtered_issues', 'filtered_offers')
    filtered_coupons, filtered_issues, filtered_offers = result

    print("\nfiltered_coupons table from db shape:", filtered_coupons.shape)
    print("filtered_issues table from db shape:", filtered_issues.shape)
    print("filtered_offers table from db shape:", filtered_offers.shape)

    # Plot a timeline
    plot_timeline_active_coupons(filtered_coupons)

    print("")
    TrackReport()

    # print_table_info(db)

    # Close the connection to the database
    db.close()
    conn.close()


############## One function that performs all the steps in the data filtering & checking process #############################

def filter_and_check_data(db, save_to_SQL_DB=False):
    """
    This function
        - Retrieves all coupons, issues, and offers from the database
        - Filters out coupons that we are not interested in
        - Then does some simple data checks on them
        - Assigns 'coupon_follow_id's to every coupon in the coupon table
          These ids are useful for tracking a stream of one unique coupon, to see
          past which members it went until it was eventually accepted (or expired)
        - Do some more data filtering by checking for consistency with the help of
          the 'coupon_follow_id's
        - More final data checks
        - Optionally write the filtered data to a MySQL database
    """
    # Retrieve tables from SQL db
    TrackTime("Retrieve from db")
    result = query_db.retrieve_from_sql_db(db, 'coupon', 'issue', 'offer')
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

    print("\nInvestigating issue-consistencies")
    # Add extra information to the coupons and issues table
    filtered_coupons, filtered_issues = add_coupon_follow_ids_to_coupons(filtered_coupons, filtered_issues, filtered_offers, verbose=True)

    # With the newly added information, filter out more data
    print("\nFiltering out inconsistent issues")
    result = filter_out_inconsistent_issues(filtered_coupons, filtered_issues, filtered_offers)
    filtered_coupons, filtered_issues, filtered_offers =  result

    print("\nAfter follow_coupon_id data filtering:")
    print("nr coupons:", len(filtered_coupons), "  (%d columns)"%len(filtered_coupons.columns))
    print("nr issues:", len(filtered_issues), "  (%d columns)"%len(filtered_issues.columns))
    print("nr offers:", len(filtered_offers), "  (%d columns)"%len(filtered_offers.columns))
    print("")

    if save_to_SQL_DB:
        TrackTime("Writing to db")
        name_to_table_mapping = {'filtered_coupons':filtered_coupons, 'filtered_issues':filtered_issues, 'filtered_offers':filtered_offers}
        query_db.save_df_to_sql(db, name_to_table_mapping)


############## Filtering out irrelevant coupons, issues, and offers #############################

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


############## Performing some simple data check on the filtered coupons, issues, and offers #############################

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


############## Investigating the stream of coupons and assigning 'coupon_follow_ids' to all coupons #############################

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


def add_coupon_follow_ids_to_coupons(filtered_coupons, filtered_issues, filtered_offers, verbose=False, export_inconsistent_issues=False):
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
        - Update the issue table with the above information
        - Update the coupon table with the coupon_follow_ids
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
            continue
        
        if nr_unique_coupon_follow_ids != issue_row['amount']:
            # Print some information to the terminal as to why the issue is found to be inconsistent
            issues_to_filter_out.append(issue_row['id'])
            if verbose:
                print("\nFound inconsistent issue nr %d (id = %d)"%(len(issues_to_filter_out), issue_row['id']))
                print("\t%30s"%"nr_first_released:", nr_coupons_sent_out_at_first)
                print("\t%30s"%"max_nr_active_coupons:", max_nr_active_coupons)
                print("\t%30s"%"nr_unique_coupon_follow_ids:", nr_unique_coupon_follow_ids)
                print("\t%30s"%"amount:", issue_row['amount'])
                print("\t%30s"%"decay:", issue_row['decay_count'])

            if True:#export_inconsistent_issues:
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

            # Go to next issue, as the current issue was found to be inconsistent
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

    print("\nissues to filter out:  ", issues_to_filter_out)

    return filtered_coupons, filtered_issues


############## With the newly assigned 'coupon_follow_ids' perform more complex data checks / filtering #############################

def filter_out_inconsistent_issues(filtered_coupons, filtered_issues, filtered_offers):
    """
    This function does the following:
        - It first computes some new information about issues (partially enabled by the newly assigned coupon_follow_ids)
            + The 'nr_unique_coupon_follow_ids'
            + The 'nr_first_released'
            + 'member_response'
        - It then filters out issues that are inconsistent based on those computed numbers
        - Then, it computes some more information about what happened to every last coupon from a stream of one coupon_follow_id.
          Any last coupon can either be
            + accepted
            + never accepted
            If a coupon is never accepted, we differentiate two cases:
              + Coupon expired and was therefore not sent to a next member
              + Coupon was not sent to a next member for reasons other than expiry (i.e. coupon was not expired yet, but also not sent to the next)
        - Also add the 'eventually_accepted' property to the coupons table
    """

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
    last_followed_coupons['accepted']    = was_coupon_accepted(last_followed_coupons)
    last_followed_coupons['expired']     = was_coupon_expired(last_followed_coupons)
    last_followed_coupons['not_sent_on'] = was_coupon_not_sent_on(last_followed_coupons)

    nr_accepted    = last_followed_coupons.groupby('issue_id').aggregate(nr_accepted=('accepted','sum')).reset_index()
    nr_expired     = last_followed_coupons.groupby('issue_id').aggregate(nr_expired=('expired','sum')).reset_index()
    nr_not_sent_on = last_followed_coupons.groupby('issue_id').aggregate(nr_not_sent_on=('not_sent_on','sum')).reset_index()
    # nr_accepted     = last_followed_coupons.groupby('issue_id').apply(lambda issue_group: was_coupon_accepted(issue_group).sum()).reset_index(name='nr_accepted')
    # nr_expired      = last_followed_coupons.groupby('issue_id').apply(lambda issue_group: was_coupon_expired(issue_group).sum()).reset_index(name='nr_expired')
    # nr_not_sent_on  = last_followed_coupons.groupby('issue_id').apply(lambda issue_group: was_coupon_not_sent_on(issue_group).sum()).reset_index(name='nr_not_sent_on')

    # Merge the info about nr_accepted, nr_expired, and nr_not_sent_on into the issues table
    last_coupons_status = pd.merge(pd.merge(nr_accepted, nr_expired, on='issue_id'), nr_not_sent_on, on='issue_id') 
    filtered_issues = pd.merge(filtered_issues, last_coupons_status, left_on='id', right_on='issue_id').drop('issue_id',axis=1)
    assert len(last_coupons_status) == len(filtered_issues)
    assert np.all(filtered_issues['amount'] >= filtered_issues['nr_accepted']), "Cannot be more coupons accepted than were supposed to be given out"
    assert np.all(filtered_issues['amount'] == filtered_issues['nr_accepted'] + filtered_issues['nr_expired'] + filtered_issues['nr_not_sent_on'])
    assert filtered_issues['nr_accepted'].sum() == (filtered_coupons['member_response'] == Event.member_accepted).sum()

    assert filtered_issues['total_issued'].sum() == len(filtered_coupons)

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
    print("\nnr_members_sent_before_accepted avg:", np.round(nr_members_sent_before_accepted['nr_sent_to'].mean(), 3))
    print("total coupons that were eventually accepted / nr_accepted:", np.round(len(filtered_coupons[filtered_coupons['eventually_accepted']]) / filtered_issues['nr_accepted'].sum(), 3))
    print("total_coupons / nr_accepted:", np.round(filtered_issues['total_issued'].sum() / filtered_issues['nr_accepted'].sum(), 3))

    return filtered_coupons, filtered_issues, filtered_offers


############## Some functions to get more information about the data #############################

def datetime_range(start_date, end_date, delta):
    result = []
    nxt = start_date
    delta = relativedelta(**delta)

    while nxt <= end_date:
        result.append(nxt)
        nxt += delta

    result.append(end_date)
    return result


def plot_timeline_active_coupons(df):
    created = df['created_at']
    created = created.sort_values(ascending=True)
    assert created.is_monotonic

    start_date = created.iloc[0] - dt.timedelta(seconds=1)
    end_date   = created.iloc[-1] + dt.timedelta(seconds=1)
    delta = {'days':7}
    intervals = datetime_range(start_date, end_date, delta)

    res = created.groupby(pd.cut(created, intervals)).count()
    res.name = "nr_coupons_per_interval"
    res = res.reset_index()

    interval_ends = list(map(lambda interval: interval.right, res.created_at))
    plt.plot(interval_ends, res.nr_coupons_per_interval)
    plt.xticks(fontsize=8)



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