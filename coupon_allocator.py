# -*- coding: utf-8 -*-
"""
Created on Wed May 24 20:46:00 2023

@author: Marco
"""

import sys
import copy
import enum
import numpy as np
import pandas as pd
import datetime as dt
from pprint import pprint
from database.lib.tracktime import TrackTime, TrackReport

import database.connect_db as connect_db
import database.query_db as query_db
Event = query_db.Event


def main():
    TrackTime("Connect to db")
    conn = connect_db.establish_host_connection()
    db   = connect_db.establish_database_connection(conn)
    print("Successfully connected to database '%s'"%str(db.engine).split("/")[-1][:-1])

    preparation = prepare_allocator(db)
    allocate_coupons(*preparation)

    print("")
    TrackReport()


def prepare_allocator(db):
    TrackTime("Retrieve from db")
    result = query_db.retrieve_from_sql_db(db, 'filtered_issues', 'filtered_offers', 'member')
    filtered_issues, filtered_offers, all_members = result

    TrackTime("Prepare for allocation")
    # No functionality to incorporate 'aborted_at' has been made (so far)
    assert np.all(filtered_issues['aborted_at'].isna())

    # No functionality to incorporate 'reissue_unused_participants_of_coupons_enabled' has been made (so far)
    # print("%d out of %d offers enabled reissued"%(np.sum(filtered_offers['reissue_unused_participants_of_coupons_enabled']), len(filtered_offers)))

    relevant_issue_columns = ['id','offer_id','sent_at','amount','expires_at']
    filtered_issues = filtered_issues[relevant_issue_columns]
    filtered_issues = filtered_issues.sort_values(by='sent_at')

    relevant_offer_columns = ['id', 'category_id', 'accept_time', 'member_criteria_gender', 'member_criteria_min_age',
                              'member_criteria_max_age', 'family_criteria_min_count', 'family_criteria_max_count',
                              'family_criteria_child_age_range_min', 'family_criteria_child_age_range_max',
                              'family_criteria_child_stages_child_stages', 'family_criteria_child_gender',
                              'family_criteria_is_single', 'family_criteria_has_children']
    filtered_offers = copy.copy(filtered_offers[relevant_offer_columns])

    # TODO: 'receive_coupon_after', 'deactivated_at', 'archived_at', 'onboarded_at', 'created_at'
    relevant_member_columns = ['id', 'active', 'member_state', 'mobile', 'date_of_birth', 'gender']
    all_members = all_members[relevant_member_columns]

    # Generate Utilities, the fit of a member to an offer
    nr_agents = len(all_members)
    nr_unique_resources = len(filtered_offers)
    Utility_member_offer = np.random.uniform(0,1,size=(nr_agents, nr_unique_resources))

    return filtered_issues, filtered_offers, all_members, Utility_member_offer


# # Relevant events according to the coupon lifecycle
# class Event(enum.Enum):
#     member_declined     = 0
#     member_accepted     = 1
#     member_let_expire   = 2 # i.e. after 3 days
#     coupon_sent         = 3
#     coupon_expired      = 4 # i.e. after 1 month


def allocate_coupons(issues, offers, members, utility_a_r):
    # members = agents
    # offers = unique resources
    # issues = stream of resources
    TrackTime("Allocate")

    issues = issues.reset_index(drop=True)
    offers = offers.reset_index(drop=True)
    members = members.reset_index(drop=True)

    print(utility_a_r.shape)

    BATCH_SIZE = 10

    max_existing_coupon_id = -1
    max_existing_coupon_follow_id = -1

    events_list = []
    events_df = pd.DataFrame(columns=['event','at','coupon_id','member_id','issue_id','offer_id','coupon_follow_id'])

    unsorted_queue_of_coupons = []

    for i, issue in issues.iterrows():
        TrackTime("print")
        if i%100 == 0:
            print("\rissue nr %d (%.1f%%)"%(i,100*i/len(issues)), end='')

        TrackTime("Making new_coupons df")
        new_coupons = pd.DataFrame(issue['amount']*[issue['sent_at']], columns=['at'])

        new_coupons['coupon_follow_id'] = np.arange(issue['amount']) + max_existing_coupon_follow_id + 1
        max_existing_coupon_follow_id += issue['amount']
        new_coupons['coupon_id'] = np.arange(issue['amount']) + max_existing_coupon_id + 1
        max_existing_coupon_id += issue['amount']

        new_coupons['issue_id'] = issue['id']
        new_coupons['offer_id'] = issue['offer_id']

        # Convert rows of dataframe to list of tuples
        TrackTime("Converting df to list[tuple]")
        to_store = list(map(tuple, new_coupons.values))
        unsorted_queue_of_coupons.extend(to_store)

        if len(unsorted_queue_of_coupons) < BATCH_SIZE:
            continue

        TrackTime("Sorting coupon queue")
        unsorted_queue_of_coupons = sorted(unsorted_queue_of_coupons, key=lambda my_tuple: my_tuple[0])

        TrackTime("Checking if batch ready")
        time_of_sending_next_batch = unsorted_queue_of_coupons[BATCH_SIZE-1][0]
        time_of_next_issue = issues.loc[i+1,'sent_at']
        if time_of_next_issue < time_of_sending_next_batch:
            continue

        print(""); pprint(unsorted_queue_of_coupons)

        batch_to_send = unsorted_queue_of_coupons[:BATCH_SIZE]
        batch_utility = filter_relevant_utilities(batch_to_send, utility_a_r)
        X_a_r = get_allocation(batch_utility)

        # TODO: add 'coupon_sent' events

        # TODO: Simulate if coupon will be accepted, declined, or let expire

        # TODO: If a coupon is not accepted, check whether to add to unsorted_queue_of_coupons
        # TODO: If not, add coupon_expired event
        break



def filter_relevant_utilities(batch_to_send, utility_a_r):
    # TODO: Decrease nr columns of utility_a_r
    # TODO: Decrease nr rows of utility_a_r
    # TODO: Adjust utilities for non-eligible members
    return utility_a_r


def get_allocation(Uar):
    # TODO
    return None


if __name__ == '__main__':
    main()