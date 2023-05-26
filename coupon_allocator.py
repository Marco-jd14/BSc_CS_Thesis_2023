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
np.random.seed(0)

import database.connect_db as connect_db
import database.query_db as query_db
# Event = query_db.Event

from get_eligible_members import get_eligible_members_static, get_eligible_members_time_dependent


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
    filtered_issues = filtered_issues.sort_values(by='sent_at').reset_index(drop=True)

    relevant_offer_columns = ['id', 'category_id', 'accept_time', 'member_criteria_gender', 'member_criteria_min_age',
                              'member_criteria_max_age', 'family_criteria_min_count', 'family_criteria_max_count',
                              'family_criteria_child_age_range_min', 'family_criteria_child_age_range_max',
                              'family_criteria_child_stages_child_stages', 'family_criteria_child_gender',
                              'family_criteria_is_single', 'family_criteria_has_children']
    filtered_offers = copy.copy(filtered_offers[relevant_offer_columns])

    # TODO: 'receive_coupon_after', 'deactivated_at', 'archived_at', 'onboarded_at', 'created_at'
    relevant_member_columns = ['id', 'active', 'member_state', 'email', 'mobile', 'date_of_birth', 'gender']
    all_members = all_members[relevant_member_columns]

    # email and phone number criteria
    all_members = all_members[~all_members['email'].isna()]
    all_members = all_members[~all_members['mobile'].isna()]

    # Generate Utilities, the fit of a member to an offer
    nr_agents = len(all_members)
    nr_unique_resources = len(filtered_offers)
    utility_values = np.random.uniform(0,0.7,size=(nr_agents, nr_unique_resources))

    # Creates a dictionary from 'id' column to index of dataframe
    member_id_to_index = all_members['id'].reset_index().set_index('id').to_dict()['index']
    offer_id_to_index = filtered_offers['id'].reset_index().set_index('id').to_dict()['index']
    utility_indices = (member_id_to_index, offer_id_to_index)

    # Put offer_id as index of the dataframe (instead of 0 until len(df))
    filtered_offers['id_index'] = filtered_offers['id']
    filtered_offers = filtered_offers.set_index('id_index')

    query = "select * from member_category"
    all_member_categories = pd.read_sql_query(query, db)
    query = "select * from member_family_member where type='child'"
    all_children = pd.read_sql_query(query, db)
    query = "select * from member_family_member where type='partner'"
    all_partners = pd.read_sql_query(query, db)
    supporting_info = (filtered_offers, all_member_categories, all_children, all_partners)

    return filtered_issues, all_members, utility_values, utility_indices, supporting_info


# Relevant events according to the coupon lifecycle
class Event(enum.Enum):
    member_declined     = 0
    member_accepted     = 1
    member_let_expire   = 2 # i.e. after 3 days
    coupon_available    = 5
    coupon_sent         = 3
    coupon_expired      = 4 # i.e. after 1 month


# def allocate_resources(resources_stream, resources_properties, agents, utility_values, utility_indices):
def allocate_coupons(issues, members, utility_values, utility_indices, supporting_info=None):
    # members = agents
    # offers = unique resources
    # issues = stream of resources
    TrackTime("Allocate")

    # The supporting info could be re-retrieved from the database constantly, but doing it only once improves performance
    offers, _, _, _ = supporting_info

    print(utility_values.shape)
    assert utility_values.shape == (len(members), len(offers))


    BATCH_SIZE = 10

    max_existing_coupon_id = -1
    max_existing_coupon_follow_id = -1

    events_list = []
    events_df = pd.DataFrame(columns=['event','at','coupon_id','coupon_follow_id','issue_id','offer_id','member_id'])

    unsorted_queue_of_coupons = []

    for i, issue in issues.iterrows():
        TrackTime("print")
        if i%100 == 0:
            print("\rissue nr %d (%.1f%%)"%(i,100*i/len(issues)), end='')

        TrackTime("Making new_coupons df")
        new_coupons = pd.DataFrame(issue['amount']*[issue['sent_at']], columns=['available_from'])

        TrackTime("Filling new_coupons df")
        new_coupons['coupon_id'] = np.arange(issue['amount']) + max_existing_coupon_id + 1
        max_existing_coupon_id += issue['amount']
        new_coupons['coupon_follow_id'] = np.arange(issue['amount']) + max_existing_coupon_follow_id + 1
        max_existing_coupon_follow_id += issue['amount']

        new_coupons['issue_id'] = issue['id']
        new_coupons['offer_id'] = issue['offer_id']

        # Convert rows of dataframe to list of tuples
        TrackTime("Converting df to list[tuple]")
        to_store = list(map(tuple, new_coupons.values))
        unsorted_queue_of_coupons.extend(to_store)

        # TODO: Event.coupon_available

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

        TrackTime("Filtering relevant utilies")
        batch_to_send = unsorted_queue_of_coupons[:BATCH_SIZE]
        batch_utility, batch_indices = filter_relevant_utilities(batch_to_send, members, utility_values, utility_indices, supporting_info)


def filter_relevant_utilities(batch_to_send, members, utility_values, utility_indices, supporting_info):
    member_id_to_index, offer_id_to_index = utility_indices

    # Decrease nr columns of utility_values based on relevant offers
    OFFER_ID_COLUMN = -1
    offer_ids_to_send = list(map(lambda coupon_tuple: coupon_tuple[OFFER_ID_COLUMN], batch_to_send))
    offer_indices_to_send = list(map(lambda offer_id: offer_id_to_index[offer_id], offer_ids_to_send))
    rel_utility_values = utility_values[:, offer_indices_to_send]


    # Determine for every offer, the set of eligible members
    offer_id_to_eligible_members = get_all_eligible_members(batch_to_send, members, supporting_info)


    # Make one big set of eligible members
    all_eligible_member_ids = set()
    for offer_id, eligible_member_list in offer_id_to_eligible_members.items():
        all_eligible_member_ids = all_eligible_member_ids.union(eligible_member_list)

    # Adjust utilities for non-eligible members to -1
    offer_ids_to_send = np.array(offer_ids_to_send)
    for offer_id, eligible_member_list in offer_id_to_eligible_members.items():

        non_eligible_members = all_eligible_member_ids - set(eligible_member_list)
        if len(non_eligible_members) == 0:
            continue # nothing to adjust

        non_eligible_members_indices = np.array(list(map(lambda member_id: member_id_to_index[member_id], non_eligible_members )))

        col_indices_to_adjust = np.where(offer_ids_to_send == offer_id)[0]
        assert len(col_indices_to_adjust) > 0

        rel_utility_values[non_eligible_members_indices, col_indices_to_adjust] = -1


    # Decrease nr rows of utility_values based on eligible members
    all_eligible_member_ids = list(all_eligible_member_ids)
    all_eligible_member_indices = list(map(lambda member_id: member_id_to_index[member_id], all_eligible_member_ids))
    rel_utility_values = rel_utility_values[all_eligible_member_indices, :]


    COUPON_ID_COLUMN = 1
    coupon_index_to_id = list(map(lambda coupon_tuple: coupon_tuple[COUPON_ID_COLUMN], batch_to_send))
    member_index_to_id = all_eligible_member_ids

    assert rel_utility_values.shape == (len(member_index_to_id), len(coupon_index_to_id))
    print(rel_utility_values.shape, "-->", utility_values.shape)
    return rel_utility_values, (coupon_index_to_id, member_index_to_id)


def get_all_eligible_members(batch_to_send, members, supporting_info):
    TrackTime("get all eligible members")
    TIMESTAMP_COLUMN = 0
    batch_sent_at = batch_to_send[-1][TIMESTAMP_COLUMN]
    offers, all_member_categories, all_children, all_partners = supporting_info

    # Phone nr and email criteria already in effect
    # member must be active to be eligible
    members = members[members['active'] == 1]

    offer_ids_to_send = np.unique(list(map(lambda coupon_tuple: coupon_tuple[-1], batch_to_send)))
    offer_id_to_eligible_members = {}

    for offer_id in offer_ids_to_send:
        TrackTime("Get offer based on ID")
        # offer = offers[offers['id'] == offer_id].squeeze()
        offer = offers.loc[offer_id, :].squeeze()
        TrackTime("get all eligible members")
        eligible_members = get_eligible_members_static(        members, offer, all_partners, all_member_categories, tracktime=False)
        eligible_members = get_eligible_members_time_dependent(members, offer, all_partners, all_children, batch_sent_at, tracktime=False)

        offer_id_to_eligible_members[offer_id] = eligible_members['id'].values

    return offer_id_to_eligible_members

def get_allocation(Uar):
    # TODO
    return None


if __name__ == '__main__':
    main()