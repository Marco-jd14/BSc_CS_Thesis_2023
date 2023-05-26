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
from make_allocation_timeline import determine_coupon_checked_expiry_time


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
    # Put issue_id as index of the dataframe (instead of 0 until len(df))
    filtered_issues['id_index'] = filtered_issues['id']
    filtered_issues = filtered_issues.set_index('id_index')

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
    print(utility_values.shape)

    # The supporting info could be re-retrieved from the database constantly, but doing it only once improves performance

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
        add_to_queue = list(map(tuple, new_coupons.values))
        unsorted_queue_of_coupons.extend(add_to_queue)

        came_available_coupons = list(map(lambda my_tuple: [Event.coupon_available] + list(my_tuple), add_to_queue))
        events_list.extend(came_available_coupons)
        print("Came available:")
        print(came_available_coupons)

        if len(unsorted_queue_of_coupons) < BATCH_SIZE:
            continue # Not enough coupon in queue to reach minimum of BATCH_SIZE


        TIMESTAMP_COLUMN = 0
        COUPON_ID_COLUMN = 1
        ISSUE_ID_COLUMN  = 3
        OFFER_ID_COLUMN  = 4


        TrackTime("Sorting coupon queue")
        unsorted_queue_of_coupons = sorted(unsorted_queue_of_coupons, key=lambda my_tuple: my_tuple[TIMESTAMP_COLUMN])

        TrackTime("Checking if batch ready")
        time_of_sending_next_batch = unsorted_queue_of_coupons[BATCH_SIZE-1][TIMESTAMP_COLUMN]
        if i+1 < len(issues):
            time_of_next_issue = issues['sent_at'].iloc[i+1]
            if time_of_next_issue < time_of_sending_next_batch:
                # Even though we have enough coupons to reach minimum of BATCH_SIZE,
                # We first have to process another issue, to include in this next batch
                continue


        TrackTime("Filtering relevant issues + offers")
        # TODO: send out all coupons that were created before the time at [BATCH_SIZE]
        batch_to_send = unsorted_queue_of_coupons[:BATCH_SIZE]
        unsorted_queue_of_coupons = unsorted_queue_of_coupons[BATCH_SIZE:]

        # Extract the offers relevant for this batch, from the supporting_info
        relevant_offers = supporting_info[0].loc[np.unique(list(map(lambda my_tuple: my_tuple[OFFER_ID_COLUMN], batch_to_send))),:]
        accept_times = relevant_offers['accept_time']
        # Extract the issue relevant for this batch
        relevant_issues = issues.loc[np.unique(list(map(lambda my_tuple: my_tuple[ISSUE_ID_COLUMN], batch_to_send))),:]
        expire_times = relevant_issues['expires_at']

        # TODO: filter on unexpired coupons? Or send out a batch earlier than desired to prevent expiry
        for coupon in batch_to_send:
            assert time_of_sending_next_batch < expire_times.loc[coupon[ISSUE_ID_COLUMN]], "Trying to send out a coupon at %s which already expired at %s"%(time_of_sending_next_batch, expire_times.loc[coupon[ISSUE_ID_COLUMN]])


        TrackTime("Filtering relevant utilies")
        # TODO: Filter members also based on history of received / pending coupons
        batch_utility, batch_indices = filter_relevant_utilities(batch_to_send, members, utility_values, utility_indices, supporting_info)
        coupon_index_to_id, member_index_to_id = batch_indices

        TrackTime("Determining optimal allocation")
        X_a_r = greedy(batch_utility)
        member_indices, coupon_indices = np.nonzero(X_a_r)


        TrackTime("Making sent-at events")
        # Add the coupon that were not allocated back to the queue
        non_allocated_coupons = set(np.arange(BATCH_SIZE)) - set(coupon_indices)
        for coupon_index in non_allocated_coupons:
            assert coupon_index_to_id[coupon_index] == batch_to_send[coupon_index][COUPON_ID_COLUMN]
            unsorted_queue_of_coupons.append(batch_to_send[coupon_index])


        TrackTime("Making sent-at events")
        # TODO: batch_to_send --> remove TIMESTAMP_COLUMN, add MEMBER_ID
        sent_coupons = []
        for member_index, coupon_index in zip(member_indices, coupon_indices):
            assert coupon_index_to_id[coupon_index] == batch_to_send[coupon_index][COUPON_ID_COLUMN]
            sent_coupon = [Event.coupon_sent, time_of_sending_next_batch] + list(batch_to_send[coupon_index])[1:] + [member_index_to_id[member_index]]
            sent_coupons.append(sent_coupon)

        print("Sent out:")
        print(sent_coupons)
        events_list.append(sent_coupons)


        TrackTime("Simulating accepted coupons")
        # Simulate if coupon will be accepted or not
        accept_probabilites = batch_utility[member_indices, coupon_indices]
        coupon_accepted = np.random.uniform(0, 1, size=len(accept_probabilites)) < accept_probabilites

        percent_available_time_used = np.random.uniform(0, 1, size=np.sum(coupon_accepted))
        accepted_coupons = []
        for i, (member_index, coupon_index) in enumerate(zip(member_indices[coupon_accepted], coupon_indices[coupon_accepted])):
            assert coupon_index_to_id[coupon_index] == batch_to_send[coupon_index][COUPON_ID_COLUMN]
            # TODO: accept time basen offer['accept_time']
            accept_time = accept_times.loc[batch_to_send[coupon_index][OFFER_ID_COLUMN]]
            accept_time = time_of_sending_next_batch + dt.timedelta(days=float(accept_time)) * percent_available_time_used[i]
            accepted_coupon = [Event.member_accepted, accept_time] + list(batch_to_send[coupon_index])[1:] + [member_index_to_id[member_index]]
            accepted_coupons.append(accepted_coupon)

        print("Accepted:")
        print(accepted_coupons)
        events_list.extend(accepted_coupons)


        TrackTime("Simulating non-accepted coupons")
        # Simulate if coupon will be declined or no response (expire)
        P_let_expire = 0.5  # TODO: improve upon P_let_expire?
        coupon_let_expire = np.random.uniform(0, 1, size=np.sum(~coupon_accepted)) < P_let_expire
        # TODO: realistic decline time
        percent_available_time_used = np.random.uniform(0, 1, size=np.sum(~coupon_accepted))

        not_accepted_coupons = []
        for i, (member_index, coupon_index) in enumerate(zip(member_indices[~coupon_accepted], coupon_indices[~coupon_accepted])):
            assert coupon_index_to_id[coupon_index] == batch_to_send[coupon_index][COUPON_ID_COLUMN]
            if coupon_let_expire[i]:
                print("let expire:", coupon_index)
                accept_time = accept_times.loc[batch_to_send[coupon_index][OFFER_ID_COLUMN]]
                expire_time = determine_coupon_checked_expiry_time(time_of_sending_next_batch, float(accept_time))
                event = [Event.member_let_expire, expire_time] + list(batch_to_send[coupon_index])[1:] + [member_index_to_id[member_index]]
            else:
                print("declined:", coupon_index)
                accept_time = accept_times.loc[batch_to_send[coupon_index][OFFER_ID_COLUMN]]
                decline_time = time_of_sending_next_batch + dt.timedelta(days=float(accept_time)) * percent_available_time_used[i]
                event = [Event.member_declined, decline_time] + list(batch_to_send[coupon_index])[1:] + [member_index_to_id[member_index]]

            not_accepted_coupons.append(event)

        print(not_accepted_coupons)
        events_list.extend(not_accepted_coupons)


        TrackTime("Checking non-accepted coupon-expiry")
        # Check if non-accepted coupons can be re-allocated to new members
        came_available_coupons = []
        expired_coupons = []
        for not_accepted_coupon in not_accepted_coupons:
            # Do a +1 here because the event column in included in the not_accepted_coupon-object
            coupon_expires_at = expire_times.loc[not_accepted_coupon[ISSUE_ID_COLUMN+1]]
            coupon_came_available_at = not_accepted_coupon[TIMESTAMP_COLUMN+1]

            if coupon_came_available_at < coupon_expires_at:
                # Coupon is not expired yet, add back to queue of coupons
                event = [Event.coupon_available] + not_accepted_coupon[1:]
                came_available_coupons.append(event)
                add_to_queue = tuple(not_accepted_coupon[1:-1])
                unsorted_queue_of_coupons.append(add_to_queue)
            else:
                # Coupon is expired
                event = [Event.coupon_expired, coupon_expires_at] + not_accepted_coupon[2:-1] + [np.nan]
                expired_coupons.append(event)

        print("Came newly available:")
        print(came_available_coupons)
        events_list.extend(came_available_coupons)

        print("Expired:")
        print(expired_coupons)
        events_list.extend(expired_coupons)

        if i>50:
            break




def greedy(Uar):
    nr_A, nr_R = Uar.shape
    # print("nr agents: %d,  nr resources: %d"%(nr_A, nr_R))
    assert nr_A >= nr_R
    Xar = np.zeros_like(Uar, dtype=int)

    # sort the utility-matrix with highest utility first
    rows, cols = np.unravel_index(np.flip(np.argsort(Uar, axis=None)), Uar.shape)
    for i, (a, r) in enumerate(zip(rows,cols)):

        nr_members_assigned_to_resource = np.sum(Xar, axis=0)
        if np.all(nr_members_assigned_to_resource > 0):
            print("All resources allocated after %d iterations"%i)
            return Xar # Every resource is already allocated

        nr_resources_allocated_to_members = np.sum(Xar, axis=1)
        if np.all(nr_resources_allocated_to_members > 0):
            print("All members have a resource after %d iterations"%i)
            return Xar # Every member already as a resource

        if nr_members_assigned_to_resource[r] > 0:
            continue

        if nr_resources_allocated_to_members[a] == 0:
            Xar[a,r] = 1

    print("This should not get printed")
    return Xar


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

        # TODO: get eligible_members based on historically received coupons / pending coupons

        offer_id_to_eligible_members[offer_id] = eligible_members['id'].values

    return offer_id_to_eligible_members



if __name__ == '__main__':
    main()