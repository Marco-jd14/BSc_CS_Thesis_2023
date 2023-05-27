# -*- coding: utf-8 -*-
"""
Created on Wed May 24 20:46:00 2023

@author: Marco
"""

import sys
import copy
import enum
import traceback
import numpy as np
import pandas as pd
import datetime as dt
from pprint import pprint
import matplotlib.pyplot as plt
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

    try:
        preparation = prepare_allocator(db)
        events_df = allocate_coupons(*preparation)
    except:
        print("\n",traceback.format_exc())
        db.close()

    TrackTime("Export")
    export_path = './timelines/events_list_%s.xlsx'%(str(dt.datetime.now().strftime("%Y.%m.%d-%H.%M.%S")))
    events_df.to_excel(export_path, "events")

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

    # # res = filtered_issues
    # res = filtered_issues[['offer_id','sent_at','expires_at','amount','total_issued','nr_accepted']][filtered_issues['sent_at'] == filtered_issues['expires_at']]
    # print(res)
    # print(np.all(res['total_issued'] == 1))
    # print(np.all(res['amount'] == 1))
    # print(np.sum(res['nr_accepted']))

    # to_join = filtered_offers[['id','redeem_till','redeem_type']]
    # res = pd.merge(res, to_join, left_on='offer_id', right_on='id')
    # print(res['redeem_till'].unique())
    # print(res['redeem_type'].unique())
    # print(res.groupby('redeem_type').aggregate(count=('id','count')))
    # print(res.groupby('redeem_till').aggregate(count=('id','count')))
    
    # # print(filtered_issues[['sent_at','expires_at']][filtered_issues['sent_at'] == filtered_issues['expires_at']])
    # sys.exit()

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
    member_accepted     = 0
    member_declined     = 1
    member_let_expire   = 2 # i.e. after 3 days
    coupon_available    = 3
    coupon_sent         = 4
    coupon_expired      = 5 # i.e. after 1 month

    def __lt__(self, other):
        self.value < other.value


def release_new_issue(issue, max_existing_coupon_id, max_existing_coupon_follow_id, verbose):

    TrackTime("Releasing new issue")
    new_coupon_ids        = np.arange(issue['amount']) + max_existing_coupon_id + 1
    new_coupon_follow_ids = np.arange(issue['amount']) + max_existing_coupon_follow_id + 1
    max_existing_coupon_id        += issue['amount']
    max_existing_coupon_follow_id += issue['amount']

    add_to_queue = []
    for new_id, new_follow_id in zip(new_coupon_ids, new_coupon_follow_ids):
        new_coupon = (issue['sent_at'], new_id, new_follow_id, issue['id'], issue['offer_id'])
        add_to_queue.append(new_coupon)

    came_available_coupons = list(map(lambda my_tuple: [Event.coupon_available] + list(my_tuple), add_to_queue))

    return (max_existing_coupon_id, max_existing_coupon_follow_id), (add_to_queue, came_available_coupons)


def simulate_member_responses(batch_utility, member_indices, coupon_indices, coupon_index_to_id,
                              member_index_to_id, batch_to_send, batch_sent_at, offers, verbose):

    TrackTime("Simulating accepted coupons")
    # Simulate if coupon will be accepted or not
    accept_probabilites = batch_utility[member_indices, coupon_indices]
    coupon_accepted = np.random.uniform(0, 1, size=len(accept_probabilites)) < accept_probabilites

    percent_available_time_used = np.random.uniform(0, 1, size=np.sum(coupon_accepted))
    accepted_coupons = []
    for i, (member_index, coupon_index) in enumerate(zip(member_indices[coupon_accepted], coupon_indices[coupon_accepted])):
        assert coupon_index_to_id[coupon_index] == batch_to_send[coupon_index][COUPON_ID_COLUMN]
        accept_time = offers.loc[batch_to_send[coupon_index][OFFER_ID_COLUMN], 'accept_time']
        accept_time = batch_sent_at + dt.timedelta(days=float(accept_time)) * percent_available_time_used[i]

        accepted_coupon = [Event.member_accepted, accept_time] + list(batch_to_send[coupon_index])[1:] + [member_index_to_id[member_index]]
        accepted_coupons.append(accepted_coupon)


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
            # print("let expire:", coupon_index)
            accept_time = offers.loc[batch_to_send[coupon_index][OFFER_ID_COLUMN], 'accept_time']
            expire_time = determine_coupon_checked_expiry_time(batch_sent_at, float(accept_time))
            expire_time = expire_time[0] # Take first of suggested possible expiry times
            event = [Event.member_let_expire, expire_time] + list(batch_to_send[coupon_index])[1:] + [member_index_to_id[member_index]]
        else:
            # print("declined:", coupon_index)
            accept_time = offers.loc[batch_to_send[coupon_index][OFFER_ID_COLUMN], 'accept_time']
            decline_time = batch_sent_at + dt.timedelta(days=float(accept_time)) * percent_available_time_used[i]
            event = [Event.member_declined, decline_time] + list(batch_to_send[coupon_index])[1:] + [member_index_to_id[member_index]]

        not_accepted_coupons.append(event)


    return accepted_coupons, not_accepted_coupons


def re_release_non_accepted_coupons(not_accepted_coupons, issues, max_existing_coupon_id):
    TrackTime("Checking non-accepted coupon-expiry")

    # Check if non-accepted coupons can be re-allocated to new members
    came_available_coupons = []
    expired_coupons = []
    add_to_queue = []
    
    for not_accepted_coupon in not_accepted_coupons:
        # Do a +1 here because the event column in included in the not_accepted_coupon-object
        coupon_expires_at = issues.loc[not_accepted_coupon[ISSUE_ID_COLUMN+1],'expires_at']
        # coupon_expires_at = expire_times.loc[not_accepted_coupon[ISSUE_ID_COLUMN+1]]
        coupon_came_available_at = not_accepted_coupon[TIMESTAMP_COLUMN+1]

        if coupon_came_available_at < coupon_expires_at:
            # Coupon is not expired yet, add back to queue of coupons
            new_coupon_id = max_existing_coupon_id + 1
            max_existing_coupon_id += 1

            event = [Event.coupon_available, not_accepted_coupon[1], new_coupon_id] + not_accepted_coupon[3:-1] + [np.nan]
            came_available_coupons.append(event)
            new_coupon = tuple([not_accepted_coupon[1], new_coupon_id] + not_accepted_coupon[3:-1])
            add_to_queue.append(new_coupon)
        else:
            # Coupon is expired
            event = [Event.coupon_expired, coupon_expires_at, np.nan] + not_accepted_coupon[3:-1] + [np.nan]
            expired_coupons.append(event)

    return max_existing_coupon_id, (came_available_coupons, expired_coupons, add_to_queue)


def send_out_chosen_coupons(BATCH_SIZE, member_indices, coupon_indices, coupon_index_to_id, member_index_to_id,
                            batch_to_send, batch_sent_at):

    TrackTime("Making sent-at events")
    # Add the coupons, that were not allocated, back to the queue
    add_to_queue = []
    non_allocated_coupons = set(np.arange(BATCH_SIZE)) - set(coupon_indices)
    for coupon_index in non_allocated_coupons:
        assert coupon_index_to_id[coupon_index] == batch_to_send[coupon_index][COUPON_ID_COLUMN]
        add_to_queue.append(batch_to_send[coupon_index])

    # TODO: batch_to_send --> remove TIMESTAMP_COLUMN, add MEMBER_ID
    sent_coupons = []
    for member_index, coupon_index in zip(member_indices, coupon_indices):
        assert coupon_index_to_id[coupon_index] == batch_to_send[coupon_index][COUPON_ID_COLUMN]
        sent_coupon = [Event.coupon_sent, batch_sent_at] + list(batch_to_send[coupon_index])[1:] + [member_index_to_id[member_index]]
        sent_coupons.append(sent_coupon)

    return add_to_queue, sent_coupons


def is_batch_ready_to_be_sent(unsorted_queue_of_coupons, BATCH_SIZE, issue_counter, issues):
    if len(unsorted_queue_of_coupons) < BATCH_SIZE:
        return False

    TrackTime("Sorting coupon queue")
    # unsorted_queue_of_coupons = sorted(unsorted_queue_of_coupons, key=lambda my_tuple: my_tuple[TIMESTAMP_COLUMN])
    unsorted_queue_of_coupons.sort(key=lambda my_tuple: my_tuple[TIMESTAMP_COLUMN])

    TrackTime("Checking if batch ready")
    time_of_sending_next_batch = unsorted_queue_of_coupons[BATCH_SIZE-1][TIMESTAMP_COLUMN]
    if issue_counter+1 < len(issues):
        time_of_next_issue = issues['sent_at'].iloc[issue_counter+1]
        if time_of_next_issue < time_of_sending_next_batch:
            # Even though we have enough coupons to reach minimum of BATCH_SIZE,
            # We first have to process another issue, to include in this next batch
            return False

    return True



# def allocate_resources(resources_stream, resources_properties, agents, utility_values, utility_indices):
def allocate_coupons(issues, members, utility_values, utility_indices, supporting_info=None, verbose=False):
    # members = agents
    # offers = unique resources
    # issues = stream of resources
    TrackTime("Allocate")
    print(utility_values.shape)

    # The supporting info could be re-retrieved from the database constantly, but doing it only once improves performance

    BATCH_SIZE = 1

    max_existing_coupon_id = -1
    max_existing_coupon_follow_id = -1

    events_list = []
    events_df = pd.DataFrame(columns=['event','at','coupon_id','coupon_follow_id','issue_id','offer_id','member_id'])

    # Initialize queue and define the columns of an element in the queue
    unsorted_queue_of_coupons = []

    global TIMESTAMP_COLUMN, COUPON_ID_COLUMN, ISSUE_ID_COLUMN, OFFER_ID_COLUMN
    TIMESTAMP_COLUMN, COUPON_ID_COLUMN, ISSUE_ID_COLUMN, OFFER_ID_COLUMN = 0, 1, 3, 4

    unsorted = False

    # Loop over all issues to release
    for issue_counter, (issue_id, issue) in enumerate(issues.iterrows()):
        TrackTime("print")
        if issue_counter%20 == 0:
            print("\rissue nr %d (%.1f%%)"%(issue_counter,100*issue_counter/len(issues)), end='')

        # if issue_counter>1000:
        #     break


        # Release the new issue
        max_coupon_ids, coupon_lists = release_new_issue(issue, max_existing_coupon_id,
                                                         max_existing_coupon_follow_id, verbose)

        # Update the maximum existing ids
        max_existing_coupon_id, max_existing_coupon_follow_id = max_coupon_ids
        # Add the new issue to the queue and events list
        add_to_queue, came_available_coupons = coupon_lists
        unsorted_queue_of_coupons.extend(add_to_queue)
        events_list.extend(came_available_coupons)


        counter = 0
        while is_batch_ready_to_be_sent(unsorted_queue_of_coupons, BATCH_SIZE, issue_counter, issues):
            counter += 1
            # if counter > 1:
            #     print("OMGGG", counter)

            unsorted_queue_of_coupons, events_list, max_existing_coupon_id = send_out_new_batch(issues, members, utility_values, utility_indices,
                                                                                                supporting_info, unsorted_queue_of_coupons, events_list,
                                                                                                BATCH_SIZE, max_existing_coupon_id, verbose)


    TrackTime("Make events df")
    events_df = pd.DataFrame(events_list, columns=events_df.columns)
    events_df = events_df.sort_values(by=['at','event']).reset_index(drop=True)
    return events_df




def send_out_new_batch(issues, members, utility_values, utility_indices, supporting_info,
                       unsorted_queue_of_coupons, events_list, BATCH_SIZE, max_existing_coupon_id, verbose):

    batch_sent_at = unsorted_queue_of_coupons[BATCH_SIZE-1][TIMESTAMP_COLUMN]


    TrackTime("Extracting batch")
    # TODO: send out all coupons that were created before the time at [BATCH_SIZE]
    batch_to_send = unsorted_queue_of_coupons[:BATCH_SIZE]
    unsorted_queue_of_coupons = unsorted_queue_of_coupons[BATCH_SIZE:]

    TrackTime("Checking expiry")
    # TODO: filter on unexpired coupons? Or send out a batch earlier than desired to prevent expiry
    for coupon in batch_to_send:
        if issues.loc[coupon[ISSUE_ID_COLUMN],'expires_at'] == issues.loc[coupon[ISSUE_ID_COLUMN],'sent_at']:
            # If the issue expires the moment the coupons are sent out, we tolerate one round of sending out coupons, even if it is at most a few days after expiry
            assert abs(batch_sent_at - issues.loc[coupon[ISSUE_ID_COLUMN],'expires_at']) < dt.timedelta(days=3)
        else:
            assert batch_sent_at < issues.loc[coupon[ISSUE_ID_COLUMN],'expires_at'], "Trying to send out a coupon at %s which already expired at %s"%(batch_sent_at, issues.loc[coupon[ISSUE_ID_COLUMN],'expires_at'])


    TrackTime("Filtering relevant utilies")
    # TODO: Filter members also based on history of received / pending coupons
    batch_utility, batch_indices = filter_relevant_utilities(batch_to_send, members, utility_values, utility_indices, supporting_info)
    coupon_index_to_id, member_index_to_id = batch_indices


    TrackTime("Determining optimal allocation")
    # Determine allocation of coupons based on utilities
    X_a_r = greedy(batch_utility)
    member_indices, coupon_indices = np.nonzero(X_a_r)


    # Send out the coupons that were allocated to the right members
    unsent_coupons, sent_coupons = send_out_chosen_coupons(BATCH_SIZE, member_indices, coupon_indices, coupon_index_to_id,
                                                           member_index_to_id, batch_to_send, batch_sent_at)


    # Simulate the member responses of those coupons that were sent out
    accepted_coupons, not_accepted_coupons = simulate_member_responses(batch_utility, member_indices, coupon_indices,
                                                                        coupon_index_to_id, member_index_to_id, batch_to_send,
                                                                        batch_sent_at, supporting_info[0], verbose)


    # Check if not-accepted coupons can be re-allocated, or have expired
    max_existing_coupon_id, lists_of_coupons = re_release_non_accepted_coupons(not_accepted_coupons, issues,
                                                                               max_existing_coupon_id)
    came_available_coupons, expired_coupons, coupons_for_re_release = lists_of_coupons

    if verbose:
        print("Sent out:")
        print(sent_coupons)
        print("Accepted:")
        print(accepted_coupons)
        print("Not accepted:")
        print(not_accepted_coupons)
        print("Came newly available:")
        print(came_available_coupons)
        print("Expired:")
        print(expired_coupons)

    # Add some coupons back to the queue
    unsorted_queue_of_coupons.extend(unsent_coupons)
    unsorted_queue_of_coupons.extend(coupons_for_re_release)

    # Add all the events to the events_list
    events_list.extend(sent_coupons)
    events_list.extend(accepted_coupons)
    events_list.extend(not_accepted_coupons)
    events_list.extend(came_available_coupons)
    events_list.extend(expired_coupons)

    return unsorted_queue_of_coupons, events_list, max_existing_coupon_id


def greedy(Uar, verbose=False):
    nr_A, nr_R = Uar.shape
    # print("nr agents: %d,  nr resources: %d"%(nr_A, nr_R))
    assert nr_A >= nr_R
    Xar = np.zeros_like(Uar, dtype=int)

    # sort the utility-matrix with highest utility first
    rows, cols = np.unravel_index(np.flip(np.argsort(Uar, axis=None)), Uar.shape)
    for i, (a, r) in enumerate(zip(rows,cols)):

        nr_members_assigned_to_resource = np.sum(Xar, axis=0)
        if np.all(nr_members_assigned_to_resource > 0):
            if verbose: print("All resources allocated after %d iterations"%i)
            return Xar # Every resource is already allocated

        nr_resources_allocated_to_members = np.sum(Xar, axis=1)
        if np.all(nr_resources_allocated_to_members > 0):
            if verbose: print("All members have a resource after %d iterations"%i)
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
    # print(utility_values.shape, "-->", rel_utility_values.shape)
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
        eligible_members = get_eligible_members_static(        members, offer, all_partners, all_member_categories, tracktime=True)
        eligible_members = get_eligible_members_time_dependent(members, offer, all_partners, all_children, batch_sent_at, tracktime=True)

        # TODO: get eligible_members based on historically received coupons / pending coupons

        offer_id_to_eligible_members[offer_id] = eligible_members['id'].values

    return offer_id_to_eligible_members



if __name__ == '__main__':
    main()