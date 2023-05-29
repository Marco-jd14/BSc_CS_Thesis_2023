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
from collections import Counter
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

        TrackTime("Export")
        export_path = './timelines/events_list_%s.xlsx'%(str(dt.datetime.now().strftime("%Y.%m.%d-%H.%M.%S")))
        events_df.to_excel(export_path, "events")

        print("")
        TrackReport()
    except:
        print("\n",traceback.format_exc())
        db.close()

        print("")
        TrackReport()

        print("\n!!\nCould not finish making timeline\n!!")




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


def send_out_chosen_coupons(member_indices, coupon_indices, coupon_index_to_id, member_index_to_id,
                            batch_to_send, batch_sent_at):

    TrackTime("Making sent-at events")
    # Add the coupons, that were not allocated, back to the queue
    add_to_queue = []
    non_allocated_coupons = set(np.arange(len(batch_to_send))) - set(coupon_indices)
    for coupon_index in non_allocated_coupons:
        assert coupon_index_to_id[coupon_index] == batch_to_send[coupon_index][COUPON_ID_COLUMN]
        add_to_queue.append(batch_to_send[coupon_index])

    sent_coupons = []
    for member_index, coupon_index in zip(member_indices, coupon_indices):
        assert coupon_index_to_id[coupon_index] == batch_to_send[coupon_index][COUPON_ID_COLUMN]
        sent_coupon = [Event.coupon_sent, batch_sent_at] + list(batch_to_send[coupon_index])[1:] + [member_index_to_id[member_index]]
        sent_coupons.append(sent_coupon)

    return add_to_queue, sent_coupons


def check_expiry_unsent_coupons(batch_sent_at, issues, prev_batch_unsent_coupons):
    unsent_coupons_to_retry, unsent_coupons_now_expired = [], []

    for coupon in prev_batch_unsent_coupons:
        if abs(issues.loc[coupon[ISSUE_ID_COLUMN],'expires_at'] - issues.loc[coupon[ISSUE_ID_COLUMN],'sent_at']) < dt.timedelta(seconds=3):
            # If the issue expires the moment the coupons are sent out, we tolerate one round of sending out coupons, even if it is at most a few days after expiry
            if abs(batch_sent_at - issues.loc[coupon[ISSUE_ID_COLUMN],'expires_at']) < dt.timedelta(days=3):
                unsent_coupons_to_retry.append(coupon)
            else:
                event = [Event.coupon_expired, issues.loc[coupon[ISSUE_ID_COLUMN],'expires_at'], np.nan] + list(coupon[2:]) + [np.nan]
                unsent_coupons_now_expired.append(event)
        else:
            if batch_sent_at < issues.loc[coupon[ISSUE_ID_COLUMN],'expires_at']:
                unsent_coupons_to_retry.append(coupon)
            else:
                event = [Event.coupon_expired, issues.loc[coupon[ISSUE_ID_COLUMN],'expires_at'], np.nan] + list(coupon[2:]) + [np.nan]
                unsent_coupons_now_expired.append(event)

    return unsent_coupons_to_retry, unsent_coupons_now_expired


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

    BATCH_SIZE = 5

    max_existing_coupon_id = -1
    max_existing_coupon_follow_id = -1

    events_list = []
    events_df = pd.DataFrame(columns=['event','at','coupon_id','coupon_follow_id','issue_id','offer_id','member_id'])

    # Initialize queue and define the columns of an element in the queue
    unsorted_queue_of_coupons = []
    prev_batch_unsent_coupons = []

    global TIMESTAMP_COLUMN, COUPON_ID_COLUMN, ISSUE_ID_COLUMN, OFFER_ID_COLUMN
    TIMESTAMP_COLUMN, COUPON_ID_COLUMN, ISSUE_ID_COLUMN, OFFER_ID_COLUMN = 0, 1, 3, 4

    # Define historical context and the 4 indices of a value based on member-id (key)
    historical_context = {member_id: [None, None, set()] for member_id in members['id'].values}

    global ACCEPTED_LAST_COUPON_AT, LET_LAST_COUPON_EXPIRE_AT, SET_OF_RECEIVED_OFFER_IDS
    ACCEPTED_LAST_COUPON_AT, LET_LAST_COUPON_EXPIRE_AT, SET_OF_RECEIVED_OFFER_IDS = 0, 1, 2

    # Loop over all issues to release
    batch_counter = 0
    for issue_counter, (issue_id, issue) in enumerate(issues.iterrows()):
        TrackTime("print")
        if issue_counter%20 == 0:
            print("\rissue nr %d (%.1f%%)     batch nr %d"%(issue_counter,100*issue_counter/len(issues), batch_counter), end='')

        # if issue_counter > 1000:
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

        # Send out the next batch while we have enough coupons
        while is_batch_ready_to_be_sent(unsorted_queue_of_coupons, BATCH_SIZE, issue_counter, issues):
            if batch_counter%10 == 0:
                print("\rissue nr %d (%.1f%%)     batch nr %d"%(issue_counter,100*issue_counter/len(issues), batch_counter), end='')
            batch_counter += 1

            result = send_out_new_batch(issues, members, utility_values, utility_indices, supporting_info,
                                        historical_context, events_list, unsorted_queue_of_coupons, BATCH_SIZE,
                                        prev_batch_unsent_coupons, max_existing_coupon_id, verbose)

            # Unpack the return values
            unsorted_queue_of_coupons, events_list, historical_context, \
                prev_batch_unsent_coupons, max_existing_coupon_id = result


    TrackTime("Make events df")
    events_df = pd.DataFrame(events_list, columns=events_df.columns)
    events_df = events_df.sort_values(by=['at','coupon_follow_id','event']).reset_index(drop=True)
    return events_df


# multiset_unsent_coupons = Counter()

def send_out_new_batch(issues, members, utility_values, utility_indices, supporting_info, historical_context,
                       events_list, unsorted_queue_of_coupons, BATCH_SIZE, prev_batch_unsent_coupons, 
                       max_existing_coupon_id, verbose):

    batch_sent_at = unsorted_queue_of_coupons[BATCH_SIZE-1][TIMESTAMP_COLUMN]

    TrackTime("Process prev batch unsent coupons")
    unsent_coupons_to_retry, unsent_coupons_now_expired = check_expiry_unsent_coupons(batch_sent_at, issues,
                                                                                      prev_batch_unsent_coupons)

    if len(unsent_coupons_now_expired) > 0:
        offer_ids = list(map(lambda coupon: coupon[OFFER_ID_COLUMN+1], unsent_coupons_now_expired))
        print("\n%d coupons expired without ever getting sent to an eligible member (offer-id: %s)"%(len(unsent_coupons_now_expired), str(list(set(offer_ids)))))
    events_list.extend(unsent_coupons_now_expired)


    TrackTime("Extracting batch")
    # actual_batch_size = BATCH_SIZE
    actual_batch_size = BATCH_SIZE-1
    while True:
        if len(unsorted_queue_of_coupons) == actual_batch_size:
            break
        if unsorted_queue_of_coupons[actual_batch_size][TIMESTAMP_COLUMN] - batch_sent_at > dt.timedelta(seconds=3):
            break
        actual_batch_size += 1

    batch_to_send = unsent_coupons_to_retry + unsorted_queue_of_coupons[:actual_batch_size]
    unsorted_queue_of_coupons = unsorted_queue_of_coupons[actual_batch_size:]

    if verbose: print("batch:"); pprint(batch_to_send)

    # TrackTime("Checking expiry")
    # # TODO: filter on unexpired coupons? Or send out a batch earlier than desired to prevent expiry
    # for coupon in batch_to_send:
    #     if abs(issues.loc[coupon[ISSUE_ID_COLUMN],'expires_at'] - issues.loc[coupon[ISSUE_ID_COLUMN],'sent_at']) < dt.timedelta(seconds=3):
    #         # If the issue expires the moment the coupons are sent out, we tolerate one round of sending out coupons, even if it is at most a few days after expiry
    #         assert abs(batch_sent_at - issues.loc[coupon[ISSUE_ID_COLUMN],'expires_at']) < dt.timedelta(days=3)
    #     else:
    #         assert batch_sent_at < issues.loc[coupon[ISSUE_ID_COLUMN],'expires_at'], "Trying to send out a coupon at %s which already expired at %s"%(batch_sent_at, issues.loc[coupon[ISSUE_ID_COLUMN],'expires_at'])


    TrackTime("Filtering relevant utilies")
    batch_utility, batch_indices = filter_relevant_utilities(batch_to_send, members, utility_values, utility_indices, 
                                                             supporting_info, historical_context)
    coupon_index_to_id, member_index_to_id = batch_indices

    # Zero eligible members
    if len(member_index_to_id) == 0:
        unsent_coupons = batch_to_send
        print("\nCould not find any eligible members to send the batch to")
        return unsorted_queue_of_coupons, events_list, historical_context, unsent_coupons, max_existing_coupon_id


    TrackTime("Determining optimal allocation")
    # Determine allocation of coupons based on utilities
    X_a_r = greedy(batch_utility, verbose)
    member_indices, coupon_indices = np.nonzero(X_a_r)
    assert len(set(member_indices)) == len(member_indices), "One member got more than one coupons?"

    # Send out the coupons that were allocated to the right members
    unsent_coupons, sent_coupons = send_out_chosen_coupons(member_indices, coupon_indices, coupon_index_to_id,
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
        print("Not sent out:")
        print(unsent_coupons)
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
    unsorted_queue_of_coupons.extend(coupons_for_re_release)

    # Add all the events to the events_list
    events_list.extend(sent_coupons)
    events_list.extend(accepted_coupons)
    events_list.extend(not_accepted_coupons)
    events_list.extend(came_available_coupons)
    events_list.extend(expired_coupons)

    assert len(batch_to_send) == len(sent_coupons) + len(unsent_coupons), "%d != %d (%d + %d)"%(len(batch_to_send), len(sent_coupons) + len(unsent_coupons), len(sent_coupons), len(unsent_coupons))
    assert len(sent_coupons) == len(accepted_coupons) + len(not_accepted_coupons), "%d != %d (%d + %d)"%(len(sent_coupons), len(accepted_coupons) + len(not_accepted_coupons), len(accepted_coupons), len(not_accepted_coupons))
    assert len(not_accepted_coupons) == len(came_available_coupons) + len(expired_coupons), "%d != %d (%d + %d)"%(len(not_accepted_coupons), len(came_available_coupons) + len(expired_coupons), len(came_available_coupons), len(expired_coupons))

    # Update historical context
    for accepted_coupon in accepted_coupons:
        member_id = accepted_coupon[-1]
        historical_context[member_id][ACCEPTED_LAST_COUPON_AT]  = accepted_coupon[1+TIMESTAMP_COLUMN]
        historical_context[member_id][SET_OF_RECEIVED_OFFER_IDS].add(accepted_coupon[1+OFFER_ID_COLUMN])
    for not_accepted_coupon in not_accepted_coupons:
        if not_accepted_coupon[0] == Event.member_let_expire:
            member_id = not_accepted_coupon[-1]
            historical_context[member_id][LET_LAST_COUPON_EXPIRE_AT]  = not_accepted_coupon[1+TIMESTAMP_COLUMN]
            historical_context[member_id][SET_OF_RECEIVED_OFFER_IDS].add(not_accepted_coupon[1+OFFER_ID_COLUMN])


    # unsent_coupon_ids = list(map(lambda unsent_coupon: unsent_coupon[COUPON_ID_COLUMN], unsent_coupons))
    # multiset_unsent_coupons.update(unsent_coupon_ids)

    # coupon_ids_unsent_more_than_5 = list(filter(lambda coupon_id: multiset_unsent_coupons[coupon_id] >= 5, unsent_coupon_ids))
    # sent_coupon_ids = list(map(lambda sent_coupon: sent_coupon[COUPON_ID_COLUMN+1], sent_coupons))
    # for coupon_id in coupon_ids_unsent_more_than_5:
    #     if coupon_id in sent_coupon_ids:
    #         print("Finally sent out a coupon after %d tries"%(multiset_unsent_coupons[coupon_id]))


    if len(unsent_coupons) == len(batch_to_send):
        print("\nDid not allocate any coupons from last batch")
    if verbose:
        if len(unsent_coupons) > 0:
            print("\nCould not allocate %d out of %d coupons"%(len(unsent_coupons),len(batch_to_send)))

    return unsorted_queue_of_coupons, events_list, historical_context, unsent_coupons, max_existing_coupon_id


def greedy(Uar, verbose=False):
    nr_A, nr_R = Uar.shape
    if not nr_A >= nr_R:
        if verbose: print("\nLess eligible members than resources: %d !>= %d"%(nr_A,nr_R))
    Xar = np.zeros_like(Uar, dtype=int)

    nr_non_eligible_members = np.sum(Uar < 0)
    nr_eligible_options = np.prod(Uar.shape) - nr_non_eligible_members

    # sort the utility-matrix with highest utility first
    rows, cols = np.unravel_index(np.flip(np.argsort(Uar, axis=None)), Uar.shape)
    for i, (a, r) in enumerate(zip(rows,cols)):
        if i >= nr_eligible_options:
            # Do not allocate coupons when utilities are negative (aka members not eligible)
            if verbose: print("Not all resources could be allocated after %d iterations"%i)
            return Xar

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

    if verbose: print("Last coupon allocated on last iteration")
    assert i == np.prod(Uar.shape) - 1, "iteration %d != %d (shape=%s)"%(i, np.prod(Uar.shape) - 1, str(Uar.shape))
    assert np.all(np.sum(Xar, axis=0) > 0) or np.all(np.sum(Xar, axis=1) > 0), "Not all resources allocated and not all members got a resource"
    return Xar


def filter_relevant_utilities(batch_to_send, members, utility_values, utility_indices, supporting_info, historical_context):
    member_id_to_index, offer_id_to_index = utility_indices

    # Decrease nr columns of utility_values based on relevant offers
    offer_ids_to_send = list(map(lambda coupon_tuple: coupon_tuple[OFFER_ID_COLUMN], batch_to_send))
    offer_indices_to_send = list(map(lambda offer_id: offer_id_to_index[offer_id], offer_ids_to_send))
    rel_utility_values = utility_values[:, offer_indices_to_send]


    TrackTime("Process eligible members")
    # Determine for every offer, the set of eligible members
    offer_id_to_eligible_members = get_all_eligible_members(batch_to_send, members, supporting_info, historical_context)

    TrackTime("Process eligible members")
    # Make one big set of eligible members
    all_eligible_member_ids = set()
    for offer_id, eligible_member_list in offer_id_to_eligible_members.items():
        all_eligible_member_ids = all_eligible_member_ids.union(eligible_member_list)

    TrackTime("Adjusting non-eligible utilities")
    # Adjust utilities for non-eligible members to -1
    offer_ids_to_send = np.array(offer_ids_to_send)
    for offer_id, eligible_member_list in offer_id_to_eligible_members.items():

        non_eligible_members = all_eligible_member_ids - set(eligible_member_list)
        if len(non_eligible_members) == 0:
            continue # nothing to adjust

        non_eligible_members_indices = np.array(list(map(lambda member_id: member_id_to_index[member_id], non_eligible_members )))

        col_indices_to_adjust = np.where(offer_ids_to_send == offer_id)[0]
        assert len(col_indices_to_adjust) > 0, "%d !> 0"%len(col_indices_to_adjust)

        indices_to_adjust = np.ix_(non_eligible_members_indices, col_indices_to_adjust)
        rel_utility_values[indices_to_adjust] = -1


    TrackTime("Filtering relevant utilies")
    # Decrease nr rows of utility_values based on eligible members
    all_eligible_member_ids = list(all_eligible_member_ids)
    all_eligible_member_indices = list(map(lambda member_id: member_id_to_index[member_id], all_eligible_member_ids))
    rel_utility_values = rel_utility_values[all_eligible_member_indices, :]


    coupon_index_to_id = list(map(lambda coupon_tuple: coupon_tuple[COUPON_ID_COLUMN], batch_to_send))
    member_index_to_id = all_eligible_member_ids

    assert rel_utility_values.shape == (len(member_index_to_id), len(coupon_index_to_id)), "%s != %s"%(str(rel_utility_values.shape), str((len(member_index_to_id), len(coupon_index_to_id))))
    return rel_utility_values, (coupon_index_to_id, member_index_to_id)


def get_all_eligible_members(batch_to_send, members, supporting_info, historical_context):
    batch_sent_at = batch_to_send[-1][TIMESTAMP_COLUMN]
    offers, all_member_categories, all_children, all_partners = supporting_info

    # Phone nr and email criteria already in effect
    # member must be active to be eligible
    members = members[members['active'] == 1]

    offer_ids_to_send = Counter(list(map(lambda coupon_tuple: coupon_tuple[OFFER_ID_COLUMN], batch_to_send)))
    offer_id_to_eligible_members = {}

    for offer_id, nr_coupons_to_send in offer_ids_to_send.items():
        offer = offers.loc[offer_id, :].squeeze()

        TrackTime("get all eligible members static")
        eligible_members = get_eligible_members_static(        members, offer, all_partners, all_member_categories, tracktime=False)

        TrackTime("get all eligible members history")
        eligible_members = get_eligible_members_history(       eligible_members, offer_id, nr_coupons_to_send, historical_context, batch_sent_at, phase_one=True)

        TrackTime("get all eligible members time")
        eligible_members = get_eligible_members_time_dependent(eligible_members, offer, all_partners, all_children, batch_sent_at, tracktime=False)

        TrackTime("get all eligible members history")
        eligible_members = get_eligible_members_history(       eligible_members, offer_id, nr_coupons_to_send, historical_context, batch_sent_at, phase_one=False)

        TrackTime("Process eligible members")
        # if len(eligible_members) == 0:
        #     print("\nCoupons from offer_id %d have 0 eligible members"%offer_id)
        offer_id_to_eligible_members[offer_id] = eligible_members['id'].values

    return offer_id_to_eligible_members


def get_eligible_members_history(members, offer_id, nr_coupons_to_send, historical_context, batch_sent_at, phase_one=True):
    def let_coupon_expire_last_month(member_id):
        let_last_coupon_expire_at = historical_context[member_id][LET_LAST_COUPON_EXPIRE_AT]
        if let_last_coupon_expire_at is None:
            return False
        return (batch_sent_at - let_last_coupon_expire_at).days <= 30
        # return batch_sent_at - let_last_coupon_expire_at < dt.timedelta(days=30)

    def accepted_coupon_last_month(member_id):
        accepted_last_coupon_at = historical_context[member_id][ACCEPTED_LAST_COUPON_AT]
        if accepted_last_coupon_at is None:
            return False
        return (batch_sent_at - accepted_last_coupon_at).days <= 30
        # return batch_sent_at - accepted_last_coupon_at <= dt.timedelta(days=30)

    def has_outstanding_coupon(member_id):
        accepted_last_coupon_at = historical_context[member_id][ACCEPTED_LAST_COUPON_AT]
        let_last_coupon_expire_at = historical_context[member_id][LET_LAST_COUPON_EXPIRE_AT]
        dates = []
        if accepted_last_coupon_at is not None:
            dates.append(accepted_last_coupon_at)
        if let_last_coupon_expire_at is not None:
            dates.append(let_last_coupon_expire_at)
        if len(dates) == 0:
            return False
        last_date = max(dates)
        # If last_accepted or last_let_expire is in the future, the member is yet to respond to the outstanding coupon with that response
        return batch_sent_at < last_date

    # ACCEPTED_LAST_COUPON_AT, LET_LAST_COUPON_EXPIRE_AT, SET_OF_RECEIVED_OFFER_IDS
    # TODO: get eligible_members based on historically received coupons / pending coupons
    if phase_one:
        TrackTime("already received offer")
        members_who_already_received_this_offer = \
            list(filter(lambda member_id: offer_id in historical_context[member_id][SET_OF_RECEIVED_OFFER_IDS], historical_context.keys()))
        members = members[~members['id'].isin(members_who_already_received_this_offer)]

        TrackTime("recently let coupon expire")
        members_who_let_coupon_expire_in_last_month = \
            list(filter(let_coupon_expire_last_month, historical_context.keys()))
        members = members[~members['id'].isin(members_who_let_coupon_expire_in_last_month)]
        return members

    else:
        if len(members) <= nr_coupons_to_send:
            # print("No point in filtering, as %d is already <= %d"%(len(members), nr_coupons_to_send))
            return members

        TrackTime("recently accepted coupon")
        # members_to_filter = list(filter(lambda member_id: historical_context[member_id][ACCEPTED_LAST_COUPON_AT] is not None, historical_context.keys()))
        members_who_accepted_coupon_in_last_month = \
            list(filter(accepted_coupon_last_month, historical_context.keys()))

        TrackTime("has outstanding coupon")
        members_with_outstanding = \
            list(filter(has_outstanding_coupon, historical_context.keys()))

        TrackTime("check whether to filter")
        potentially_ineligible_members = list(set(members_with_outstanding).union(set(members_who_accepted_coupon_in_last_month)))
        filtered_members = members[~members['id'].isin(potentially_ineligible_members)]
        if len(filtered_members) >= nr_coupons_to_send:
            return filtered_members
        else:
            # print("Too little eligible members, otherwise %d --> %d < %d"%(len(members), len(filtered_members), nr_coupons_to_send))
            return members
    
    # print("filtered out %d members based on historical context"%(len(members_who_already_received_this_offer) + len(members_who_let_coupon_expire_in_last_month)))


if __name__ == '__main__':
    main()