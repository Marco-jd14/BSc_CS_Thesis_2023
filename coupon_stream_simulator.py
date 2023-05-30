# -*- coding: utf-8 -*-
"""
Created on Wed May 24 20:46:00 2023

@author: Marco
"""

import os
import re
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

from get_eligible_members import get_all_eligible_members
import database.connect_db as connect_db
import database.query_db as query_db
Event = query_db.Event

from allocator_algorithms import greedy

np.random.seed(0)

def main():
    TrackTime("Connect to db")
    conn = connect_db.establish_host_connection()
    db   = connect_db.establish_database_connection(conn)
    print("Successfully connected to database '%s'"%str(db.engine).split("/")[-1][:-1])

    BATCH_SIZE = 5
    export_folder = './timelines/'

    # convert_events_pkl_to_excel(4, export_folder)
    # sys.exit()

    try:
        preparation = prepare_simulation_data(db)
        events_df = simulate_coupon_allocations(BATCH_SIZE, greedy, *preparation)

        TrackTime("Export")
        export_results(events_df, preparation[2], preparation[3], export_folder)

        print("")
        TrackReport()
    except:
        print("\n",traceback.format_exc())
        db.close()

        print("")
        TrackReport()

        print("\n!!\nCould not finish making timeline\n!!")


def export_results(events_df, utility_values, utility_indices, export_folder):

    member_id_to_index, offer_id_to_index = utility_indices

    member_id_to_index = pd.DataFrame.from_dict(member_id_to_index, orient='index', columns=['index'])
    member_id_to_index = member_id_to_index.reset_index().rename(columns={'level_0':'member_id'}).sort_values(by='index')
    offer_id_to_index = pd.DataFrame.from_dict(offer_id_to_index, orient='index', columns=['index'])
    offer_id_to_index = offer_id_to_index.reset_index().rename(columns={'level_0':'offer_id'}).sort_values(by='index')

    assert np.all(offer_id_to_index['index'].values == np.arange(len(offer_id_to_index)))
    assert np.all(member_id_to_index['index'].values == np.arange(len(member_id_to_index)))

    utility_df = pd.DataFrame(utility_values, index=member_id_to_index['member_id'].values, columns=offer_id_to_index['offer_id'].values)


    # version_to_read = 2
    # contents = os.listdir(export_folder)
    # contents = list(filter(lambda name: re.search("^%d_.*\.pkl"%version_to_read, name), contents))
    # events_df  = pd.read_pickle(export_folder + (contents[0] if "events_df" in contents[0] else contents[1]))
    # utility_df = pd.read_pickle(export_folder + (contents[0] if "utility_df" in contents[0] else contents[1]))

    time = str(dt.datetime.now().strftime("%Y.%m.%d-%H.%M.%S"))
    contents = os.listdir(export_folder)
    contents = list(filter(lambda name: re.search("^[0-9]+_.*\.pkl", name), contents))
    versions = list(map(lambda name: int(name.split('_')[0]), contents))
    next_version = max(versions) + 1

    events_df.to_pickle(export_folder + '%d_events_df_%s.pkl'%(next_version, time))
    utility_df.to_pickle(export_folder + '%d_utility_df_%s.pkl'%(next_version, time))

    if False:
        convert_events_pkl_to_excel(next_version, export_folder)


def convert_events_pkl_to_excel(version, export_folder):
    contents = os.listdir(export_folder)
    contents = list(filter(lambda name: re.search("^%d_events_df.*\.pkl"%version, name), contents))
    events_df  = pd.read_pickle(export_folder + contents[0])
    time = os.path.splitext(contents[0])[0].split("_")[-1]

    writer = pd.ExcelWriter(export_folder + "%d_events_list_%s.xlsx"%(version, time))
    events_df.to_excel(writer, "events")
    writer.close()


# Batch column definitions
TIMESTAMP_COLUMN, COUPON_ID_COLUMN, ISSUE_ID_COLUMN, OFFER_ID_COLUMN = 0, 1, 3, 4
# Historical context column definitions
ACCEPTED_LAST_COUPON_AT, LET_LAST_COUPON_EXPIRE_AT, SET_OF_RECEIVED_OFFER_IDS = 0, 1, 2


# def allocate_resources(resources_stream, resources_properties, agents, utility_values, utility_indices):
def simulate_coupon_allocations(batch_size, get_allocation, issues, members, utility_values, utility_indices, supporting_info=None, verbose=False):
    # members = agents
    # offers = unique resources
    # issues = stream of resources
    # The supporting info could be re-retrieved from the database constantly, but doing it only once improves performance

    TrackTime("Allocate")

    # To be able to easily generate new unique coupon (follow) ids
    max_existing_coupon_id = -1
    max_existing_coupon_follow_id = -1

    # The list / df of events which will be exported in the end
    events_list = []
    events_df = pd.DataFrame(columns=['event','at','coupon_id','coupon_follow_id','issue_id','offer_id','member_id'])

    # Initialize queue and define the columns of an element in the queue
    unsorted_queue_of_coupons = []
    prev_batch_unsent_coupons = []

    # Initialize historical context and the 3 values based on member-id (key)
    historical_context = {member_id: [None, None, set()] for member_id in members['id'].values}

    print("\nStarting simulation\n")

    # Loop over all issues to release
    batch_counter = 0
    for issue_counter, (issue_id, issue) in enumerate(issues.iterrows()):
        TrackTime("print")
        if issue_counter%20 == 0:
            print("\rissue nr %d (%.1f%%)     batch nr %d"%(issue_counter,100*issue_counter/len(issues), batch_counter), end='')

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
        while is_batch_ready_to_be_sent(unsorted_queue_of_coupons, batch_size, issue_counter, issues):
            batch_counter += 1
            if batch_counter%10 == 0:
                print("\rissue nr %d (%.1f%%)     batch nr %d"%(issue_counter,100*issue_counter/len(issues), batch_counter), end='')

            # Generate events for the next batch
            result = send_out_new_batch(get_allocation, issues, members, utility_values, utility_indices, supporting_info,
                                        historical_context, events_list, unsorted_queue_of_coupons, batch_size,
                                        prev_batch_unsent_coupons, max_existing_coupon_id, verbose)

            # Unpack the return values
            unsorted_queue_of_coupons, events_list, historical_context, \
                prev_batch_unsent_coupons, max_existing_coupon_id = result


    # Turn the events_list into a dataframe and sort it
    TrackTime("Make events df")
    events_df = pd.DataFrame(events_list, columns=events_df.columns)
    events_df = events_df.sort_values(by=['at','coupon_follow_id','event']).reset_index(drop=True)
    events_df['at'] = events_df['at'].apply(lambda time: time.replace(microsecond=0))
    return events_df



def send_out_new_batch(get_allocation, issues, members, utility_values, utility_indices, supporting_info,
                       historical_context, events_list, unsorted_queue_of_coupons, batch_size,
                       prev_batch_unsent_coupons, max_existing_coupon_id, verbose):

    batch_sent_at = unsorted_queue_of_coupons[batch_size-1][TIMESTAMP_COLUMN]

    # Check if we can try to resend coupons, or if they have already expired
    unsent_coupons_to_retry, unsent_coupons_now_expired = check_expiry_unsent_coupons(batch_sent_at, issues,
                                                                                      prev_batch_unsent_coupons)

    if len(unsent_coupons_now_expired) > 0:
        offer_ids = list(map(lambda coupon: coupon[OFFER_ID_COLUMN+1], unsent_coupons_now_expired))
        if verbose: print("\n%d coupons expired without ever getting sent to an eligible member (offer-id: %s)"%(len(unsent_coupons_now_expired), str(list(set(offer_ids)))))
    events_list.extend(unsent_coupons_now_expired)


    TrackTime("Extracting batch")
    # actual_batch_size = batch_size
    actual_batch_size = batch_size-1
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

    # Prepare the matrix of utilities for the allocation-algorithm(s)
    batch_utility, batch_indices = filter_relevant_utilities(batch_to_send, members, utility_values, utility_indices, 
                                                             supporting_info, historical_context)
    coupon_index_to_id, member_index_to_id = batch_indices

    # Zero eligible members
    if len(member_index_to_id) == 0:
        unsent_coupons = batch_to_send
        if verbose: print("\nCould not find any eligible members to send the batch to")
        return unsorted_queue_of_coupons, events_list, historical_context, unsent_coupons, max_existing_coupon_id


    TrackTime("Determining optimal allocation")
    # Determine allocation of coupons based on utilities
    X_a_r = get_allocation(batch_utility, verbose)
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


    if len(unsent_coupons) == len(batch_to_send):
        print("\nDid not allocate any coupons from last batch")
    if verbose:
        if len(unsent_coupons) > 0:
            print("\nCould not allocate %d out of %d coupons"%(len(unsent_coupons),len(batch_to_send)))

    return unsorted_queue_of_coupons, events_list, historical_context, unsent_coupons, max_existing_coupon_id


def is_batch_ready_to_be_sent(unsorted_queue_of_coupons, batch_size, issue_counter, issues):
    if len(unsorted_queue_of_coupons) < batch_size:
        return False

    TrackTime("Sorting coupon queue")
    # unsorted_queue_of_coupons = sorted(unsorted_queue_of_coupons, key=lambda my_tuple: my_tuple[TIMESTAMP_COLUMN])
    unsorted_queue_of_coupons.sort(key=lambda my_tuple: my_tuple[TIMESTAMP_COLUMN])

    TrackTime("Checking if batch ready")
    time_of_sending_next_batch = unsorted_queue_of_coupons[batch_size-1][TIMESTAMP_COLUMN]
    if issue_counter+1 < len(issues):
        time_of_next_issue = issues['sent_at'].iloc[issue_counter+1]
        if time_of_next_issue < time_of_sending_next_batch:
            # Even though we have enough coupons to reach minimum of batch_size,
            # We first have to process another issue, to include in this next batch
            return False

    return True


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
    # TODO: evaluate 'redeem_till' if 'redeem_type' in ['on_date', 'til_date']
    TrackTime("Process prev batch unsent coupons")
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


############# PREPARATORY METHOD BEFORE CALLING AN ALLOCATOR_ALGORITHM #########################################

def filter_relevant_utilities(batch_to_send, members, utility_values, utility_indices, supporting_info, historical_context):
    TrackTime("Filtering relevant utilies")
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


############# PROTOCOL FOR DETERMINING EXPIRY TIME OF COUPON #########################################

def determine_coupon_checked_expiry_time(created_at, accept_time, check=False):
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
        if check and checked_expiry.date() == dt.date(2021, 10, 29):
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
            if check and expired.date() == dt.date(2022, 9, 28):
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


############# RETRIEVE DATA FROM DATABASE HERE FOR EASE OF ACCESS LATER #########################################

def prepare_simulation_data(db):
    TrackTime("Retrieve from db")
    result = query_db.retrieve_from_sql_db(db, 'filtered_issues', 'filtered_offers', 'member')
    filtered_issues, filtered_offers, all_members = result

    TrackTime("Prepare for allocation")
    # No functionality to incorporate 'aborted_at' has been made (so far)
    assert np.all(filtered_issues['aborted_at'].isna())

    # No functionality to incorporate 'reissue_unused_participants_of_coupons_enabled' has been made (so far)
    # print("%d out of %d offers enabled reissued"%(np.sum(filtered_offers['reissue_unused_participants_of_coupons_enabled']), len(filtered_offers)))

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



if __name__ == '__main__':
    main()