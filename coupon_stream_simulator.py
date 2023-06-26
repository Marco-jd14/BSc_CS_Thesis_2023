# -*- coding: utf-8 -*-
"""
Created on Wed May 24 20:46:00 2023

@author: Marco
"""

import os
import re
import sys
import enum
import copy
import json
import pickle
import numpy as np
import pandas as pd
import datetime as dt
from pprint import pprint
import matplotlib.pyplot as plt
from database.lib.tracktime import TrackTime, TrackReport

from get_eligible_members import get_all_eligible_members
import database.connect_db as connect_db
import database.query_db as query_db
Event = query_db.Event

from allocator_algorithms import greedy, max_sum_utility, maximin_utility

np.random.seed(0)

global_folder = './timelines/'
run_folder = './timelines/run_%d/'


allocator_algorithms = {'greedy':           greedy,
                        'max_sum_utility':  max_sum_utility,
                        'maximin_utility':  maximin_utility,
                        }

class Util_Type(enum.Enum):
    full_historical     = 0
    partial_historical  = 1
    time_discounted     = 2


def main():
    # convert_events_pkl_to_excel(57,0)
    # sys.exit()

    USE_DB = False
    if USE_DB:
        conn = connect_db.establish_host_connection()
        db   = connect_db.establish_database_connection(conn)
        print("Successfully connected to database '%s'"%str(db.engine).split("/")[-1][:-1])

        util_prep, data_prep = prepare_simulation_data(db)
        db.close()
    else:
        util_prep, data_prep = prepare_simulation_data(None)


    NR_SIMULATIONS = 50
    ALLOCATOR_ALGORITHMS = ['greedy']
    UTILITY_TYPES = [Util_Type.full_historical]
    BATCH_SIZES = [1]

    # for batch_size, alloc_alg, util_type in zip(BATCH_SIZES, ALLOCATOR_ALGORITHMS, UTILITY_TYPES):
    for alloc_alg in ALLOCATOR_ALGORITHMS:
        for batch_size in BATCH_SIZES:
            for util_type in UTILITY_TYPES:
                # try:
                    run_experiment(data_prep, util_prep, batch_size, alloc_alg, util_type, NR_SIMULATIONS)
                # except:
                #     TrackReport()


def run_experiment(data_prep, util_prep, batch_size, alloc_alg_name, util_type, NR_SIMULATIONS):
    run_folders = list(filter(lambda name: os.path.isdir(global_folder + name), os.listdir(global_folder)))
    def extract_run_nr(folder_name):
        try:    return int(folder_name.split("_")[-1])
        except: return 0
    run_nr = max(list(map(extract_run_nr, run_folders))) + 1 if len(run_folders) > 0 else 1

    run_info = {'batch_size':           batch_size,
                'allocator_algorithm':  alloc_alg_name,
                'utility_type':         str(util_type).replace('Util_Type.',''),
                'version_info':         '',
                'version_tag':          '',
                }
    export_run_info(*util_prep, run_info, run_nr)


    start_time = dt.datetime.now().replace(microsecond=0)
    for sim_nr in range(NR_SIMULATIONS):
        print("\nStarting simulation %d (run %d) at %s (%s later)"%(sim_nr, run_nr, dt.datetime.now().replace(microsecond=0), str(dt.datetime.now() - start_time).split('.')[0] ))
        start_time = dt.datetime.now().replace(microsecond=0)

        TrackTime("Other")
        events_df = simulate_coupon_allocations(batch_size, allocator_algorithms[alloc_alg_name], util_type, *util_prep, *data_prep)
        TrackTime("Export")
        export_timeline(events_df, sim_nr, run_nr)

        sim_info = {'runtime':str(dt.datetime.now() - start_time).split('.')[0]}
        with open(run_folder%run_nr + '%d.%d_sim_info.json'%(run_nr, sim_nr), 'w') as fp:
            json.dump(sim_info, fp)


    print("\nFinished all simulations at %s (%s later)"%( dt.datetime.now().replace(microsecond=0), str(dt.datetime.now() - start_time).split('.')[0] ))

    print("")
    TrackReport()
    print("\n")


def export_timeline(events_df, sim_nr, run_nr):
    folder = run_folder%run_nr
    if not os.path.exists(folder):
        os.makedirs(folder)

    events_df.to_pickle(folder + '%d.%d_events_df.pkl'%(run_nr, sim_nr))

    if False:
        convert_events_pkl_to_excel(run_nr, sim_nr)


def export_run_info(utility_values, utility_indices, run_info, run_nr):
    folder = run_folder%run_nr
    if not os.path.exists(folder):
        os.makedirs(folder)

    member_id_to_index, offer_id_to_index = utility_indices

    member_id_to_index = pd.DataFrame.from_dict(member_id_to_index, orient='index', columns=['index'])
    member_id_to_index = member_id_to_index.reset_index().rename(columns={'level_0':'member_id'}).sort_values(by='index')
    offer_id_to_index = pd.DataFrame.from_dict(offer_id_to_index, orient='index', columns=['index'])
    offer_id_to_index = offer_id_to_index.reset_index().rename(columns={'level_0':'offer_id'}).sort_values(by='index')

    assert np.all(offer_id_to_index['index'].values == np.arange(len(offer_id_to_index)))
    assert np.all(member_id_to_index['index'].values == np.arange(len(member_id_to_index)))

    utility_df = pd.DataFrame(utility_values, index=member_id_to_index['member_id'].values, columns=offer_id_to_index['offer_id'].values)

    utility_df.to_pickle(folder + '%d_utility_df.pkl'%(run_nr))
    with open(folder + '%d_info.json'%(run_nr), 'w') as fp:
        json.dump(run_info, fp)



def convert_events_pkl_to_excel(run_nr, sim_nr):
    contents = os.listdir(run_folder%run_nr)
    contents = list(filter(lambda name: re.search("^%d.%d_events_df.pkl"%(run_nr, sim_nr), name), contents))
    events_df  = pd.read_pickle(run_folder%run_nr + contents[0])

    writer = pd.ExcelWriter(run_folder%run_nr + "%d.%d_events_list.xlsx"%(run_nr, sim_nr))
    events_df.to_excel(writer, "events")
    writer.close()


# Batch column definitions
TIMESTAMP_COLUMN, COUPON_ID_COLUMN, ISSUE_ID_COLUMN, OFFER_ID_COLUMN = 0, 1, 3, 4
# Historical context column definitions
OFFER_CONTEXT, MEMBER_CONTEXT, DISCOUNTED_UTILITIES = 0, 1, 2
# Member context column definitions
ACCEPTED_LAST_COUPON_AT, LET_LAST_COUPON_EXPIRE_AT = 0, 1

TIME_DISCOUNT_RATIO = 0.1 ** (1/30)  # After 30 days, 10% of utility remains


# def allocate_resources(resources_stream, resources_properties, agents, utility_values, utility_indices):
def simulate_coupon_allocations(batch_size, get_allocation, util_type, utility_values, utility_indices, issues, members,
                                supporting_info, distribution_info, verbose=False):
    # members = agents
    # offers = unique resources
    # issues = stream of resources
    # The supporting info could be re-retrieved from the database constantly, but doing it only once improves performance


    # To be able to easily generate new unique coupon (follow) ids
    max_existing_coupon_id = -1
    max_existing_coupon_follow_id = -1

    # The list / df of events which will be exported in the end
    events_list = []
    events_df = pd.DataFrame(columns=['event','at','coupon_id','coupon_follow_id','issue_id','offer_id','member_id','batch_id','expiry_type'])

    # Initialize queue and define the columns of an element in the queue
    unsorted_queue_of_coupons = []
    prev_batch_unsent_coupons = []
    prev_batch_time = None

    # Initialize historical context and the 3 values based on member-id (key)
    offers = supporting_info[0]
    offers_context = {offer_id: set() for offer_id in offers['id'].values}
    member_context = {member_id: [-np.inf, -np.inf] for member_id in members['id'].values}
    time_discounted_total_U_per_member = np.zeros(len(members))
    historical_context = (offers_context, member_context, time_discounted_total_U_per_member)

    # Loop over all issues to release
    batch_counter = 0
    for issue_counter, (issue_id, issue) in enumerate(issues.iterrows()):
        TrackTime("Releasing new issue")
        if issue_counter%20 == 0:
            print("\rissue nr %d (%.1f%%)     batch nr %d"%(issue_counter,100*issue_counter/len(issues), batch_counter), end='')

        if issue_counter == 0:
            sim_start_time = issue['sent_at']

        # Release the new issue
        max_coupon_ids, coupon_lists = release_new_issue(issue, max_existing_coupon_id, max_existing_coupon_follow_id)

        # Update the maximum existing ids
        max_existing_coupon_id, max_existing_coupon_follow_id = max_coupon_ids
        # Add the new issue to the queue and events list
        add_to_queue, came_available_coupons = coupon_lists
        unsorted_queue_of_coupons.extend(add_to_queue)
        events_list.extend(came_available_coupons)

        # Send out the next batch while we have enough coupons
        while is_batch_ready_to_be_sent(unsorted_queue_of_coupons, batch_size, issue_counter, issues):
            TrackTime("Sending out batch")
            batch_counter += 1
            if batch_counter%10 == 0:
                print("\rissue nr %d (%.1f%%)     batch nr %d"%(issue_counter,100*issue_counter/len(issues), batch_counter), end='')

            # Generate events for the next batch
            result = send_out_new_batch(get_allocation, issues, members, utility_values, utility_indices, supporting_info, distribution_info,
                                        historical_context, util_type, events_list, unsorted_queue_of_coupons, batch_size, batch_counter,
                                        prev_batch_unsent_coupons, prev_batch_time, max_existing_coupon_id, sim_start_time, verbose)

            # Unpack the return values
            unsorted_queue_of_coupons, events_list, historical_context, \
                prev_batch_unsent_coupons, prev_batch_time, max_existing_coupon_id = result


    # Send out last batch
    if len(unsorted_queue_of_coupons) > 0:
        result = send_out_new_batch(get_allocation, issues, members, utility_values, utility_indices, supporting_info, distribution_info,
                                    historical_context, util_type, events_list, unsorted_queue_of_coupons, len(unsorted_queue_of_coupons), batch_counter+1,
                                    prev_batch_unsent_coupons, prev_batch_time, max_existing_coupon_id, sim_start_time, verbose)
        # Unpack the return values
        _, events_list, _, _, _, _ = result


    # Turn the events_list into a dataframe and sort it
    TrackTime("Make events df")
    events_df = pd.DataFrame(events_list, columns=events_df.columns)
    events_df = events_df.sort_values(by=['at','coupon_follow_id','event']).reset_index(drop=True)
    events_df['at'] = events_df['at'].apply(lambda time: time.replace(microsecond=0))
    return events_df


def send_out_new_batch(get_allocation, issues, members, utility_values, utility_indices, supporting_info, distribution_info,
                       historical_context, util_type, events_list, unsorted_queue_of_coupons, batch_size, batch_ID,
                       prev_batch_unsent_coupons, prev_batch_time, max_existing_coupon_id, sim_start_time, verbose):
    if verbose: print("\nBatch", batch_ID)
    TrackTime("Sending out batch")

    batch_sent_at = unsorted_queue_of_coupons[batch_size-1][TIMESTAMP_COLUMN]
    if prev_batch_time is None:
        prev_batch_time = batch_sent_at

    days_passed_since_prev_batch = int((batch_sent_at - prev_batch_time).total_seconds() / 60 / 60 / 24)


    # Check if we can try to resend coupons, or if they have already expired
    unsent_coupons_to_retry, unsent_coupons_now_expired = check_expiry_unsent_coupons(batch_sent_at, issues,
                                                                                      prev_batch_unsent_coupons)

    if len(unsent_coupons_now_expired) > 0:
        offer_ids = list(map(lambda coupon: coupon[OFFER_ID_COLUMN+1], unsent_coupons_now_expired))
        if verbose: print("\n%d coupons expired while waiting to send out next batch (offer-ids: %s)"%(len(unsent_coupons_now_expired), str(list(set(offer_ids)))))
    events_list.extend(unsent_coupons_now_expired)


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

    if verbose: print("retrying %d from previous batch, and %d new coupons"%(len(unsent_coupons_to_retry), actual_batch_size)) # print("batch:"); pprint(batch_to_send)

    # TrackTime("Checking expiry")
    # # TODO: low-priority --> filter on unexpired coupons? Or send out a batch earlier than desired to prevent expiry
    # for coupon in batch_to_send:
    #     if abs(issues.loc[coupon[ISSUE_ID_COLUMN],'expires_at'] - issues.loc[coupon[ISSUE_ID_COLUMN],'sent_at']) < dt.timedelta(seconds=3):
    #         # If the issue expires the moment the coupons are sent out, we tolerate one round of sending out coupons, even if it is at most a few days after expiry
    #         assert abs(batch_sent_at - issues.loc[coupon[ISSUE_ID_COLUMN],'expires_at']) < dt.timedelta(days=3)
    #     else:
    #         assert batch_sent_at < issues.loc[coupon[ISSUE_ID_COLUMN],'expires_at'], "Trying to send out a coupon at %s which already expired at %s"%(batch_sent_at, issues.loc[coupon[ISSUE_ID_COLUMN],'expires_at'])

    # Prepare the matrix of utilities for the allocation-algorithm(s)
    utility_to_allocate, actual_utility, batch_indices = filter_relevant_utilities(batch_to_send, members, utility_values, utility_indices, 
                                                                                   supporting_info, historical_context, util_type, sim_start_time)
    coupon_index_to_id, member_index_to_id = batch_indices

    # Zero eligible members
    if len(member_index_to_id) == 0:
        unsent_coupons = batch_to_send
        if verbose: print("\nCould not find any eligible members to send the batch to")
        return unsorted_queue_of_coupons, events_list, historical_context, unsent_coupons, batch_sent_at, max_existing_coupon_id

    # Remove coupons without any eligible members from batch_utility for a speed-up
    coupons_without_eligible_members = np.all(utility_to_allocate == -1, axis=0)
    if np.any(coupons_without_eligible_members):
        # old_utility_to_allocate = utility_to_allocate
        utility_to_allocate = utility_to_allocate[:,~coupons_without_eligible_members]

    TrackTime("Determining optimal allocation")
    # Determine allocation of coupons based on utilities
    # start = dt.datetime.now()
    # print(" Start:", start.time())
    X_a_r = get_allocation(utility_to_allocate, verbose)
    # print(" Duration:", (dt.datetime.now() - start))

    # Add the coupons without eligible members back
    if np.any(coupons_without_eligible_members):
        new_X_a_r = np.zeros(shape=(len(member_index_to_id), len(coupon_index_to_id)))
        new_X_a_r[:,~coupons_without_eligible_members] = X_a_r[:,:]

        X_a_r = new_X_a_r
        assert X_a_r.shape == actual_utility.shape
        # utility_to_allocate = old_utility_to_allocate

    assert X_a_r.shape == (len(member_index_to_id), len(coupon_index_to_id))

    member_indices, coupon_indices = np.nonzero(X_a_r)
    assert len(set(member_indices)) == len(member_indices), "One member got more than one coupon?"
    TrackTime("Send out batch & simulate")

    # Send out the coupons that were allocated to the right members
    unsent_coupons, sent_coupons = send_out_chosen_coupons(member_indices, coupon_indices, coupon_index_to_id,
                                                           member_index_to_id, batch_to_send, batch_sent_at, batch_ID)


    # Simulate the member responses of those coupons that were sent out
    accepted_coupons, not_accepted_coupons = simulate_member_responses(actual_utility, member_indices, coupon_indices,
                                                                       coupon_index_to_id, member_index_to_id, batch_to_send,
                                                                       batch_sent_at, supporting_info[0], distribution_info, verbose)


    # Check if not-accepted coupons can be re-allocated, or have expired
    max_existing_coupon_id, lists_of_coupons = re_release_non_accepted_coupons(not_accepted_coupons, issues,
                                                                               max_existing_coupon_id)
    came_available_coupons, expired_coupons, coupons_for_re_release_to_queue = lists_of_coupons

    if verbose:
        print("%d coupons not sent out"%len(unsent_coupons), end='\t')
        print("%d coupons sent out"%len(sent_coupons), end='\t')
        print("%d coupons accepted"%len(accepted_coupons), end='\t')
        print("%d coupons not accepted"%len(not_accepted_coupons), end='\t')
        print("%d coupons came newly available"%len(came_available_coupons), end='\t')
        print("%d coupons expired"%len(expired_coupons))

    # Add some coupons back to the queue
    unsorted_queue_of_coupons.extend(coupons_for_re_release_to_queue)

    # Add all the events to the events_list
    events_list.extend(sent_coupons)
    events_list.extend(accepted_coupons)
    events_list.extend(not_accepted_coupons)
    events_list.extend(came_available_coupons)
    events_list.extend(expired_coupons)

    assert len(batch_to_send) == len(sent_coupons) + len(unsent_coupons), "%d != %d (%d + %d)"%(len(batch_to_send), len(sent_coupons) + len(unsent_coupons), len(sent_coupons), len(unsent_coupons))
    assert len(sent_coupons) == len(accepted_coupons) + len(not_accepted_coupons), "%d != %d (%d + %d)"%(len(sent_coupons), len(accepted_coupons) + len(not_accepted_coupons), len(accepted_coupons), len(not_accepted_coupons))
    assert len(not_accepted_coupons) == len(came_available_coupons) + len(expired_coupons), "%d != %d (%d + %d)"%(len(not_accepted_coupons), len(came_available_coupons) + len(expired_coupons), len(came_available_coupons), len(expired_coupons))

    TrackTime("Update historical context")
    historical_context = update_historical_context(historical_context, utility_values, utility_indices, sent_coupons, accepted_coupons,
                                                   not_accepted_coupons, days_passed_since_prev_batch, sim_start_time)

    if len(unsent_coupons) == len(batch_to_send):
        print("\nDid not allocate any coupons from last batch")
    if verbose:
        if len(unsent_coupons) > 0:
            print("\nCould not allocate %d out of %d coupons"%(len(unsent_coupons),len(batch_to_send)))

    return unsorted_queue_of_coupons, events_list, historical_context, unsent_coupons, batch_sent_at, max_existing_coupon_id


def update_historical_context(historical_context, utility_values, utility_indices, sent_coupons, accepted_coupons,
                              not_accepted_coupons, days_passed_since_prev_batch, sim_start_time):
    # Update time-discounted utilities with the number of days_passed_since_prev_batch
    historical_context[DISCOUNTED_UTILITIES][:] = historical_context[DISCOUNTED_UTILITIES] * (TIME_DISCOUNT_RATIO ** days_passed_since_prev_batch)
    member_id_to_index, offer_id_to_index = utility_indices

    # Update historical offer context
    for sent_coupon in sent_coupons:
        member_id = sent_coupon[-2]
        offer_id = sent_coupon[1+OFFER_ID_COLUMN]
        historical_context[OFFER_CONTEXT][offer_id].add(member_id)
    # Update historical member context: last_accepted_at
    for accepted_coupon in accepted_coupons:
        member_id = accepted_coupon[-2]
        accepted_at_day_nr = (accepted_coupon[1+TIMESTAMP_COLUMN] - sim_start_time).total_seconds() / 60 / 60 / 24
        historical_context[MEMBER_CONTEXT][member_id][ACCEPTED_LAST_COUPON_AT] = accepted_at_day_nr

        # Add utility of accepted coupon to time-discounted sum of utilities
        offer_id = accepted_coupon[1+OFFER_ID_COLUMN]
        member_index, offer_index = member_id_to_index[member_id], offer_id_to_index[offer_id]
        coupon_utility = utility_values[member_index, offer_index]

        historical_context[DISCOUNTED_UTILITIES][member_index] += coupon_utility

    # Update historical member context: last_let_expire_at
    for not_accepted_coupon in not_accepted_coupons:
        if not_accepted_coupon[0] == Event.member_let_expire:
            member_id = not_accepted_coupon[-2]
            let_expire_at_day_nr = (not_accepted_coupon[1+TIMESTAMP_COLUMN] - sim_start_time).total_seconds() / 60 / 60 / 24
            historical_context[MEMBER_CONTEXT][member_id][LET_LAST_COUPON_EXPIRE_AT] = let_expire_at_day_nr

            # Add utility of let_expire coupon to time-discounted sum of utilities (as form of punishment)
            offer_id = not_accepted_coupon[1+OFFER_ID_COLUMN]
            member_index, offer_index = member_id_to_index[member_id], offer_id_to_index[offer_id]
            coupon_utility = utility_values[member_index, offer_index]

            historical_context[DISCOUNTED_UTILITIES][member_index] += coupon_utility

    return historical_context


def is_batch_ready_to_be_sent(unsorted_queue_of_coupons, batch_size, issue_counter, issues):
    if len(unsorted_queue_of_coupons) < batch_size:
        return False

    # unsorted_queue_of_coupons = sorted(unsorted_queue_of_coupons, key=lambda my_tuple: my_tuple[TIMESTAMP_COLUMN])
    unsorted_queue_of_coupons.sort(key=lambda my_tuple: my_tuple[TIMESTAMP_COLUMN])

    time_of_sending_next_batch = unsorted_queue_of_coupons[batch_size-1][TIMESTAMP_COLUMN]
    if issue_counter+1 < len(issues):
        time_of_next_issue = issues['sent_at'].iloc[issue_counter+1]
        if time_of_next_issue < time_of_sending_next_batch:
            # Even though we have enough coupons to reach minimum of batch_size,
            # We first have to process another issue, to include in this next batch
            return False

    return True


def release_new_issue(issue, max_existing_coupon_id, max_existing_coupon_follow_id):

    new_coupon_ids        = np.arange(issue['amount']) + max_existing_coupon_id + 1
    new_coupon_follow_ids = np.arange(issue['amount']) + max_existing_coupon_follow_id + 1
    max_existing_coupon_id        += issue['amount']
    max_existing_coupon_follow_id += issue['amount']

    add_to_queue = []
    for new_id, new_follow_id in zip(new_coupon_ids, new_coupon_follow_ids):
        new_coupon = (issue['sent_at'], new_id, new_follow_id, issue['id'], issue['offer_id'])
        add_to_queue.append(new_coupon)

    came_available_coupons = list(map(lambda my_tuple: [Event.coupon_available] + list(my_tuple) + [np.nan, np.nan], add_to_queue))

    return (max_existing_coupon_id, max_existing_coupon_follow_id), (add_to_queue, came_available_coupons)


def simulate_member_responses(batch_utility, member_indices, coupon_indices, coupon_index_to_id,
                              member_index_to_id, batch_to_send, batch_sent_at, offers, distribution_info, verbose):

    # Simulate if coupon will be accepted or not
    accept_probabilites = batch_utility[member_indices, coupon_indices]
    coupon_accepted = np.random.uniform(0, 1, size=len(accept_probabilites)) < accept_probabilites

    percent_available_time_used = np.random.uniform(0, 1, size=np.sum(coupon_accepted))
    accepted_coupons = []
    for i, (member_index, coupon_index) in enumerate(zip(member_indices[coupon_accepted], coupon_indices[coupon_accepted])):
        assert coupon_index_to_id[coupon_index] == batch_to_send[coupon_index][COUPON_ID_COLUMN]
        accept_time = offers.loc[batch_to_send[coupon_index][OFFER_ID_COLUMN], 'accept_time']
        accept_time = batch_sent_at + dt.timedelta(days=float(accept_time)) * percent_available_time_used[i]

        accepted_coupon = [Event.member_accepted, accept_time] + list(batch_to_send[coupon_index])[1:] + [member_index_to_id[member_index], np.nan]
        accepted_coupons.append(accepted_coupon)


    # Simulate if coupon will be declined or no response (expire)
    P_let_expire_given_not_accepted, valid_decline_times = distribution_info
    # TODO: improve upon P_let_expire --> member dependent?
    P_let_expire = P_let_expire_given_not_accepted
    coupon_let_expire = np.random.uniform(0, 1, size=np.sum(~coupon_accepted)) < P_let_expire

    # Draw from realistic decline times
    random_indices = np.random.randint(len(valid_decline_times), size=np.sum(~coupon_accepted))
    # percent_available_time_used = np.random.uniform(0, 1, size=np.sum(~coupon_accepted))

    not_accepted_coupons = []
    for i, (member_index, coupon_index) in enumerate(zip(member_indices[~coupon_accepted], coupon_indices[~coupon_accepted])):
        assert coupon_index_to_id[coupon_index] == batch_to_send[coupon_index][COUPON_ID_COLUMN]
        if coupon_let_expire[i]:
            accept_time = offers.loc[batch_to_send[coupon_index][OFFER_ID_COLUMN], 'accept_time']
            expire_time = determine_coupon_checked_expiry_time(batch_sent_at, float(accept_time))
            expire_time = expire_time[0] # Take first of suggested possible expiry times
            event = [Event.member_let_expire, expire_time] + list(batch_to_send[coupon_index])[1:] + [member_index_to_id[member_index], np.nan]
        else:
            accept_time = offers.loc[batch_to_send[coupon_index][OFFER_ID_COLUMN], 'accept_time']
            decline_time = batch_sent_at + dt.timedelta(days=float(accept_time)) * valid_decline_times[random_indices[i]]#percent_available_time_used[i]
            event = [Event.member_declined, decline_time] + list(batch_to_send[coupon_index])[1:] + [member_index_to_id[member_index], np.nan]

        not_accepted_coupons.append(event)


    return accepted_coupons, not_accepted_coupons


def re_release_non_accepted_coupons(not_accepted_coupons, issues, max_existing_coupon_id):

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

            event = [Event.coupon_available, not_accepted_coupon[1], new_coupon_id] + not_accepted_coupon[3:-2] + [np.nan, np.nan]
            came_available_coupons.append(event)
            new_coupon = tuple([not_accepted_coupon[1], new_coupon_id] + not_accepted_coupon[3:-2])
            add_to_queue.append(new_coupon)
        else:
            # Coupon is expired
            event = [Event.coupon_expired, coupon_expires_at, np.nan] + not_accepted_coupon[3:-2] + [np.nan, np.nan]
            expired_coupons.append(event)

    return max_existing_coupon_id, (came_available_coupons, expired_coupons, add_to_queue)


def send_out_chosen_coupons(member_indices, coupon_indices, coupon_index_to_id, member_index_to_id,
                            batch_to_send, batch_sent_at, batch_ID):

    # Add the coupons, that were not allocated, back to the queue
    add_to_queue = []
    non_allocated_coupons = set(np.arange(len(batch_to_send))) - set(coupon_indices)
    for coupon_index in non_allocated_coupons:
        assert coupon_index_to_id[coupon_index] == batch_to_send[coupon_index][COUPON_ID_COLUMN]
        add_to_queue.append(batch_to_send[coupon_index])

    sent_coupons = []
    for member_index, coupon_index in zip(member_indices, coupon_indices):
        assert coupon_index_to_id[coupon_index] == batch_to_send[coupon_index][COUPON_ID_COLUMN]
        sent_coupon = [Event.coupon_sent, batch_sent_at] + list(batch_to_send[coupon_index])[1:] + [member_index_to_id[member_index], batch_ID]
        sent_coupons.append(sent_coupon)

    return add_to_queue, sent_coupons


def check_expiry_unsent_coupons(batch_sent_at, issues, prev_batch_unsent_coupons):
    # TODO: evaluate 'redeem_till' if 'redeem_type' in ['on_date', 'til_date']
    unsent_coupons_to_retry, unsent_coupons_now_expired = [], []

    for coupon in prev_batch_unsent_coupons:
        if abs(issues.loc[coupon[ISSUE_ID_COLUMN],'expires_at'] - issues.loc[coupon[ISSUE_ID_COLUMN],'sent_at']) < dt.timedelta(seconds=3):
            # If the issue expires the moment the coupons are sent out, we tolerate one round of sending out coupons, even if it is at most a few days after expiry
            if abs(batch_sent_at - issues.loc[coupon[ISSUE_ID_COLUMN],'expires_at']) < dt.timedelta(days=3):
                unsent_coupons_to_retry.append(coupon)
            else:
                event = [Event.coupon_expired, issues.loc[coupon[ISSUE_ID_COLUMN],'expires_at'], np.nan] + list(coupon[2:]) + [np.nan, np.nan, "while awaiting batch"]
                unsent_coupons_now_expired.append(event)
        else:
            if batch_sent_at < issues.loc[coupon[ISSUE_ID_COLUMN],'expires_at']:
                unsent_coupons_to_retry.append(coupon)
            else:
                event = [Event.coupon_expired, issues.loc[coupon[ISSUE_ID_COLUMN],'expires_at'], np.nan] + list(coupon[2:]) + [np.nan, np.nan, "while awaiting batch"]
                unsent_coupons_now_expired.append(event)

    return unsent_coupons_to_retry, unsent_coupons_now_expired


############# PREPARATORY METHOD BEFORE CALLING AN ALLOCATOR_ALGORITHM #########################################

def filter_relevant_utilities(batch_to_send, members, utility_values, utility_indices, supporting_info,
                              historical_context, util_type, sim_start_time):

    TrackTime("Filtering relevant utilies")
    member_id_to_index, offer_id_to_index = utility_indices

    # Decrease nr columns of utility_values based on relevant offers
    offer_ids_to_send = list(map(lambda coupon_tuple: coupon_tuple[OFFER_ID_COLUMN], batch_to_send))
    offer_indices_to_send = list(map(lambda offer_id: offer_id_to_index[offer_id], offer_ids_to_send))
    utility_to_allocate = utility_values[:, offer_indices_to_send]
    actual_utility = copy.copy(utility_to_allocate)

    if util_type == Util_Type.time_discounted:
        TrackTime("Adjusting utilities (time-discounted)")
        utility_to_allocate -= historical_context[DISCOUNTED_UTILITIES].reshape(-1,1)
        utility_to_allocate[utility_to_allocate<0] = 0

    TrackTime("Get all eligible members")
    # Determine for every offer, the set of eligible members
    offer_id_to_eligible_members = get_all_eligible_members(batch_to_send, members, supporting_info,
                                                            historical_context, util_type, sim_start_time)

    TrackTime("Filtering relevant utilies")
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
        assert len(col_indices_to_adjust) > 0, "%d !> 0"%len(col_indices_to_adjust)

        indices_to_adjust = np.ix_(non_eligible_members_indices, col_indices_to_adjust)
        utility_to_allocate[indices_to_adjust] = -1


    # Decrease nr rows of utility_values based on eligible members
    all_eligible_member_ids = list(all_eligible_member_ids)
    all_eligible_member_indices = list(map(lambda member_id: member_id_to_index[member_id], all_eligible_member_ids))
    utility_to_allocate = utility_to_allocate[all_eligible_member_indices, :]
    actual_utility = actual_utility[all_eligible_member_indices, :]


    coupon_index_to_id = list(map(lambda coupon_tuple: coupon_tuple[COUPON_ID_COLUMN], batch_to_send))
    member_index_to_id = all_eligible_member_ids

    assert utility_to_allocate.shape == actual_utility.shape
    assert utility_to_allocate.shape == (len(member_index_to_id), len(coupon_index_to_id)), "%s != %s"%(str(utility_to_allocate.shape), str((len(member_index_to_id), len(coupon_index_to_id))))
    return utility_to_allocate, actual_utility, (coupon_index_to_id, member_index_to_id)


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

def read_utility(members=None, offers=None):

    if members is not None and offers is not None:
        # Generate Utilities, the fit of a member to an offer
        nr_agents = len(members)
        nr_unique_resources = len(offers)
        utility_values = np.random.uniform(0,0.7,size=(nr_agents, nr_unique_resources))

        # Creates a dictionary from 'id' column to index of dataframe
        member_id_to_index = members['id'].reset_index(drop=True).reset_index().set_index('id').to_dict()['index']
        offer_id_to_index = offers['id'].reset_index(drop=True).reset_index().set_index('id').to_dict()['index']
        utility_indices = (member_id_to_index, offer_id_to_index)

    else:
        utility_df = pd.read_pickle(global_folder + "utility_df.pkl")
        member_indices = np.arange(utility_df.shape[0])
        member_id_to_index = {utility_df.index[member_index]: member_index for member_index in member_indices}

        offer_indices = np.arange(utility_df.shape[1])
        offer_id_to_index = {utility_df.columns[offer_index]: offer_index for offer_index in offer_indices}

        utility_values = utility_df.values
        utility_indices = (member_id_to_index, offer_id_to_index)

    return utility_values, utility_indices


def prepare_simulation_data(db=None):
    if db:
        result = query_db.retrieve_from_sql_db(db, 'filtered_coupons', 'filtered_issues', 'filtered_offers', 'member')
        filtered_coupons, filtered_issues, filtered_offers, all_members = result

        query = "select * from member_category"
        all_member_categories = pd.read_sql_query(query, db)
        query = "select * from member_family_member where type='child'"
        all_children = pd.read_sql_query(query, db)
        query = "select * from member_family_member where type='partner'"
        all_partners = pd.read_sql_query(query, db)

    else:
        with open('./database/data/filtered_coupons.pkl', 'rb') as fp:      filtered_coupons        = pickle.load(fp)
        with open('./database/data/filtered_issues.pkl', 'rb') as fp:       filtered_issues         = pickle.load(fp)
        with open('./database/data/filtered_offers.pkl', 'rb') as fp:       filtered_offers         = pickle.load(fp)
        with open('./database/data/all_members.pkl', 'rb') as fp:           all_members             = pickle.load(fp)
        with open('./database/data/all_member_categories.pkl', 'rb') as fp: all_member_categories   = pickle.load(fp)
        with open('./database/data/all_children.pkl', 'rb') as fp:          all_children            = pickle.load(fp)
        with open('./database/data/all_partners.pkl', 'rb') as fp:          all_partners            = pickle.load(fp)

    # No functionality to incorporate 'aborted_at' has been made (so far)
    assert np.all(filtered_issues['aborted_at'].isna())
    filtered_issues = filtered_issues.sort_values(by='sent_at')

    # No functionality to incorporate 'reissue_unused_participants_of_coupons_enabled' has been made (so far)
    # print("%d out of %d offers enabled reissued"%(np.sum(filtered_offers['reissue_unused_participants_of_coupons_enabled']), len(filtered_offers)))

    relevant_offer_columns = ['id', 'category_id', 'accept_time', 'member_criteria_gender', 'member_criteria_min_age',
                              'member_criteria_max_age', 'family_criteria_min_count', 'family_criteria_max_count',
                              'family_criteria_child_age_range_min', 'family_criteria_child_age_range_max',
                              'family_criteria_child_stages_child_stages', 'family_criteria_child_gender',
                              'family_criteria_is_single', 'family_criteria_has_children']
    filtered_offers = copy.copy(filtered_offers[relevant_offer_columns])

    # TODO: low-priority --> 'receive_coupon_after', 'deactivated_at', 'archived_at', 'onboarded_at', 'created_at'
    relevant_member_columns = ['id', 'active', 'member_state', 'email', 'mobile', 'date_of_birth', 'gender']
    all_members = all_members[relevant_member_columns]

    # email and phone number criteria
    all_members = all_members[~all_members['email'].isna()]
    all_members = all_members[~all_members['mobile'].isna()]

    # Put offer_id as index of the dataframe (instead of 0 until len(df))
    filtered_offers['id_index'] = filtered_offers['id']
    filtered_offers = filtered_offers.set_index('id_index')
    # Put issue_id as index of the dataframe (instead of 0 until len(df))
    filtered_issues['id_index'] = filtered_issues['id']
    filtered_issues = filtered_issues.set_index('id_index')

    # Force pd.Timestamp to be datetime.datetime objects (datetime is faster in non-vectorized instances, like this simulation)
    datetimes = filtered_issues['sent_at'].dt.to_pydatetime()
    filtered_issues['sent_at'] = pd.Series(datetimes, index=filtered_issues.index, dtype=object)


    supporting_info = (filtered_offers, all_member_categories, all_children, all_partners)

    # Calculate the global probability that a member lets a coupon expire, given that he does not accept the coupon
    member_scores = filtered_coupons[['member_id','member_response','id']].pivot_table(index=['member_id'], columns='member_response', aggfunc='count', fill_value=0)['id'].reset_index()
    member_scores.columns = [member_scores.columns[0]] + list(map(lambda event: "nr_" + "_".join(str(event).split('_')[1:]), member_scores.columns[1:]))
    member_scores['nr_sent'] = member_scores['nr_accepted'] + member_scores['nr_declined'] + member_scores['nr_let_expire']
    member_scores = member_scores[['member_id', 'nr_sent', 'nr_accepted',  'nr_declined', 'nr_let_expire']]
    P_let_expire_given_not_accepted = member_scores['nr_let_expire'].sum() / (member_scores['nr_declined'].sum() + member_scores['nr_let_expire'].sum())

    # Determine the distribution of decline times
    declined_coupons = filtered_coupons[(filtered_coupons['status'] == 'declined') & (filtered_coupons['sub_status'].isna())]
    decline_times = declined_coupons['status_updated_at'] - declined_coupons['created_at']
    decline_times_days = decline_times.astype('timedelta64[s]') / 60 / 60 / 24
    percent_decline_time = decline_times_days / declined_coupons['accept_time']
    valid_decline_times = percent_decline_time[percent_decline_time<=1]
    # print("Avg percentage of decline time needed: ", valid_decline_times.mean())
    # print("Median percentage of decline time needed: ", valid_decline_times.median())
    # plt.hist(valid_decline_times, bins=50)
    # plt.title("Distribution of needed decline time as fraction of total given time")
    # plt.show()
    distribution_info = (P_let_expire_given_not_accepted, valid_decline_times.values.reshape(-1))


    utility_values, utility_indices = get_utility(all_members['id'], filtered_offers['id'], filtered_coupons)

    util_prep = (utility_values, utility_indices)
    data_prep = (filtered_issues, all_members, supporting_info, distribution_info)
    return util_prep, data_prep


def get_utility(member_ids=None, offer_ids=None, filtered_coupons=None):
    # utility_df = pd.read_pickle(global_folder + "utility_df.pkl")
    if member_ids is None or offer_ids is None or filtered_coupons is None:
        utility_df = pd.read_pickle(global_folder + "utility_df.pkl")
    else:
        utility_df = make_utility(member_ids, offer_ids, filtered_coupons)


    member_indices = np.arange(utility_df.shape[0])
    member_id_to_index = {utility_df.index[member_index]: member_index for member_index in member_indices}

    offer_indices = np.arange(utility_df.shape[1])
    offer_id_to_index = {utility_df.columns[offer_index]: offer_index for offer_index in offer_indices}

    utility_values = utility_df.values
    utility_indices = (member_id_to_index, offer_id_to_index)
    return utility_values, utility_indices


def make_utility(member_ids, offer_ids, filtered_coupons):

    member_scores = filtered_coupons[['member_id','member_response','id']].pivot_table(index=['member_id'], columns='member_response', aggfunc='count', fill_value=0)['id'].reset_index()
    offer_scores  = filtered_coupons[['offer_id','member_response','id']].pivot_table(index=['offer_id'], columns='member_response', aggfunc='count', fill_value=0)['id'].reset_index()
    member_scores.columns = [member_scores.columns[0]] + list(map(lambda event: "nr_" + "_".join(str(event).split('_')[1:]), member_scores.columns[1:]))
    offer_scores.columns  = [offer_scores.columns[0]] + list(map(lambda event: "nr_" + "_".join(str(event).split('_')[1:]), offer_scores.columns[1:]))

    # Adjustments To make sure no probabilities are exactly 1 or 0
    member_scores['nr_accepted'] += 1
    member_scores['nr_declined'] += 1
    offer_scores['nr_accepted'] += 1
    offer_scores['nr_declined'] += 1

    # Calculate per member the probability of accepting
    P_accept_members = member_scores['nr_accepted'] / (member_scores['nr_accepted'] + member_scores['nr_declined'] + member_scores['nr_let_expire'])
    P_accept_members.index = member_scores['member_id']
    # If a member has never received any coupon, assume the average P_accept
    remaining_members = set(member_ids) - set(member_scores['member_id'])
    P_remaining_members = pd.DataFrame([P_accept_members.mean()] * len(remaining_members), index=list(remaining_members))
    # Put all probabilities together
    if not P_remaining_members.empty:
        P_accept_members = pd.concat([P_accept_members, P_remaining_members])
    assert len(set(P_accept_members.index)) == len(member_ids)

    # Calculate per offer the probability of accepting
    P_accept_offers = offer_scores['nr_accepted'] / (offer_scores['nr_accepted'] + offer_scores['nr_declined'] + offer_scores['nr_let_expire'])
    P_accept_offers.index = offer_scores['offer_id']
    # If an offer has never been sent to any member, assume the average P_accept
    remaining_offers = set(offer_ids) - set(offer_scores['offer_id'])
    P_remaining_offers = pd.DataFrame([P_accept_offers.mean()] * len(remaining_offers), index=list(remaining_offers))
    # Put all probabilities together
    if not P_remaining_offers.empty:
        P_accept_offers = pd.concat([P_accept_offers, P_remaining_offers])
    assert len(set(P_accept_offers.index)) == len(offer_ids)

    # Matrix-multiply all probabilities to generate the matrix
    utility_matrix = P_accept_members.values.reshape(-1,1) @ P_accept_offers.values.reshape(1,-1)
    utility_df = pd.DataFrame(utility_matrix, columns=P_accept_offers.index, index=P_accept_members.index)

    # # To be able to make adjustments to utility based on whether the member has accepted or declined a specific offer
    # zero_matrix = pd.DataFrame(np.zeros(utility_df.values.shape, dtype=int), columns=utility_df.columns, index=utility_df.index)

    accepted_coupons = filtered_coupons[filtered_coupons['member_response'] == Event.member_accepted]
    # offer_per_member = accepted_coupons[['member_id','offer_id','id']].pivot_table(index=['member_id'], columns='offer_id', aggfunc='count', fill_value=0)['id']#.reset_index()
    # offer_per_member = (zero_matrix + offer_per_member).fillna(0)
    # member_accepted_offer = offer_per_member>0

    declined_coupons = filtered_coupons[filtered_coupons['member_response'] == Event.member_declined]
    # offer_per_member = declined_coupons[['member_id','offer_id','id']].pivot_table(index=['member_id'], columns='offer_id', aggfunc='count', fill_value=0)['id']#.reset_index()
    # offer_per_member = (zero_matrix + offer_per_member).fillna(0)
    # member_declined_offer = offer_per_member>0

    let_expire_coupons = filtered_coupons[filtered_coupons['member_response'] == Event.member_let_expire]
    avg_P_accept = len(accepted_coupons) / (len(accepted_coupons) + len(declined_coupons) + len(let_expire_coupons))

    # # Make adjustments to utility based on whether the member has accepted or declined a specific offer
    # utility_df = utility_df + 0.1*member_accepted_offer
    # utility_df[utility_df>=1] -= 0.1
    # utility_df = utility_df - 0.1*member_declined_offer
    # utility_df[utility_df<=0] += 0.1
    assert np.all(utility_df.values < 1) and np.all(utility_df.values > 0)

    ratio = np.log(avg_P_accept) / np.log(np.average(utility_df.values))
    utility_df = np.power(utility_df, ratio)

    plt.figure()
    plt.hist(utility_df.values.reshape(-1), bins=30, density=False)
    plt.xlabel("P_accept")
    plt.ylabel("Frequency")
    plt.title("Distribution of P_accept")
    plt.show()
    # print(utility_df.shape)

    # utility_values, (member_id_to_index, offer_id_to_index) = get_utility()
    # utility_df = pd.DataFrame(utility_values, columns=offer_id_to_index, index=member_id_to_index)
    # print(utility_df.shape)
    # print(utility_df)
    # plt.figure()
    # plt.hist(utility_df.values.reshape(-1), bins=30, density=True)


    return utility_df



if __name__ == '__main__':
    main()