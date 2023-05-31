# -*- coding: utf-8 -*-
"""
Created on Mon May 29 20:12:40 2023

@author: Marco
"""
import os
import re
import sys
import copy
import enum
import json
import traceback
import numpy as np
import pandas as pd
import datetime as dt
from pprint import pprint
from collections import Counter
import matplotlib.pyplot as plt
from database.lib.tracktime import TrackTime, TrackReport

import database.connect_db as connect_db
import database.query_db as query_db
Event = query_db.Event

pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 90)

def main():

    # baseline = pd.read_csv('./timelines/baseline_events.csv')
    # baseline.to_pickle('./timelines/baseline_events.pkl')
    export_folder = './timelines/'
    versions_to_read = [5,6,7]

    TrackTime("Connect to db")
    conn = connect_db.establish_host_connection()
    db   = connect_db.establish_database_connection(conn)
    print("Successfully connected to database '%s'"%str(db.engine).split("/")[-1][:-1])

    TrackTime("Retrieve from db")
    result = query_db.retrieve_from_sql_db(db, 'filtered_issues', 'member')
    filtered_issues, members = result
    print("")

    # Read simulated events and info
    sim_data = {}
    TrackTime("Read pickle")
    for version_to_read in versions_to_read:
        sim_events, info = read_input_pickles(version_to_read, export_folder)
        sim_events['event'] = sim_events['event'].apply(lambda event: Event[str(event)[len("Event."):]])
        sim_events = sim_events.convert_dtypes()
        sim_data[info['allocator_algorithm'] + "_" + str(info['batch_size'])] = sim_events

    # Read utility
    utility_df = pd.read_pickle(export_folder + "utility_df.pkl")

    # Read baseline events
    base_events = pd.read_pickle(export_folder + 'baseline_events.pkl')
    base_events['event'] = base_events['event'].apply(lambda event: Event[str(event)[len("Event."):]])
    base_events = base_events.convert_dtypes()

    TrackTime("Evaluate timeline")
    perform_sense_check(base_events, filtered_issues)
    for result_name, sim_events in sim_data.items():
        perform_sense_check(sim_events, filtered_issues)

    evaluate_timeline(sim_data, base_events, utility_df, members, filtered_issues['id'])

    db.close()
    TrackReport()


def read_input_pickles(version, export_folder):
    contents = os.listdir(export_folder)
    contents = list(filter(lambda name: re.search("^%d_.*\.pkl"%version, name), contents))
    events_df  = pd.read_pickle(export_folder + contents[0])
    # utility_df = pd.read_pickle(export_folder + (contents[0] if "utility_df" in contents[0] else contents[1]))

    f = open(export_folder + contents[0].replace("events_df",'info').replace('.pkl','.json'))
    info = json.load(f)
    f.close()

    return events_df, info


def evaluate_timeline(sim_data, base_events, utility_df, members, issue_ids):
    # print(events_df)
    member_ids = members['id']

    base_events = base_events[~base_events['member_id'].isna()]
    base_scores = calculate_member_scores(base_events, utility_df, member_ids)
    assert len(set(base_scores['member_id'].values)) == len(base_scores)
    base_members = set(base_scores['member_id'][base_scores['utility'] > 0].values)

    sim_utilities = {'baseline': base_scores}

    for result_name, sim_events in sim_data.items():
        print("")
        sim_events = sim_events[~sim_events['member_id'].isna()]
        sim_scores = calculate_member_scores(sim_events, utility_df, member_ids)
        assert len(set(sim_scores['member_id'].values)) == len(sim_scores)

        sim_members = set(sim_scores['member_id'][sim_scores['utility'] > 0].values)
        print("nr members received coupon in simulation:", len(sim_members))
        print("nr members received coupon in baseline:", len(base_members))
        print("\nnr members received coupon in baseline not in simulation:", len(base_members - sim_members))
        print("nr members received coupon in simulation not in baseline:", len(sim_members - base_members))

        inactive_members = set(members['id'][members['member_state'] != 'active'].values)
        print('inactive baseline:', len(base_members.intersection(inactive_members)))
        print('inactive simulation:', len(sim_members.intersection(inactive_members)))

        sim_utilities[result_name] = sim_scores


    for result_name, scores in sim_utilities.items():
        summarize_utilities(scores, result_name)


    TrackTime("plots")
    plot_utilities(sim_utilities)


def summarize_utilities(scores, result_name):
    utilities = np.array(scores['utility'].astype(float).values)
    perc_utilities = utilities/np.sum(utilities)

    print("\n"+result_name)
    print("\t\t\t\tTotal Utility \t= %.5f"%(np.sum(utilities)))
    print("\t\t\t  Average Utility \t= %.5f"%(np.average(utilities)))
    print("\t Average non-zero Utility \t= %.5f"%(np.average(utilities[utilities>0])))
    print("   Maximum (non-zero) Utility \t= %.5f"%np.max(utilities))
    print("\t Minimum non-zero Utility \t= %.5f"%(np.min(utilities[utilities>0])))
    print("\t  Median non-zero Utility \t= %.5f"%np.sort(utilities[utilities>0])[int(0.5*np.sum(utilities>0))])


def plot_utilities(sim_utilities):
    fig_names = ['lorenz', 'sorted_utilities', 'nonzero_utilities_histogram']
    
    
    # fig_names = set()
    for i, (result_name, scores) in enumerate(sim_utilities.items()):
        utilities = np.array(scores['utility'].astype(float).values)

        # fig_names.update('lorenz')
        plt.figure('lorenz')
        plt.plot(np.cumsum(np.sort(utilities/np.sum(utilities))), label=result_name)

        if i == 0:
            equality = np.arange(len(utilities))/len(utilities)
            plt.plot(equality, 'k--', alpha=0.5, label='equality')

        # fig_names.update('sorted_utilities')
        plt.figure('sorted_utilities')
        plt.plot(np.sort(utilities), label=result_name)

        # fig_names.update('utilities_histogram')
        # plt.figure('utilities_histogram')
        # plt.hist(sim_utilities, bins=30, alpha=0.5, color='red', label='simulation')
        # plt.hist(base_utilities, bins=30, alpha=0.5, color='blue', label='baseline')


    # fig_names.update('nonzero_utilities_histogram')
    plt.figure('nonzero_utilities_histogram')
    colors = ['red', 'blue', 'yellow', 'green']
    for i, (result_name, scores) in enumerate(sim_utilities.items()):
        utilities = np.array(scores['utility'].astype(float).values)

        plt.hist(utilities[utilities>0], bins=30, alpha=0.5, color=colors[i], label=result_name)


    for fig_name in fig_names:
        plt.figure(fig_name)
        plt.title(fig_name)
        plt.legend()



def calculate_member_scores(events_df, utility_df, member_ids):

    scores = events_df[['member_id','event','coupon_id']].pivot_table(index=['member_id'], columns='event', aggfunc='count', fill_value=0)['coupon_id'].reset_index()
    scores.columns = [scores.columns[0]] + list(map(lambda event: "nr_" + "_".join(str(event).split('_')[1:]), scores.columns[1:]))
    scores = scores[['member_id', 'nr_sent', 'nr_accepted',  'nr_declined', 'nr_let_expire']]
    assert np.all(scores['nr_sent'] == scores['nr_accepted'] + scores['nr_declined'] + scores['nr_let_expire'])

    accepted_offers = events_df[events_df['event'] == Event.member_accepted][['member_id','offer_id']]
    accepted_offers = accepted_offers.set_index('offer_id')
    accepted_offers_per_member = accepted_offers.groupby('member_id').groups

    total_utility_per_member = {member_id: calc_utility(utility_df, member_id, accepted_offers) for member_id, accepted_offers in accepted_offers_per_member.items()}
    total_utility_per_member = pd.DataFrame.from_dict(total_utility_per_member, orient='index', columns=['utility']).reset_index()
    total_utility_per_member.columns = ['member_id'] + list(total_utility_per_member.columns[1:])

    scores = scores.merge(total_utility_per_member)

    member_ids = pd.DataFrame(member_ids.values, columns=['member_id'])
    scores = pd.merge(member_ids, scores, 'left').fillna(0)

    return scores.convert_dtypes()


def calc_utility(utility_df, member_id, accepted_offers):
    utility_scores = list(map(lambda offer_id: utility_df.loc[member_id, offer_id], accepted_offers))
    return np.sum(utility_scores)


def perform_sense_check(events_df, issues):
    relevant_issue_columns_check   = ['id', 'offer_id', 'sent_at', 'amount', 'expires_at']
    relevant_issue_columns_compare = ['id', 'total_issued', 'nr_accepted', 'nr_expired', 'nr_not_sent_on']
    issues = issues[relevant_issue_columns_check + relevant_issue_columns_compare[1:]]
    issues = issues.rename(columns={'id':'issue_id'}).sort_values('issue_id').reset_index(drop=True)

    accepted_events = events_df[events_df['event'] == Event.member_accepted]
    nr_accepted = accepted_events.groupby('issue_id').aggregate(nr_accepted=('coupon_id','count')).reset_index()

    declined_events = events_df[events_df['event'] == Event.member_declined]
    nr_declined = declined_events.groupby('issue_id').aggregate(nr_declined=('coupon_id','count')).reset_index()

    let_expire_events = events_df[events_df['event'] == Event.member_let_expire]
    nr_let_expire = let_expire_events.groupby('issue_id').aggregate(nr_let_expire=('coupon_id','count')).reset_index()

    available_events = events_df[events_df['event'] == Event.coupon_available]
    first_follow_available = available_events.groupby('coupon_follow_id').first().reset_index()
    nr_first_available = first_follow_available.groupby('issue_id').aggregate(nr_first_available=('coupon_id','count')).reset_index()

    # NB: follow.last() is not always expired or accepted, declined or let_expire can be after expiry

    list_of_tables = [nr_accepted, nr_declined, nr_let_expire, nr_first_available]
    simulation_results = join_tables(list_of_tables, ['issue_id'], issues[['issue_id']])
    simulation_results = simulation_results.sort_values(by='issue_id').reset_index(drop=True)
    # print(simulation_results)

    to_compare = simulation_results.merge(issues.add_suffix('_baseline').rename(columns={'issue_id_baseline':'issue_id'}), on='issue_id')

    assert np.all(to_compare['amount_baseline'] == to_compare['nr_first_available'])
    diff = (to_compare['nr_accepted'] - to_compare['nr_accepted_baseline']).abs()
    # print(np.sum(diff), np.sort(diff.fillna(0))[-10:])



def join_tables(list_of_tables, join_on, all_ids=None):
    if all_ids is None:
        result = list_of_tables[0][join_on]
    else:
        for i, col in enumerate(all_ids.columns):
            all_ids = all_ids.rename(columns={col:join_on[i]})
        result = all_ids

    for i, table in enumerate(list_of_tables):
        result = pd.merge(result, table, on=join_on, how='left')

    return result.convert_dtypes()


if __name__ == '__main__':
    main()
