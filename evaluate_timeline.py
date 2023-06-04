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
import pickle
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

global_folder = './timelines/'
run_folder = './timelines/run_%d/'

def main():

    # baseline = pd.read_csv('./timelines/baseline_events.csv')
    # baseline.to_pickle('./timelines/baseline_events.pkl')
    run_nrs_to_read = [4,5,6]

    TrackTime("Connect to db")
    conn = connect_db.establish_host_connection()
    db   = connect_db.establish_database_connection(conn)
    print("Successfully connected to database '%s'"%str(db.engine).split("/")[-1][:-1])

    TrackTime("Retrieve from db")
    result = query_db.retrieve_from_sql_db(db, 'filtered_issues', 'member')
    filtered_issues, members = result
    print("")

    # Read utility
    utility_df = pd.read_pickle(global_folder + "utility_df.pkl")

    # Read simulated events and info
    TrackTime("Get all run data")
    all_run_data = {}
    for run_nr in run_nrs_to_read:
        if not os.path.exists(run_folder%run_nr + '%d_run_data.pkl'%run_nr):
            run_name, run_data = extract_relevant_info(run_nr, utility_df, members['id'], filtered_issues)
            with open(run_folder%run_nr + '%d_run_name.pkl'%run_nr, 'wb') as fp:
                pickle.dump(run_name, fp)
            with open(run_folder%run_nr + '%d_run_data.pkl'%run_nr, 'wb') as fp:
                pickle.dump(run_data, fp)
        else:
            TrackTime("Open pickle")
            with open(run_folder%run_nr + '%d_run_name.pkl'%run_nr, 'rb') as fp:
                run_name = pickle.load(fp)
            with open(run_folder%run_nr + '%d_run_data.pkl'%run_nr, 'rb') as fp:
                run_data = pickle.load(fp)

        all_run_data[run_name] = run_data

    # Read baseline events
    base_events = pd.read_pickle(global_folder + 'baseline_events.pkl')
    base_events['event'] = base_events['event'].apply(lambda event: Event[str(event)[len("Event."):]])
    base_events = base_events.convert_dtypes()

    base_data = extract_timeline_info(base_events, utility_df, members['id'])
    perform_sense_check(base_events, filtered_issues)

    # Evaluate and plot
    TrackTime("Evaluate")
    evaluate_timelines(all_run_data, base_data, utility_df, members, filtered_issues['id'])
    TrackTime("Plots")
    plot_utilities(all_run_data, base_data)

    db.close()
    TrackReport()


def extract_relevant_info(run_nr, utility_df, member_ids, filtered_issues):
    contents = os.listdir(run_folder%run_nr)
    timeline_files = list(filter(lambda name: re.search("^%d.[0-9]+_events_df.pkl"%run_nr, name), contents))
    timeline_nrs = list(map(lambda name: int(name.split('_')[0].split('.')[-1]), timeline_files))

    member_utils_all_sims = None
    for sim_nr in timeline_nrs:
        TrackTime("Read pickle")
        sim_events = pd.read_pickle(run_folder%run_nr + "%d.%d_events_df.pkl"%(run_nr, sim_nr))
        sim_events['event'] = sim_events['event'].apply(lambda event: Event[str(event)[len("Event."):]])
        sim_events = sim_events.convert_dtypes()

        TrackTime("Perform sense check")
        perform_sense_check(sim_events, filtered_issues)

        TrackTime("Extract timeline info")
        sim_info = extract_timeline_info(sim_events, utility_df, member_ids)
        member_utils, = sim_info

        TrackTime("Get all run data")
        if member_utils_all_sims is None:
            member_utils_all_sims = member_utils.rename(columns={'utility':'sim_%d'%sim_nr})
        else:
            assert np.all(member_utils.index == member_utils_all_sims.index)
            member_utils_all_sims['sim_%d'%sim_nr] = member_utils['utility']

    TrackTime("Read info")
    f = open(run_folder%run_nr + "%d_info.json"%run_nr)
    info = json.load(f)
    f.close()

    run_name = info['allocator_algorithm'] + "_" + str(info['batch_size'])
    return run_name, (member_utils_all_sims, )


def extract_timeline_info(sim_events, utility_df, member_ids):

    sim_events = sim_events[~sim_events['member_id'].isna()]
    sim_scores = calculate_member_scores(sim_events, utility_df, member_ids)
    assert len(set(sim_scores['member_id'].values)) == len(sim_scores)

    member_utils = sim_scores[['utility']]

    return member_utils,


def evaluate_timelines(all_run_data, base_data, utility_df, members, issue_ids):
    # print(events_df)
    member_ids = members['id']

    base_member_utils, = base_data

    # assert len(set(base_scores['member_id'].values)) == len(base_scores)
    # base_members = set(base_scores['member_id'][base_scores['utility'] > 0].values)

    # sim_utilities = {'baseline': base_scores}

    # for run_name, run_data in all_run_data.items():
    #     print("")

    #     sim_members = set(sim_scores['member_id'][sim_scores['utility'] > 0].values)
    #     print("nr members received coupon in simulation:", len(sim_members))
    #     print("nr members received coupon in baseline:", len(base_members))
    #     print("\nnr members received coupon in baseline not in simulation:", len(base_members - sim_members))
    #     print("nr members received coupon in simulation not in baseline:", len(sim_members - base_members))

    #     inactive_members = set(members['id'][members['member_state'] != 'active'].values)
    #     print('inactive baseline:', len(base_members.intersection(inactive_members)))
    #     print('inactive simulation:', len(sim_members.intersection(inactive_members)))

    #     sim_utilities[result_name] = sim_scores


    # for result_name, scores in sim_utilities.items():
    #     summarize_utilities(scores, result_name)
    return



def summarize_utility_distribution(member_utils_all_sims):

    all_sim_summaries = {}
    for col_name in member_utils_all_sims.columns:
        col = member_utils_all_sims[col_name]
        sim_summary = {}
        sim_summary['sum'] = col.sum()
        sim_summary['average'] = col.mean()
        sim_summary['avg_nonzero'] = col[col>0].mean()
        sim_summary['max'] = col.max()
        sim_summary['min_nonzero'] = col[col>0].min()
        median_index = int(0.5*len(col))
        sim_summary['median'] = col.sort_values().values[median_index]
        median_nonzero_index = int(0.5*np.sum(col>0))
        sim_summary['median_nonzero'] = col[col>0].sort_values().values[median_nonzero_index]

        all_sim_summaries[col_name] = sim_summary

    all_sim_summaries = pd.DataFrame.from_dict(all_sim_summaries, orient='index')
    # print(all_sim_summaries)
    return all_sim_summaries


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


def plot_utilities(all_run_data, base_data):
    fig_names = ['avg_lorenz', 'sorted_utilities', 'nonzero_utilities_histogram']#, 'summary']

    base_member_utils, = base_data
    base_summary = summarize_utility_distribution(base_member_utils)
    colors = ['red', 'blue', 'yellow']
    base_col = 'green'

    for i, (run_name, run_data) in enumerate(all_run_data.items()):
        member_utils_all_sims, = run_data
        summaries_all_sims = summarize_utility_distribution(member_utils_all_sims)


        ####### Lorenz Curve Plot #############################
        plt.figure('avg_lorenz')
        if i == 0:
            equality = np.arange(len(member_utils_all_sims))/len(member_utils_all_sims)
            plt.plot(equality, 'k--', alpha=0.5, label='equality')

            baseline = base_member_utils.values.reshape(-1)
            plt.plot(np.cumsum(np.sort(baseline)/np.sum(baseline)), label='baseline', color=base_col)

        sorted_utils = np.sort(member_utils_all_sims.values, axis=0)
        avg_utils = np.average(sorted_utils, axis=1)

        plt.plot(np.cumsum(np.sort(avg_utils)/np.sum(avg_utils)), label=run_name, color=colors[i])


        ####### Sorted Utilities Plot #############################
        plt.figure('sorted_utilities')
        if i == 0:
            plt.plot(np.sort(base_member_utils.values.reshape(-1)), label='baseline', color=base_col)
        plt.plot(np.sort(avg_utils), label=run_name, color=colors[i])


        ####### Summary Subplots #######################################
        plt.figure('summary', figsize=(10,15))
        subplot_data_cols = ['average', 'avg_nonzero', 'median', 'median_nonzero', 'max', 'min_nonzero']
        for j, col_name in enumerate(subplot_data_cols, 1):
            plt.subplot(3,2,j)
            data = summaries_all_sims[col_name].values
            plt.hist(data, bins=12, alpha=0.5, color=colors[i], label=run_name, density=True)

            if i == len(all_run_data)-1:
                # x_value = base_summary[col_name]
                # xmin, xmax, ymin, ymax = plt.axis()
                # plt.plot([x_value, x_value], [ymin, ymax], '-.', color=base_col, linewidth=5, label='baseline')
                # plt.ylim([ymin, ymax])
                plt.legend()
                plt.title(col_name)


        ####### Utilities histogram #######################################
        plt.figure('nonzero_utilities_histogram')
        if i == 0:
            plt.hist(base_member_utils.values[base_member_utils.values>0], bins=30, alpha=0.5, color=base_col, label='baseline', density=True)

        member_utils_all_sims, = run_data
        flat_utils = member_utils_all_sims.values.reshape(-1)
        plt.hist(flat_utils[flat_utils>0], bins=30, alpha=0.5, color=colors[i], label=run_name, density=True)


    # Set the title and legend
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
