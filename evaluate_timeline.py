# -*- coding: utf-8 -*-
"""
Created on Mon May 29 20:12:40 2023

@author: Marco
"""
import os
import re
import sys
import copy
import json
import pickle
import numpy as np
import pandas as pd
import datetime as dt
from pprint import pprint
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from stat import  S_IREAD, S_IRGRP, S_IROTH, S_IWUSR # Need to add this import to the ones above
from dateutil.relativedelta import relativedelta
from database.lib.tracktime import TrackTime, TrackReport
import scipy.stats


import database.connect_db as connect_db
import database.query_db as query_db
from database.query_db import Event

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 90)

global_folder = './timelines/'
run_folder = './timelines/run_%d/'

# import warnings
# warnings.filterwarnings("ignore")

SHOW_BASELINE = True
RECOMPUTE = False
FONTSIZE = 20
plt.style.use('seaborn')

colors = ['black', 'yellow', 'red'] + ['green'] * 20
base_col = 'deepskyblue'

ALL_SIMS_BOXPLOT = 'false'
def main():
    global ALL_SIMS_BOXPLOT

    # BATCH SIZE COMPARISON
    # run_nrs_to_read = [63,64,65]     # greedy_x_TD    # biggest differences
    # run_nrs_to_read = [69,66,67]     # greedy_x_FH    # nothing really interesting
    # run_nrs_to_read = [71,68,70]     # greedy_x_PH    # small diff, only in nr_sent boxplot+quantiles
    # run_nrs_to_read = [63,91,90]     # max_sum_x_TD   # id, nothing really interesting
    # run_nrs_to_read = [69,86,88]     # max_sum_x_FH   # Some results show slightly more fair
    # run_nrs_to_read = [71,87,89]     # max_sum_x_PH   # slightly more fair batch 200 tov batch 50
    # ALL_SIMS_BOXPLOT = 'batch_size'
    # run_nrs_to_read = [63,64,65,69,66,67,71,68,70,63,91,90,93,86,88,71,87,89]

    # UTIL_TYPE COMPARISON
    # run_nrs_to_read = [23,69,71]  # greedy_1_x     # Lorenz curve MUST be discussed
    # run_nrs_to_read = [64,66,68]  # greedy_50_x    # PH is very unfair, FH + TD very close
    # run_nrs_to_read = [65,67,70]  # greedy_200_x   # Same story, FH probably bit better
    # run_nrs_to_read = [91,86,87]  # max_sum_50_x   # Hard to say
    # run_nrs_to_read = [90,88,89]  # max_sum_200_x  # Decent differences among all 3
    # ALL_SIMS_BOXPLOT = 'historical_context'
    # run_nrs_to_read = [63,69,71,64,66,68,65,67,70,91,86,87,90,88,89]

    # GREEDY vs MAX_SUM COMPARISON
    # run_nrs_to_read = [69,93]     # x_1_FH         # Exactly the same, as expected
    # run_nrs_to_read = [64,91]     # x_50_TD        # Enigzins verschil
    # run_nrs_to_read = [65,90]     # x_200_TD       # Heeel klein beetje verschil
    # run_nrs_to_read = [66,86]     # x_50_FH        # Barely a difference
    # run_nrs_to_read = [67,88]     # x_200_FH       # No difference
    # run_nrs_to_read = [68,87]     # x_50_PH        # Geen verschil
    # run_nrs_to_read = [70,89]     # x_200_PH       # Minuscuul verschil
    # ALL_SIMS_BOXPLOT = 'objective'
    # run_nrs_to_read = [69,93,64,91,65,90,66,86,67,88,68,87,70,89]

    run_nrs_to_read = [66,64]#63,69,64,66]


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
    all_run_data, all_run_info = {}, {}

    # Loop over all run_nrs interested in
    for run_nr in run_nrs_to_read:
        # Recompute data
        if RECOMPUTE or not os.path.exists(run_folder%run_nr + '%d_run_data.pkl'%run_nr):
            # utility_df = pd.read_pickle(run_folder%run_nr + "%d_utility_df.pkl"%run_nr)
            run_name, run_data = extract_relevant_info(run_nr, utility_df, members['id'], filtered_issues)

            # Dump the calculated things to pickle
            with open(run_folder%run_nr + '%d_run_name.pkl'%run_nr, 'wb') as fp:
                pickle.dump(run_name, fp)
            with open(run_folder%run_nr + '%d_run_data.pkl'%run_nr, 'wb') as fp:
                pickle.dump(run_data, fp)

        else:
            # Read the dumped pickle files
            TrackTime("Open pickle")
            with open(run_folder%run_nr + '%d_run_name.pkl'%run_nr, 'rb') as fp:
                run_name = pickle.load(fp)
            with open(run_folder%run_nr + '%d_run_data.pkl'%run_nr, 'rb') as fp:
                run_data = pickle.load(fp)

        # Parse json into a dictionary
        f = open(run_folder%run_nr + "%d_info.json"%run_nr)
        run_info = json.load(f)
        f.close()

        # Save run data and info
        all_run_data[run_name] = run_data
        all_run_info[run_name] = run_info

    if RECOMPUTE:
        print("\nSet 'RECOMPUTE' variable to False to make plots")
        sys.exit()

    # Read baseline events
    base_events = pd.read_pickle(global_folder + 'baseline_events.pkl')
    base_events['event'] = base_events['event'].apply(lambda event: Event[str(event)[len("Event."):]])
    base_events = base_events.convert_dtypes()
    # Extract baseline information
    perform_timeline_sense_check(base_events, filtered_issues)
    base_data = extract_timeline_info(base_events, utility_df, members['id'])

    # Plot and make dataframe
    TrackTime("Member plots")
    plot_member_stats(all_run_data, base_data)
    # TrackTime("Monthly plots")
    # plot_monthly_stats(all_run_data, base_data)
    # TrackTime("Batch plots")
    # plot_batch_stats(all_run_data, base_data, all_run_info)

    db.close()
    # TrackReport()


def extract_relevant_info(run_nr, utility_df, member_ids, filtered_issues):
    contents = os.listdir(run_folder%run_nr)
    timeline_files = list(filter(lambda name: re.search("^%d.[0-9]+_events_df.pkl"%run_nr, name), contents))
    timeline_nrs = sorted(list(map(lambda name: int(name.split('_')[0].split('.')[-1]), timeline_files)))

    # Necessary for defining the run_name
    TrackTime("Read info")
    f = open(run_folder%run_nr + "%d_info.json"%run_nr)
    run_info = json.load(f)
    f.close()

    # Variables to store all simulations data in
    member_stats_all_sims = None
    monthly_stats_all_sims = None
    batch_stats_all_sims = None

    for sim_nr in timeline_nrs:
        print("\rReading data from run %d, simulation %d"%(run_nr, sim_nr), end='\t\t')
        TrackTime("Read pickle")
        sim_events = pd.read_pickle(run_folder%run_nr + "%d.%d_events_df.pkl"%(run_nr, sim_nr))
        sim_events['event'] = sim_events['event'].apply(lambda event: Event[str(event)[len("Event."):]])
        sim_events = sim_events.convert_dtypes()

        TrackTime("Perform sense check")
        perform_timeline_sense_check(sim_events, filtered_issues)

        TrackTime("Extract timeline info")
        sim_info = extract_timeline_info(sim_events, utility_df, member_ids, run_info['batch_size'])
        member_stats, monthly_stats, batch_stats = sim_info

        # Add simulation monthly info to monthly info across all sims
        TrackTime("Combine monthly info")
        flat_values = monthly_stats.values.flatten('F')
        columns = pd.MultiIndex.from_product([monthly_stats.columns, list(monthly_stats.index)], names=['event', 'month'])
        flat_monthly_stats = pd.DataFrame([flat_values], columns=columns)

        if monthly_stats_all_sims is None:
            monthly_stats_all_sims = flat_monthly_stats
        else:
            assert len(monthly_stats) == len(monthly_stats_all_sims[monthly_stats.columns[0]].columns)
            assert np.all(monthly_stats_all_sims.columns == flat_monthly_stats.columns)
            monthly_stats_all_sims = pd.concat([monthly_stats_all_sims, flat_monthly_stats], ignore_index=True)

        # Add simulation member info to member info across all sims
        TrackTime("Combine member info")
        if member_stats_all_sims is None:
            member_stats = member_stats.set_index('member_id')
            member_stats.columns = pd.MultiIndex.from_product([member_stats.columns, ['sim_0']])
            member_stats_all_sims = member_stats
        else:
            member_stats = member_stats.set_index('member_id')
            assert len(member_stats) == len(member_stats_all_sims)
            assert set(member_stats.columns) == set(member_stats_all_sims.columns.levels[0])
            for col in member_stats.columns:
                member_stats_all_sims[col, 'sim_%d'%sim_nr] = member_stats[col]

        # Add simulation batch info to batch info across all sims
        TrackTime("Combine batch info")
        batch_stats['sim_nr'] = sim_nr
        if batch_stats_all_sims is None:
            batch_stats_all_sims = batch_stats
        else:
            batch_stats_all_sims = pd.concat([batch_stats_all_sims, batch_stats], ignore_index=True)

    # Make sure the multicolumns with same name are next to each other for nice displaying
    member_stats_all_sims = member_stats_all_sims.sort_index(axis=1)

    # Define the run name
    run_name = run_info['allocator_algorithm'] + "_" + str(run_info['batch_size'])
    run_name += '_' + "".join(list(map(lambda word: word.upper()[0], run_info['utility_type'].split('_'))))
    if 'version_tag' in run_info.keys() and run_info['version_tag'].strip() != "":
        run_name += "_" + run_info['version_tag']
    return run_name, (member_stats_all_sims, monthly_stats_all_sims, batch_stats_all_sims)


def extract_timeline_info(events, utility_df, member_ids, batch_size=None):
    # Calculate information about members
    TrackTime("calculate_member_scores")
    member_events = events[~events['member_id'].isna()]
    member_stats = calculate_member_scores(member_events, utility_df, member_ids)
    assert len(set(member_stats['member_id'].values)) == len(member_stats)

    # Compute monthly statistics
    TrackTime("make_monthly_summary")
    monthly_stats = make_monthly_summary(events)

    # Calculate statistics about batches, and offers
    TrackTime("other stats")
    batch_stats = get_batch_stats(events, batch_size)
    offer_stats = make_offer_stats(events, utility_df, member_ids)
    offer_stats

    return member_stats, monthly_stats, batch_stats

def get_batch_stats(events, batch_size):
    # Batch stats only relevant to Coupon_sent Events
    sent_events = events[events['event'] == Event.coupon_sent]

    # Compute for every batch_id when it was sent, and how many coupons were sent in the batch
    if 'batch_id' in events.columns:
        batch_info = sent_events.groupby('batch_id').aggregate(batch_sent_at=('at','first'), batch_size=('coupon_id','count')).reset_index(drop=True)
    else:
        batch_info = sent_events.groupby('at').aggregate(batch_size=('coupon_id','count')).reset_index().rename(columns={'at':'batch_sent_at'})

    # Also compute the time duration between batches
    batch_info['duration_since_prev_batch'] = batch_info['batch_sent_at'] - batch_info['batch_sent_at'].shift(1)
    return batch_info

def make_offer_stats(events, utility_df, member_ids):
    # to do
    return

def last_day_of_month(date):
    if date.month == 12:
        return date.replace(day=31)
    return date.replace(month=date.month+1, day=1) - dt.timedelta(days=1)

def datetime_range(start_date, end_date, delta):
    result = []
    nxt = start_date
    delta = relativedelta(**delta)

    while nxt <= end_date:
        result.append(nxt)
        nxt += delta

    result.append(end_date)
    return result

def make_monthly_summary(events):
    start_date = dt.datetime.combine(events['at'].iloc[0].replace(day=1).date(), dt.time(0,0,0))
    # end_date   = dt.datetime.combine(last_day_of_month(events['at'].iloc[-1]).date(), dt.time(23,59,59))
    end_date = dt.datetime.strptime("2022-12-31 23:59:59", "%Y-%m-%d %H:%M:%S")

    delta = {'months':1}
    intervals = datetime_range(start_date, end_date, delta)

    events_grouped_by_month = events.groupby(pd.cut(events['at'], intervals))
    all_months_summaries = {}
    for month, monthly_events in events_grouped_by_month:
        monthly_summary = {}

        nr_events = monthly_events[['event','at']].groupby('event').count().rename(columns={'at':'nr_events'})
        for event in Event:
            event_name = "nr_" + "_".join(str(event).split('_')[1:])
            if event in nr_events.index:
                monthly_summary[event_name] = nr_events.loc[event, 'nr_events']
            else:
                monthly_summary[event_name] = 0

        all_months_summaries[month] = monthly_summary

    all_months_summaries = pd.DataFrame.from_dict(all_months_summaries, orient='index')
    return all_months_summaries


def calculate_member_scores(events_df, utility_df, member_ids):
    scores = events_df[['member_id','event','coupon_id']].pivot_table(index=['member_id'], columns='event', aggfunc='count', fill_value=0)['coupon_id'].reset_index()
    scores.columns = [scores.columns[0]] + list(map(lambda event: "nr_" + "_".join(str(event).split('_')[1:]), scores.columns[1:]))
    scores = scores[['member_id', 'nr_sent', 'nr_accepted',  'nr_declined', 'nr_let_expire']]
    assert np.all(scores['nr_sent'] == scores['nr_accepted'] + scores['nr_declined'] + scores['nr_let_expire'])

    # Get a list of coupon-utilities per member
    summary_tables = []
    for event, name in zip([Event.coupon_sent, Event.member_accepted], ['sent','accepted']):
        sent_offers = events_df[events_df['event'] == event][['member_id','offer_id']]
        sent_offers = sent_offers.set_index('offer_id')
        sent_offers_per_member = sent_offers.groupby('member_id').groups
        received_utility_per_member = {member_id: calc_utility(utility_df, member_id, received_offers) for member_id, received_offers in sent_offers_per_member.items()}

        # Summarize the list of coupon-utilities into one measure (per member)
        total_worth_sent_coupon  = {member_id: np.sum(list_of_received_utilities) for member_id, list_of_received_utilities in received_utility_per_member.items()}
        avg_worth_sent_coupon    = {member_id: np.average(list_of_received_utilities) for member_id, list_of_received_utilities in received_utility_per_member.items()}
        median_worth_sent_coupon = {member_id: np.median(list_of_received_utilities) for member_id, list_of_received_utilities in received_utility_per_member.items()}
        min_worth_sent_coupon    = {member_id: np.min(list_of_received_utilities) for member_id, list_of_received_utilities in received_utility_per_member.items()}
        max_worth_sent_coupon    = {member_id: np.max(list_of_received_utilities) for member_id, list_of_received_utilities in received_utility_per_member.items()}

        total_worth_sent_coupon  = pd.DataFrame.from_dict(total_worth_sent_coupon, orient='index', columns=['tot_worth_%s_coupon'%name]).reset_index().rename(columns={'index':'member_id'})
        avg_worth_sent_coupon    = pd.DataFrame.from_dict(avg_worth_sent_coupon, orient='index', columns=['avg_worth_%s_coupon'%name]).reset_index().rename(columns={'index':'member_id'})
        median_worth_sent_coupon = pd.DataFrame.from_dict(median_worth_sent_coupon, orient='index', columns=['median_worth_%s_coupon'%name]).reset_index().rename(columns={'index':'member_id'})
        min_worth_sent_coupon    = pd.DataFrame.from_dict(min_worth_sent_coupon, orient='index', columns=['min_worth_%s_coupon'%name]).reset_index().rename(columns={'index':'member_id'})
        max_worth_sent_coupon    = pd.DataFrame.from_dict(max_worth_sent_coupon, orient='index', columns=['max_worth_%s_coupon'%name]).reset_index().rename(columns={'index':'member_id'})

        # Merge the tables into one
        tables_to_join = [total_worth_sent_coupon, avg_worth_sent_coupon, median_worth_sent_coupon, min_worth_sent_coupon, max_worth_sent_coupon]
        summary_worth_sent_coupons = join_tables(tables_to_join, ['member_id'])
        summary_tables.append(summary_worth_sent_coupons)

    # Merge all scores into one df
    summary_table = join_tables(summary_tables, ['member_id'])
    scores = scores.merge(summary_table, on='member_id', how='left').fillna(0)

    # Merge remaining member_ids and put to zero
    member_ids = pd.DataFrame(member_ids.values, columns=['member_id'])
    scores = pd.merge(member_ids, scores, 'left').fillna(0)

    return scores.convert_dtypes()


def calc_utility(utility_df, member_id, accepted_offers):
    return list(map(lambda offer_id: utility_df.loc[member_id, offer_id], accepted_offers))

def perform_timeline_sense_check(events_df, issues):
    assert True

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


#################### END OF CODE FOR CALCULATING ##############################
###############################################################################
######################## CODE FOR PLOTTING ####################################


def plot_offer_stats():
    return

def plot_batch_stats(all_run_data, base_data, all_run_info):
    # Extract base batch data
    _, _, base_batch_stats = base_data

    for i, (run_name, run_data) in enumerate(all_run_data.items()):
        print(run_name)
        # Extract simulation batch data
        _, _, batch_stats_all_sims = run_data

        ####### Category Histogram #############################
        fig_name = 'time_between_batches'
        plt.figure(fig_name)
        duration_categories = [ (pd.Timedelta(seconds=20), "Less than 10 seconds"),
                                (pd.Timedelta(minutes=1), "Less than 1 minute"),
                                (pd.Timedelta(minutes=10), "Less than 10 minutes"),
                                (pd.Timedelta(hours=1), "Less than 1 hour"),
                                (pd.Timedelta(hours=10), "Less than 10 hours"),
                                (pd.Timedelta(days=1), "Less than 1 day"),
                                (pd.Timedelta(days=3), "Less than 3 days"),
                                (pd.Timedelta(days=6), "Less than 6 days"),
                                (pd.Timedelta(days=26), "Less than 25 days"),
                                ]

        def get_duration_category(duration):
            if pd.isna(duration):
                return "First batch"
            duration_smaller_than_cats = list(filter(lambda cat: duration <= cat[0], duration_categories))
            category_tuple = duration_smaller_than_cats[0]
            return category_tuple[1]

        # Convert timedeltas to category_string
        batch_stats_all_sims['duration_category'] = batch_stats_all_sims['duration_since_prev_batch'].apply(get_duration_category)
        categories = list(map(lambda cat: cat[1], duration_categories))
        cat_counts = batch_stats_all_sims['duration_category'].value_counts()
        for cat in categories:
            if cat not in cat_counts.index:
                cat_counts[cat] = 0

        # Filter out 'First batch' category
        valid_categories = list(filter(lambda cat: cat in cat_counts.index, categories))

        # Transform category labels into counts
        cat_counts = pd.DataFrame([cat_counts.values / np.sum(cat_counts.values)], columns=cat_counts.index)[valid_categories]
        plt.bar(cat_counts.columns, np.cumsum(cat_counts.values.reshape(-1)), alpha=0.5, color=colors[i], label=run_name)

        # Last iteration, style plot
        if i == len(all_run_data)-1:
            plt.legend(fontsize=FONTSIZE)
            plt.xticks(fontsize=FONTSIZE)
            plt.yticks(fontsize=FONTSIZE)
            plt.ylabel("percent of batches", fontsize=FONTSIZE)
            plt.xticks(rotation=75, fontsize=17)
            # Save figure
            # plt.savefig("./plots/time_between_batches", bbox_inches='tight', pad_inches=0.05, dpi=150)


        ####### Data Quantiles #############################
        fig_name = 'time_between_batches' + '_quantiles'
        plt.figure(fig_name, figsize=(8,7))
        quantiles = [0,1,10,25,50,50,75,90,99,100]

        # Convert timedeltas to seconds, then to days
        data = batch_stats_all_sims['duration_since_prev_batch'].astype('timedelta64[s]') / 60 / 60 / 24
        data = data[~data.isna()].values
        # Compute and plot data quantiles
        data_quantiles = np.percentile(data, np.sort(np.unique(quantiles)))
        plt.plot(data_quantiles, 'o-', color=colors[i], label=run_name)

        # Last iteration, style plot
        if i == len(all_run_data)-1:
            def nr_to_label(nr):
                s = str(nr)+"%"
                if nr == 0:
                    s += "\n(min.)"
                elif nr == 50:
                    s+= "\n(median)"
                elif nr == 100:
                    s += "\n(max.)"
                return s

            plt.xticks([0,1,2,3,4,5,6,7,8], list(map(nr_to_label, np.sort(np.unique(quantiles)))), fontsize=FONTSIZE)
            plt.yticks(fontsize=FONTSIZE)
            plt.xlabel("\ndata quantiles", fontsize=FONTSIZE)
            plt.ylabel("days", fontsize=FONTSIZE)
            plt.legend(fontsize=FONTSIZE)
            # Save figure
            # plt.savefig("./plots/time_between_batches_quantiles", bbox_inches='tight', pad_inches=0.05, dpi=150)


def plot_monthly_stats(all_run_data, base_data):
    # Extract base monthly data
    _, base_monthly_stats, _ = base_data


    for i, (run_name, run_data) in enumerate(all_run_data.items()):
        # Extract simulation monthly data
        _, monthly_stats_all_sims, _ = run_data

        # Extract right sides of intervals
        interval_ends = list(map(lambda interval: interval.right, monthly_stats_all_sims['nr_accepted'].columns))

        # New figure
        plt.figure('Trash bin size per month')
        col_name = 'nr_expired'
        # Plot baseline
        if i == 0 and SHOW_BASELINE:
            plt.plot(interval_ends, base_monthly_stats[col_name], label='baseline', color=base_col)

        # Plot simulation
        nr_per_month_all_sims = monthly_stats_all_sims[col_name]
        avg_per_month = nr_per_month_all_sims.mean()
        std_dev_per_month = nr_per_month_all_sims.std()
        plt.fill_between(interval_ends, avg_per_month - 2*std_dev_per_month, avg_per_month + 2*std_dev_per_month, label=run_name, color=colors[i], alpha=0.5)

        # Last iteration, style the plot
        if i == len(all_run_data)-1:
            plt.legend(fontsize=15, loc='upper center')
            plt.suptitle('Trash bin size per month', fontsize=FONTSIZE)
            plt.xticks(rotation=30, fontsize=FONTSIZE)
            plt.yticks(fontsize=FONTSIZE)
            plt.ylabel("number of coupons", fontsize=FONTSIZE)

        # New figure
        plt.figure('Pass-through rate per month')
        # Plot baseline
        if i == 0 and SHOW_BASELINE:
            plt.plot(interval_ends, base_monthly_stats['nr_sent'] / base_monthly_stats['nr_accepted'], label='baseline', color=base_col)

        # Plot simulation
        pass_through_rate_all_sims = monthly_stats_all_sims['nr_sent'] / monthly_stats_all_sims['nr_accepted']
        avg_PTR_per_month = pass_through_rate_all_sims.mean()
        std_dev_PTR_per_month = pass_through_rate_all_sims.std()
        plt.fill_between(interval_ends, avg_PTR_per_month - 2*std_dev_PTR_per_month, avg_PTR_per_month + 2*std_dev_PTR_per_month, label=run_name, color=colors[i], alpha=0.5)

        # Last iteration, style the plot
        if i == len(all_run_data)-1:
            plt.legend(fontsize=15, loc='upper center')
            plt.suptitle('Pass-through rate per month', fontsize=FONTSIZE)
            plt.xticks(rotation=30, fontsize=FONTSIZE)
            plt.yticks(fontsize=FONTSIZE)
            plt.ylabel("number of coupons", fontsize=FONTSIZE)

    # Save both the figures
    # plt.figure('Pass-through rate per month')
    # plt.savefig("./plots/pass_through_rate", bbox_inches='tight', pad_inches=0.05, dpi=150)
    # plt.figure('Trash bin size per month')
    # plt.savefig("./plots/trash_bin", bbox_inches='tight', pad_inches=0.05, dpi=150)



######################## PLOTTING MEMBER STATISTICS ####################################


def gini(x):
    x = np.asarray(x)
    sorted_x = np.sort(x)
    n = len(x)
    cumx = np.cumsum(sorted_x, dtype=float)
    return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n

def summarize_distribution(member_utils_all_sims):
    all_sim_summaries = {}
    for sim_name in member_utils_all_sims.columns:
        sim_values = member_utils_all_sims[sim_name]
        sim_summary = {}
        sim_summary['sum'] = sim_values.sum()
        sim_summary['average'] = sim_values.mean()
        sim_summary['avg_nonzero'] = sim_values[sim_values>0].mean()
        sim_summary['max'] = sim_values.max()
        sim_summary['min_nonzero'] = sim_values[sim_values>0].min()
        sim_summary['median'] = np.median(sim_values)
        sim_summary['median_nonzero'] = np.median(sim_values[sim_values>0])
        sim_summary['gini'] = gini(sim_values)

        all_sim_summaries[sim_name] = sim_summary

    all_sim_summaries = pd.DataFrame.from_dict(all_sim_summaries, orient='index')
    return all_sim_summaries


def plot_member_stats(all_run_data, base_data):
    # Extract the base data
    base_member_stats, base_monthly_stats, _ = base_data

    # Determine whether to filter on a particular set of members
    if os.path.exists(global_folder + 'relevant_members.pkl'):
        with open(global_folder + 'relevant_members.pkl', 'rb') as fp:
            relevent_members = pickle.load(fp)
    else:
        relevent_members = set(base_member_stats['member_id'].values)

    # Filter on these members and make a summary of the distribution (i.e. mean, median, min, max, gini, etc.)
    base_member_stats = base_member_stats[base_member_stats['member_id'].isin(relevent_members)]
    base_summary = summarize_distribution(base_member_stats[['tot_worth_accepted_coupon']])

    # To store data in to make a final DataFrame / table
    summary_in_numbers_all_runs = {}
    summary_in_numbers_baseline = {}
    saved_fig_data = {}

    # Loop over all runs
    for i, (run_name, run_data) in enumerate(all_run_data.items()):
        print("Plotting data from run %d '%s'"%(i+1,run_name))
        summary_in_numbers = {}

        # Extract data from run_data
        member_stats_all_sims, monthly_stats_all_sims, _ = run_data
        member_stats_all_sims = member_stats_all_sims[member_stats_all_sims.index.isin(relevent_members)]
        nr_simulations = len(member_stats_all_sims['tot_worth_accepted_coupon'].columns)


        ####### Lorenz Curve Plot #############################
        fig_name = 'Lorenz Curve'
        plt.figure(fig_name, figsize=(8,6.2))
        if i == 0:
            # First iteration, plot equality line
            equality = np.arange(len(member_stats_all_sims))/len(member_stats_all_sims)
            plt.plot(equality, 'k--', alpha=0.5, label='equality')
            # First iteration, plot baseline
            if SHOW_BASELINE:
                baseline = base_member_stats['tot_worth_accepted_coupon'].values
                plt.plot(np.cumsum(np.sort(baseline)/np.sum(baseline)), label='baseline', color=base_col, linewidth=1)

        # Plot all simulations of this run_name
        sorted_utils = np.sort(member_stats_all_sims['tot_worth_accepted_coupon'].values, axis=0)
        avg_utils = np.average(sorted_utils, axis=1)
        plt.plot(np.cumsum(np.sort(avg_utils)/np.sum(avg_utils)), label=run_name, color=colors[i], linewidth=1)

        # Last iteration, style the plot
        if i == len(all_run_data)-1:
            plt.yticks(fontsize=FONTSIZE)
            plt.xticks(fontsize=FONTSIZE)
            plt.xlabel("members", fontsize=FONTSIZE)
            plt.ylabel("percentage of total utility", fontsize=FONTSIZE)
            plt.suptitle(fig_name, fontsize=FONTSIZE)
            # Set the legend
            ax = plt.gca()
            leg = ax.legend(fontsize=FONTSIZE)
            for line in leg.get_lines():
                line.set_linewidth(2.0) # change the line width for the legend
            # Save the figure
            # plt.savefig("./plots/lorenze_curve", bbox_inches='tight', pad_inches=0.05, dpi=150)


        ####### Sorted Utilities Plot #############################
        fig_name = 'sorted_utilities'
        plt.figure(fig_name)
        # Plot simulation data
        plt.plot(np.sort(avg_utils), label=run_name, color=colors[i])
        # Last iteration, style the plot
        if i == len(all_run_data)-1:
            # Optionally plot the baseline
            if SHOW_BASELINE:
                plt.plot(np.sort(base_member_stats['tot_worth_accepted_coupon'].values), label='baseline', color=base_col)
            # Style the plot
            plt.yticks(fontsize=FONTSIZE)
            plt.xticks(fontsize=FONTSIZE)
            plt.xlabel("members", fontsize=FONTSIZE)
            plt.ylabel("percentage of total utility", fontsize=FONTSIZE)
            plt.suptitle(fig_name, fontsize=FONTSIZE)


        ####### Utility Distribution Summary Subplots #######################################
        fig_name = 'Utility distribution summaries'
        plt.figure(fig_name, figsize=(10,10))
        # Retrieve various summary measures of the utility distribution
        utility_summary_to_plot = summarize_distribution(member_stats_all_sims['tot_worth_accepted_coupon'])

        # Loop over all 4 measures to plot
        subplot_data_cols = ['average', 'avg_nonzero', 'median', 'gini']
        for j, summary_col_name in enumerate(subplot_data_cols, 1):
            plt.subplot(2,2,j)

            # Extract the measure data and plot
            data = utility_summary_to_plot[summary_col_name].values
            plt.hist(data, bins=20, alpha=0.5, color=colors[i], label=run_name, density=True)
            # Fit a normal distribution to the data
            xmin, xmax = plt.xlim()
            x = np.linspace(xmin, xmax, 100)
            mu, std = scipy.stats.norm.fit(data)
            p = scipy.stats.norm.pdf(x, mu, std)
            plt.plot(x, p, linewidth=2, color=colors[i])

            # Add some data to generate a DataFrame
            if summary_col_name != 'gini':
                summary_in_numbers['avg %s utility per member'%(summary_col_name)] = np.average(data)
                summary_in_numbers['std %s utility per member'%(summary_col_name)] = np.std(data)
            else:
                summary_in_numbers['avg gini coefficient'] = np.average(data)
                summary_in_numbers['std gini coefficient'] = np.std(data)

            # Last iteration, style the plot
            if i == len(all_run_data)-1:
                # Optionally plot the baseline
                if SHOW_BASELINE:
                    x_value = base_summary[summary_col_name]
                    xmin, xmax, ymin, ymax = plt.axis()
                    plt.plot([x_value, x_value], [ymin, ymax], '-.', color=base_col, linewidth=2, label='baseline')
                    plt.ylim([ymin, ymax])
                    # Add data to dictionary for dataframe
                    if summary_col_name == 'gini':
                        summary_in_numbers_baseline['avg gini coefficient'] = x_value.values[0]
                    else:
                        summary_in_numbers_baseline['avg %s utility per member'%(summary_col_name)] = x_value.values[0]
                # Style the plot
                plt.title(summary_col_name)
                plt.suptitle(fig_name, fontsize=FONTSIZE)
                plt.legend(fontsize=FONTSIZE)


        ####### All combined histograms #######################################
        columns_to_plot = ['tot_worth_accepted_coupon', 'nr_accepted', 'nr_sent']
        cut_off_value = 50

        for col in columns_to_plot:
            # Define fig_name
            fig_name = '%s distribution'%col if col!='tot_worth_accepted_coupon' else 'Utility distribution'
            if '%s_boxplot'%col not in saved_fig_data: saved_fig_data['%s_boxplot'%col] = {}
            plot_data = saved_fig_data['%s_boxplot'%col]

            # Get the data
            plt.figure('%s_histogram'%col)
            to_plot = member_stats_all_sims[col].values.reshape(-1)
            # Save data for boxplot later on
            plot_data[run_name] = to_plot
            # Set cut_off_value and plot
            to_plot[to_plot > cut_off_value] = cut_off_value
            bins = np.min([int(np.max(to_plot)), cut_off_value]) + 1
            plt.hist(to_plot, bins=bins, alpha=0.5, range=None, color=colors[i], label=run_name, density=False)

            # Last iteration, style the plot
            if i == len(all_run_data)-1:
                # Optionally plot the baseline
                if SHOW_BASELINE:
                    # Get the data
                    values = base_member_stats[col].values
                    to_plot = np.array([values]*nr_simulations).reshape(-1)
                    # Save data for boxplot later on
                    plot_data['baseline'] = to_plot
                    # Set cut_off_value and plot
                    to_plot[to_plot > cut_off_value] = cut_off_value
                    bins = np.min([int(np.max(to_plot)), cut_off_value]) + 1
                    plt.hist(to_plot, bins=bins, alpha=0.3, color=base_col, label='baseline', density=False)

                # Scale the y axis labels
                ax = plt.gca()
                def scale_ints(x, *args):
                    return "%d"%(int(float(x)/nr_simulations))
                ax.yaxis.set_major_formatter(mtick.FuncFormatter(scale_ints))
                # Format the x axis labels according to cut-off-value
                def update_int_ticks(x, *args):
                    return "%d+"%x if x == cut_off_value else int(x)
                ax.xaxis.set_major_formatter(mtick.FuncFormatter(update_int_ticks))
                # Other styling
                plt.suptitle(fig_name, fontsize=FONTSIZE)
                plt.legend(fontsize=FONTSIZE)


        # Save all the gathered data under its correct 'run_name'
        summary_in_numbers_all_runs[run_name] = summary_in_numbers


        ###### TABLE DATA ##############################
        nr_members = len(member_stats_all_sims['nr_sent'])

        perc_received_any_coupon = np.sum(member_stats_all_sims['nr_sent'] > 0) / nr_members
        summary_in_numbers['avg perc_received_any_coupon'] = perc_received_any_coupon.mean()
        summary_in_numbers['std perc_received_any_coupon'] = perc_received_any_coupon.std()
        if SHOW_BASELINE:
            perc_received_any_coupon = np.sum(base_member_stats['nr_sent']>0) / nr_members
            summary_in_numbers_baseline['avg perc_received_any_coupon'] = perc_received_any_coupon

        perc_accepted_any_coupon = np.sum(member_stats_all_sims['nr_accepted'] > 0) / nr_members
        summary_in_numbers['avg perc_accepted_any_coupon'] = perc_accepted_any_coupon.mean()
        summary_in_numbers['std perc_accepted_any_coupon'] = perc_accepted_any_coupon.std()
        if SHOW_BASELINE:
            perc_accepted_any_coupon = np.sum(base_member_stats['nr_accepted']>0) / len(base_member_stats['nr_accepted'])
            summary_in_numbers_baseline['avg perc_accepted_any_coupon'] = perc_accepted_any_coupon

        sum_nr_expired_per_sim = np.sum(monthly_stats_all_sims['nr_expired'], axis=1).values
        summary_in_numbers['avg total nr_expired'] = sum_nr_expired_per_sim.mean()
        summary_in_numbers['std total nr_expired'] = sum_nr_expired_per_sim.std()
        if SHOW_BASELINE:
            nr_expired = base_monthly_stats['nr_expired'].sum()
            summary_in_numbers_baseline['avg total nr_expired'] = nr_expired

        sum_nr_accepted_per_sim = member_stats_all_sims['nr_accepted'].sum().values
        summary_in_numbers['avg total nr_accepted'] = sum_nr_accepted_per_sim.mean()
        summary_in_numbers['std total nr_accepted'] = sum_nr_accepted_per_sim.std()
        if SHOW_BASELINE:
            nr_accepted = base_member_stats['nr_accepted'].sum()
            summary_in_numbers_baseline['avg total nr_accepted'] = nr_accepted

        perc_expired_per_sim = sum_nr_expired_per_sim / (sum_nr_expired_per_sim + sum_nr_accepted_per_sim)
        summary_in_numbers['avg perc expired'] = perc_expired_per_sim.mean()
        summary_in_numbers['std perc expired'] = perc_expired_per_sim.std()
        if SHOW_BASELINE:
            perc_expired = nr_expired / (nr_expired + nr_accepted)
            summary_in_numbers_baseline['avg perc expired'] = perc_expired

        total_utility_per_sim = np.sum(member_stats_all_sims['tot_worth_accepted_coupon'])
        summary_in_numbers['avg total utility'] = total_utility_per_sim.mean()
        summary_in_numbers['std total utility'] = total_utility_per_sim.std()
        if SHOW_BASELINE:
            total_utility_baseline = np.sum(base_member_stats['tot_worth_accepted_coupon'])
            summary_in_numbers_baseline['avg total utility'] = total_utility_baseline


    # After looping over all runs and having saved up some data, make the final plots
    #### PLOT SAVED UP DATA ###############
    fig_names_to_plot_from_saved_data = ['nr_sent', 'nr_accepted', 'tot_worth_accepted_coupon']
    for dict_key in fig_names_to_plot_from_saved_data:
        if dict_key == 'nr_sent':
            fig_name = "Number of received coupons per member"
        elif dict_key == 'nr_accepted':
            fig_name = "Number of accepted coupons per member"
        elif dict_key == 'tot_worth_accepted_coupon':
            fig_name = "Member utilities"


        ###### Distribution Boxplots ###############
        plt.figure(dict_key + '_boxplot', figsize=(10,7))
        plt.suptitle(fig_name, fontsize=FONTSIZE)

        boxplot_data = saved_fig_data[dict_key + '_boxplot']
        if ALL_SIMS_BOXPLOT == 'batch_size':
            to_plot = [boxplot_data['baseline'],
                       boxplot_data['greedy_1_TD'], boxplot_data['greedy_50_TD'], boxplot_data['greedy_200_TD'],
                       boxplot_data['greedy_1_FH'], boxplot_data['greedy_50_FH'], boxplot_data['greedy_200_FH'],
                       boxplot_data['greedy_1_PH'], boxplot_data['greedy_50_PH'], boxplot_data['greedy_200_PH'],
                       boxplot_data['greedy_1_TD'], boxplot_data['max_sum_50_TD'], boxplot_data['max_sum_200_TD'],
                       boxplot_data['max_sum_1_FH'], boxplot_data['max_sum_50_FH'], boxplot_data['max_sum_200_FH'],
                       boxplot_data['greedy_1_PH'], boxplot_data['max_sum_50_PH'], boxplot_data['max_sum_200_PH'],
                       ]
            labels = ['baseline',
                      'greedy_1_TD', 'greedy_50_TD', 'greedy_200_TD',
                      'greedy_1_FH', 'greedy_50_FH', 'greedy_200_FH',
                      'greedy_1_PH', 'greedy_50_PH', 'greedy_200_PH',
                      'max_sum_1_TD', 'max_sum_50_TD', 'max_sum_200_TD',
                      'max_sum_1_FH', 'max_sum_50_FH', 'max_sum_200_FH',
                      'max_sum_1_PH', 'max_sum_50_PH', 'max_sum_200_PH',
                      ]
            plt.boxplot(to_plot, positions=[0,2,3,4,6,7,8,10,11,12,14,15,16,18,19,20,22,23,24])
        elif ALL_SIMS_BOXPLOT == 'objective':
            # Max_sum vs Greedy
            plt.boxplot(list(boxplot_data.values()), positions=[0,2,3,5,6,8,9,11,12,14,15,17,18,20,21])
            labels = list(boxplot_data.keys())
        elif ALL_SIMS_BOXPLOT == 'historical_context':
            plt.boxplot(list(boxplot_data.values()), positions=[0,2,3,4,6,7,8,10,11,12,14,15,16,18,19,20])
            labels = list(boxplot_data.keys())
        else:
            plt.boxplot(list(boxplot_data.values()))
            labels = list(boxplot_data.keys())

        # Styling
        ax = plt.gca()
        ax.set_xticklabels(labels)
        plt.xticks(rotation=80, fontsize=FONTSIZE)
        plt.yticks(fontsize=FONTSIZE)
        plt.ylabel("utility", fontsize=FONTSIZE)
        xmin, xmax, ymin, ymax = plt.axis()
        plt.ylim(-2, 50 + 2)
        # plt.savefig("./plots/all_batch_size_boxplot", bbox_inches='tight', pad_inches=0.05, dpi=150)


        ########### Distribution Quantiles #################
        plt.figure(dict_key + '_quantiles')
        quantiles = [0,1,10,25,50,50,75,90,99,100]

        for k, (run_name, run_data) in enumerate(boxplot_data.items()):
            data_quantiles = np.percentile(run_data, np.sort(np.unique(quantiles)))
            if SHOW_BASELINE:
                color = base_col if run_name == 'baseline' else colors[k-1]
            else:
                color = colors[k]
            plt.plot(data_quantiles, 'o-', color=color, label=run_name)

        # Adding labels to the xticks
        def nr_to_label(nr):
            s = str(nr)+"%"
            if nr == 0:
                s += "\n(min.)"
            elif nr == 50:
                s+= "\n(median)"
            elif nr == 100:
                s += "\n(max.)"
            return s

        # Add some labels to the x ticks
        plt.xticks([0,1,2,3,4,5,6,7,8], list(map(nr_to_label, np.sort(np.unique(quantiles)))))
        # Styling
        plt.xticks(fontsize=FONTSIZE)
        plt.yticks(fontsize=FONTSIZE)
        plt.legend(fontsize=FONTSIZE)
        plt.xlabel("\ndata quantile", fontsize=FONTSIZE)

        plt.suptitle(fig_name, fontsize=FONTSIZE)
        if dict_key == 'tot_worth_accepted_coupon':
            plt.ylabel("utility", fontsize=FONTSIZE)
            # plt.savefig("./plots/tot_utility_quantiles", bbox_inches='tight', pad_inches=0.05, dpi=150)
        elif dict_key == 'nr_sent':
            plt.ylabel("number of coupons", fontsize=FONTSIZE)
            # plt.savefig("./plots/nr_sent_quantiles", bbox_inches='tight', pad_inches=0.05, dpi=150)
        elif dict_key == 'nr_accepted':
            plt.ylabel("number of coupons", fontsize=FONTSIZE)
            # plt.savefig("./plots/nr_accepted_quantiles", bbox_inches='tight', pad_inches=0.05, dpi=150)


    # Turn the dictionary into a dataframe, and export
    summary_in_numbers_all_runs['baseline'] = summary_in_numbers_baseline
    summary_in_numbers_all_runs = pd.DataFrame.from_dict(summary_in_numbers_all_runs, orient='index')
    filename = './plots/data.csv'
    # Make the file writable
    if os.path.exists(filename):
        os.chmod(filename, S_IWUSR|S_IREAD)
    # Export to the file
    summary_in_numbers_all_runs.T.to_csv(filename)
    # Make the file read-only again
    os.chmod(filename, S_IREAD|S_IRGRP|S_IROTH)

    runs = list(all_run_data.keys())
    if len(runs) >= 2:
        statistical_test(runs, summary_in_numbers_all_runs)


def statistical_test(runs, summary_in_numbers_all_runs):
    # Perform a statistical test between run0 and run1 by calculating p-values
    one = runs[0]
    two = runs[1]

    n1 = 100
    n2 = 100
    mu1 = summary_in_numbers_all_runs.loc[one, 'avg average utility per member']
    s1 = summary_in_numbers_all_runs.loc[one, 'std average utility per member']
    mu2 = summary_in_numbers_all_runs.loc[two, 'avg average utility per member']
    s2 = summary_in_numbers_all_runs.loc[two, 'std average utility per member']

    dof = n1 + n2 - 2
    spool = np.sqrt(((n1-1) * (s1**2) + (n2-1) * (s2**2)) / dof)
    t_stat = (mu1 - mu2) / (spool  * np.sqrt(1/n1 + 1/n2))

    plt.figure()
    plt.title("Statistical test distribution")
    x = np.linspace(4, 5, 100)
    p = scipy.stats.norm.pdf(x, mu1, s1**2)
    plt.plot(x, p, linewidth=2)
    p = scipy.stats.norm.pdf(x, mu2, s2**2)
    plt.plot(x, p, linewidth=2)

    # p-value for 2-sided test
    p1 = 2*(1 - scipy.stats.t.cdf(abs(t_stat), dof))
    # two-sided pvalue = Prob(abs(t)>tt)
    p2 = scipy.stats.t.sf(np.abs(t_stat), dof)*2
    # p1 and p2 should be the same for values larger than 1e-20
    # print(p1, p2)



if __name__ == '__main__':
    main()
