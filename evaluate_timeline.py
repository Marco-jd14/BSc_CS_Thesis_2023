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
from dateutil.relativedelta import relativedelta
from database.lib.tracktime import TrackTime, TrackReport

import database.connect_db as connect_db
import database.query_db as query_db
Event = query_db.Event

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 90)

global_folder = './timelines/'
run_folder = './timelines/run_%d/'

SHOW_BASELINE = True
RECOMPUTE = True

def main():

    # baseline = pd.read_csv('./timelines/baseline_events.csv')
    # baseline.to_pickle('./timelines/baseline_events.pkl')
    run_nrs_to_read = [45]

    TrackTime("Connect to db")
    conn = connect_db.establish_host_connection()
    db   = connect_db.establish_database_connection(conn)
    print("Successfully connected to database '%s'"%str(db.engine).split("/")[-1][:-1])

    TrackTime("Retrieve from db")
    result = query_db.retrieve_from_sql_db(db, 'filtered_issues', 'member')
    filtered_issues, members = result
    print("")


    # Read simulated events and info
    TrackTime("Get all run data")
    all_run_data, all_run_info, all_run_nr_info = {}, {}, {}
    for run_nr in run_nrs_to_read:
        if RECOMPUTE or not os.path.exists(run_folder%run_nr + '%d_run_data.pkl'%run_nr):

            utility_df = pd.read_pickle(run_folder%run_nr + "%d_utility_df.pkl"%run_nr)
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


        f = open(run_folder%run_nr + "%d_info.json"%run_nr)
        run_info = json.load(f)
        f.close()

        # # Redefine run-name
        # run_name = run_info['allocator_algorithm'] + "_" + str(run_info['batch_size'])
        # run_name += '_' + "".join(list(map(lambda word: word.upper()[0], run_info['utility_type'].split('_'))))
        # if 'version_tag' in run_info.keys() and run_info['version_tag'].strip() != "":
        #     run_name += "_" + run_info['version_tag']
        # with open(run_folder%run_nr + '%d_run_name.pkl'%run_nr, 'wb') as fp:
        #     pickle.dump(run_name, fp)

        all_run_data[run_name] = run_data
        all_run_info[run_name] = run_info
        all_run_nr_info[run_nr] = run_info

    # f = open(global_folder + 'run_info_overview.json')
    # prev_all_run_nr_info = json.load(f)
    # f.close()
    # prev_all_run_nr_info.update(all_run_nr_info)
    # with open(global_folder + 'run_info_overview.json', 'w') as fp:
    #     json.dump(prev_all_run_nr_info, fp)

    # Read utility
    utility_df = pd.read_pickle(global_folder + "new_utility_df.pkl")

    # Read baseline events
    base_events = pd.read_pickle(global_folder + 'baseline_events.pkl')
    base_events['event'] = base_events['event'].apply(lambda event: Event[str(event)[len("Event."):]])
    base_events = base_events.convert_dtypes()
    # Extract baseline information
    perform_timeline_sense_check(base_events, filtered_issues)
    base_data = extract_timeline_info(base_events, utility_df, members['id'])

    # Evaluate and plot
    TrackTime("Evaluate")
    evaluate_timelines(all_run_data, base_data, utility_df, members, filtered_issues['id'])
    TrackTime("Member plots")
    plot_member_stats(all_run_data, base_data)
    TrackTime("Monthly plots")
    plot_monthly_stats(all_run_data, base_data)
    # TrackTime("Batch plots")
    # plot_batch_stats(all_run_data, base_data, all_run_info)

    db.close()
    TrackReport()


def extract_relevant_info(run_nr, utility_df, member_ids, filtered_issues):
    contents = os.listdir(run_folder%run_nr)
    timeline_files = list(filter(lambda name: re.search("^%d.[0-9]+_events_df.pkl"%run_nr, name), contents))
    timeline_nrs = sorted(list(map(lambda name: int(name.split('_')[0].split('.')[-1]), timeline_files)))

    TrackTime("Read info")
    f = open(run_folder%run_nr + "%d_info.json"%run_nr)
    run_info = json.load(f)
    f.close()

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

        for event_name in monthly_stats.columns:
            for month in list(monthly_stats.index):
                assert monthly_stats.loc[month, event_name] == monthly_stats_all_sims[event_name][month].iloc[sim_nr]


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


        TrackTime("Combine batch info")
        batch_stats['sim_nr'] = sim_nr
        if batch_stats_all_sims is None:
            batch_stats_all_sims = batch_stats
        else:
            batch_stats_all_sims = pd.concat([batch_stats_all_sims, batch_stats], ignore_index=True)


    member_stats_all_sims = member_stats_all_sims.sort_index(axis=1)

    run_name = run_info['allocator_algorithm'] + "_" + str(run_info['batch_size'])
    run_name += '_' + "".join(list(map(lambda word: word.upper()[0], run_info['utility_type'].split('_'))))
    if 'version_tag' in run_info.keys() and run_info['version_tag'].strip() != "":
        run_name += "_" + run_info['version_tag']
    return run_name, (member_stats_all_sims, monthly_stats_all_sims, batch_stats_all_sims)


def extract_timeline_info(events, utility_df, member_ids, batch_size=None):

    TrackTime("calculate_member_scores")
    member_events = events[~events['member_id'].isna()]
    member_stats = calculate_member_scores(member_events, utility_df, member_ids)
    assert len(set(member_stats['member_id'].values)) == len(member_stats)

    TrackTime("make_monthly_summary")
    monthly_stats = make_monthly_summary(events)

    TrackTime("other stats")
    batch_stats = get_batch_stats(events, batch_size)
    offer_stats = make_offer_stats(events, utility_df, member_ids)

    return member_stats, monthly_stats, batch_stats


def get_batch_stats(events, batch_size):
    sent_events = events[events['event'] == Event.coupon_sent]

    if 'batch_id' in events.columns:
        batch_info = sent_events.groupby('batch_id').aggregate(batch_sent_at=('at','first'), batch_size=('coupon_id','count')).reset_index(drop=True)
    else:
        batch_info = sent_events.groupby('at').aggregate(batch_size=('coupon_id','count')).reset_index().rename(columns={'at':'batch_sent_at'})

    batch_info['duration_since_prev_batch'] = batch_info['batch_sent_at'] - batch_info['batch_sent_at'].shift(1)
    return batch_info

def make_offer_stats(events, utility_df, member_ids):
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



def evaluate_timelines(all_run_data, base_data, utility_df, members, issue_ids):
    return



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
        median_index = int(0.5*len(sim_values))
        sim_summary['median'] = sim_values.sort_values().values[median_index]
        median_nonzero_index = int(0.5*np.sum(sim_values>0))
        sim_summary['median_nonzero'] = sim_values[sim_values>0].sort_values().values[median_nonzero_index]

        all_sim_summaries[sim_name] = sim_summary

    all_sim_summaries = pd.DataFrame.from_dict(all_sim_summaries, orient='index')
    return all_sim_summaries


def plot_offer_stats():
    return

def plot_batch_stats(all_run_data, base_data, all_run_info):
    fig_names = ['time_between_batches']

    _, _, base_batch_stats = base_data

    colors = ['red', 'blue', 'yellow']
    base_col = 'green'

    for i, (run_name, run_data) in enumerate(all_run_data.items()):
        _, _, batch_stats_all_sims = run_data
        
        # plt.figure('batch_size')
        # bins = list(np.linspace(min_batch_size - 0.5, min_batch_size + 10.5)) + [0.5 + max(batch_stats_all_sims['batch_size'])]
        # plt.hist(batch_stats_all_sims['batch_size'], bins=bins, alpha=0.5, color=colors[i], label=run_name, density=False)
        print(run_name)
        min_batch_size = all_run_info[run_name]['batch_size']
        print("avg batch size: %.2f"%np.average(batch_stats_all_sims['batch_size'].values))
        print("nr batches larger than batch size:", np.sum(batch_stats_all_sims['batch_size'].values > min_batch_size))
        print("nr batches smaller than batch size:", np.sum(batch_stats_all_sims['batch_size'].values < min_batch_size))

        plt.figure('time_between_batches')
        duration_categories = [(pd.Timedelta(seconds=3), "Less than 3 seconds"),
                               (pd.Timedelta(seconds=20), "Less than 20 seconds"),
                               (pd.Timedelta(minutes=1), "Less than 1 minute"),
                               (pd.Timedelta(minutes=20), "Less than 20 minutes"),
                               (pd.Timedelta(hours=1), "Less than 1 hour"),
                               (pd.Timedelta(hours=5), "Less than 5 hours"),
                               (pd.Timedelta(hours=10), "Less than 10 hours"),
                               (pd.Timedelta(days=1), "Less than 1 day"),
                               (np.max(batch_stats_all_sims['duration_since_prev_batch']), "More than 1 day")]

        def get_duration_category(duration):
            if pd.isna(duration):
                return "First batch"

            duration_smaller_than_cats = list(filter(lambda cat: duration <= cat[0], duration_categories))
            category_tuple = duration_smaller_than_cats[0]
            return category_tuple[1]


        batch_stats_all_sims['duration_category'] = batch_stats_all_sims['duration_since_prev_batch'].apply(get_duration_category)
        # duration_mins = batch_stats_all_sims['duration_since_prev_batch'].astype('timedelta64[s]') / 60
        categories = list(map(lambda cat: cat[1], duration_categories))
        cat_counts = batch_stats_all_sims['duration_category'][batch_stats_all_sims['duration_category'].isin(categories)].value_counts()
        cat_counts = pd.DataFrame([cat_counts.values / np.sum(cat_counts.values)], columns=cat_counts.index)[categories]
        plt.bar(cat_counts.columns, cat_counts.values.reshape(-1), alpha=0.5, color=colors[i], label=run_name)



    # Set the title and legend
    for fig_name in fig_names:
        plt.figure(fig_name)
        plt.suptitle(fig_name)
        plt.xticks(rotation=70)
        plt.legend()


def plot_monthly_stats(all_run_data, base_data):
    _, base_monthly_stats, _ = base_data

    colors = ['red', 'blue', 'yellow']
    base_col = 'green'

    for i, (run_name, run_data) in enumerate(all_run_data.items()):
        _, monthly_stats_all_sims, _ = run_data

        interval_ends = list(map(lambda interval: interval.right, monthly_stats_all_sims['nr_accepted'].columns))

        plt.figure('timelines', figsize=(10,10))
        subplot_data_cols = ['nr_sent', 'nr_expired', 'nr_accepted']
        for j, col_name in enumerate(subplot_data_cols, 1):
            plt.subplot(2,2,j)
            if i == 0 and SHOW_BASELINE:
                plt.plot(interval_ends, base_monthly_stats[col_name], label='baseline', color=base_col)
            nr_per_month_all_sims = monthly_stats_all_sims[col_name]
            avg_per_month = nr_per_month_all_sims.mean()
            std_dev_per_month = nr_per_month_all_sims.std()
            # plt.plot(interval_ends, avg_per_month, label=run_name, color=colors[i])
            plt.fill_between(interval_ends, avg_per_month - 2*std_dev_per_month, avg_per_month + 2*std_dev_per_month, label=run_name, color=colors[i], alpha=0.5)

            if i == len(all_run_data)-1:
                plt.legend()
                plt.title(col_name)


        plt.subplot(2,2,4)
        if i == 0 and SHOW_BASELINE:
            plt.plot(interval_ends, base_monthly_stats['nr_sent'] / base_monthly_stats['nr_accepted'], label='baseline', color=base_col)
        pass_through_rate_all_sims = monthly_stats_all_sims['nr_sent'] / monthly_stats_all_sims['nr_accepted']
        avg_PTR_per_month = pass_through_rate_all_sims.mean()
        std_dev_PTR_per_month = pass_through_rate_all_sims.std()
        # plt.plot(interval_ends, avg_PTR_per_month, label=run_name, color=colors[i])
        plt.fill_between(interval_ends, avg_PTR_per_month - 2*std_dev_PTR_per_month, avg_PTR_per_month + 2*std_dev_PTR_per_month, label=run_name, color=colors[i], alpha=0.5)

        if i == len(all_run_data)-1:
            plt.legend()
            plt.title('pass through rate')




def plot_member_stats(all_run_data, base_data):
    fig_names = ['avg_lorenz', 'sorted_utilities']

    base_member_stats, _, _ = base_data
    base_summary = summarize_distribution(base_member_stats[['total_utility']])

    colors = ['red', 'blue', 'yellow']
    base_col = 'green'

    for i, (run_name, run_data) in enumerate(all_run_data.items()):
        member_stats_all_sims, _, _ = run_data
        nr_simulations = len(member_stats_all_sims['total_utility'].columns)


        ####### Lorenz Curve Plot #############################
        plt.figure('avg_lorenz')
        if i == 0:
            equality = np.arange(len(member_stats_all_sims))/len(member_stats_all_sims)
            plt.plot(equality, 'k--', alpha=0.5, label='equality')

            baseline = base_member_stats['total_utility'].values
            plt.plot(np.cumsum(np.sort(baseline)/np.sum(baseline)), label='baseline', color=base_col, linewidth=1)

        sorted_utils = np.sort(member_stats_all_sims['total_utility'].values, axis=0)
        avg_utils = np.average(sorted_utils, axis=1)

        plt.plot(np.cumsum(np.sort(avg_utils)/np.sum(avg_utils)), label=run_name, color=colors[i], linewidth=1)


        ####### Sorted Utilities Plot #############################
        plt.figure('sorted_utilities')
        if i == 0:
            plt.plot(np.sort(base_member_stats['total_utility'].values), label='baseline', color=base_col)
        plt.plot(np.sort(avg_utils), label=run_name, color=colors[i])


        ####### Complete Summary Subplots #######################################
        columns_to_plot = ['total_utility']
        fig_names.extend(list(map(lambda s: '%s_summary'%s, columns_to_plot)))

        for data_col_name in columns_to_plot:
            plt.figure('%s_summary'%data_col_name, figsize=(10,15))
            summary_to_plot = summarize_distribution(member_stats_all_sims[data_col_name])
            subplot_data_cols = ['average', 'avg_nonzero', 'median', 'median_nonzero', 'max', 'min_nonzero']
            for j, summary_col_name in enumerate(subplot_data_cols, 1):
                plt.subplot(3,2,j)
                data = summary_to_plot[summary_col_name].values
                plt.hist(data, bins=12, alpha=0.5, color=colors[i], label=run_name, density=False)

                if i == len(all_run_data)-1:
                    # if SHOW_BASELINE:
                    #     x_value = base_summary[summary_col_name]
                    #     xmin, xmax, ymin, ymax = plt.axis()
                    #     plt.plot([x_value, x_value], [ymin, ymax], '-.', color=base_col, linewidth=5, label='baseline')
                    #     plt.ylim([ymin, ymax])
                    plt.legend()
                    plt.title(summary_col_name)


        ####### Sent Coupon Worth 2 * Subplots #######################################
        columns_to_plot = ['avg_worth_sent_coupon', 'median_worth_sent_coupon',
                          'min_worth_sent_coupon', 'max_worth_sent_coupon']
        fig_names.append('average worth_sent_coupons per simulation')

        plt.figure('average worth_sent_coupons per simulation', figsize=(10,10))
        for j, data_col_name in enumerate(columns_to_plot, 1):

            summary_to_plot = summarize_distribution(member_stats_all_sims[data_col_name])
            plt.subplot(2,2,j)
            data = summary_to_plot['average'].values
            plt.hist(data, bins=12, alpha=0.5, color=colors[i], label=run_name, density=False)

            if i == len(all_run_data)-1:
                # if SHOW_BASELINE:
                #     x_value = base_summary['average']
                #     xmin, xmax, ymin, ymax = plt.axis()
                #     plt.plot([x_value, x_value], [ymin, ymax], '-.', color=base_col, linewidth=5, label='baseline')
                #     plt.ylim([ymin, ymax])
                plt.legend()
                plt.title('average of [%s sent-coupon-utility per member]'%data_col_name.split("_")[0])

        plt.figure('worth_sent_coupons per member', figsize=(10,10))
        fig_names.append('worth_sent_coupons per member')
        only_nonzero = True
        prev_ax = None
        for j, col in enumerate(columns_to_plot, 1):
            ax = plt.subplot(2,2,j, sharex=prev_ax, sharey=prev_ax)
            prev_ax = ax
            flat_values = member_stats_all_sims[col].values.reshape(-1)
            to_plot = flat_values[flat_values>0] if only_nonzero else flat_values
            plt.hist(to_plot, bins=30, alpha=0.5, range=None, color=colors[i], label=run_name, density=False)

            if i == len(all_run_data)-1:
                if SHOW_BASELINE:
                    values = base_member_stats[col].values
                    to_plot = np.array([values[values>0] if only_nonzero else values]*nr_simulations).reshape(-1)
                    plt.hist(to_plot, bins=30, alpha=0.3, color=base_col, label='baseline', density=False)

                plt.legend()
                plt.title('%s worth of sent-coupon-utilities per member'%col.split("_")[0])
                def scale_floats(x, *args):
                    return "{:.1f}".format(float(x)/nr_simulations)
                ax.yaxis.set_major_formatter(mtick.FuncFormatter(scale_floats))


        ####### All combined histograms #######################################
        columns_to_plot = ['total_utility', 'nr_accepted', 'nr_sent']#'nr_declined', 'nr_let_expire']
        fig_names.extend(list(map(lambda s: '%s_histogram'%s, columns_to_plot)))
        only_nonzeros = [True, False, False]# False, False]
        cut_off_value = 50
        bins = cut_off_value + 1

        for col, only_nonzero in zip(columns_to_plot, only_nonzeros):
            plt.figure('%s_histogram'%col)
            if i == 0 and SHOW_BASELINE:
                values = base_member_stats[col].values
                to_plot = values[values>0] if only_nonzero else values
                to_plot = np.array([to_plot]*nr_simulations).reshape(-1)
                to_plot[to_plot > cut_off_value] = cut_off_value
                plt.hist(to_plot, bins=bins, alpha=0.3, color=base_col, label='baseline', density=False)

            flat_values = member_stats_all_sims[col].values.reshape(-1)
            to_plot = flat_values[flat_values>0] if only_nonzero else flat_values
            to_plot[to_plot > cut_off_value] = cut_off_value
            plt.hist(to_plot, bins=bins, alpha=0.5, range=None, color=colors[i], label=run_name, density=False)

            if i == len(all_run_data)-1:
                def scale_ints(x, *args):
                    return "%d"%(int(float(x)/nr_simulations))
                ax = plt.gca()
                ax.yaxis.set_major_formatter(mtick.FuncFormatter(scale_ints))

                def update_int_ticks(x, *args):
                    return "%d+"%x if x == cut_off_value else int(x)
                ax.xaxis.set_major_formatter(mtick.FuncFormatter(update_int_ticks))


    # Set the title and legend
    for fig_name in fig_names:
        plt.figure(fig_name)
        plt.suptitle(fig_name)
        plt.legend()
        # plt.save()
        # plt.show()





def calculate_member_scores(events_df, utility_df, member_ids):

    scores = events_df[['member_id','event','coupon_id']].pivot_table(index=['member_id'], columns='event', aggfunc='count', fill_value=0)['coupon_id'].reset_index()
    scores.columns = [scores.columns[0]] + list(map(lambda event: "nr_" + "_".join(str(event).split('_')[1:]), scores.columns[1:]))
    scores = scores[['member_id', 'nr_sent', 'nr_accepted',  'nr_declined', 'nr_let_expire']]
    assert np.all(scores['nr_sent'] == scores['nr_accepted'] + scores['nr_declined'] + scores['nr_let_expire'])

    accepted_offers = events_df[events_df['event'] == Event.member_accepted][['member_id','offer_id']]
    accepted_offers = accepted_offers.set_index('offer_id')
    accepted_offers_per_member = accepted_offers.groupby('member_id').groups

    total_utility_per_member = {member_id: np.sum(calc_utility(utility_df, member_id, accepted_offers)) for member_id, accepted_offers in accepted_offers_per_member.items()}
    total_utility_per_member = pd.DataFrame.from_dict(total_utility_per_member, orient='index', columns=['total_utility']).reset_index().rename(columns={'index':'member_id'})
    # avg_worth_accepted_coupon = total_util / nr_accepted

    # Get a list of coupon-utilities per member
    sent_offers = events_df[events_df['event'] == Event.coupon_sent][['member_id','offer_id']]
    sent_offers = sent_offers.set_index('offer_id')
    sent_offers_per_member = sent_offers.groupby('member_id').groups
    received_utility_per_member = {member_id: calc_utility(utility_df, member_id, received_offers) for member_id, received_offers in sent_offers_per_member.items()}

    # Summarize the list of coupon-utilities into one measure (per member)
    avg_worth_sent_coupon    = {member_id: np.average(list_of_received_utilities) for member_id, list_of_received_utilities in received_utility_per_member.items()}
    median_worth_sent_coupon = {member_id: np.median(list_of_received_utilities) for member_id, list_of_received_utilities in received_utility_per_member.items()}
    min_worth_sent_coupon    = {member_id: np.min(list_of_received_utilities) for member_id, list_of_received_utilities in received_utility_per_member.items()}
    max_worth_sent_coupon    = {member_id: np.max(list_of_received_utilities) for member_id, list_of_received_utilities in received_utility_per_member.items()}

    avg_worth_sent_coupon    = pd.DataFrame.from_dict(avg_worth_sent_coupon, orient='index', columns=['avg_worth_sent_coupon']).reset_index().rename(columns={'index':'member_id'})
    median_worth_sent_coupon = pd.DataFrame.from_dict(median_worth_sent_coupon, orient='index', columns=['median_worth_sent_coupon']).reset_index().rename(columns={'index':'member_id'})
    min_worth_sent_coupon    = pd.DataFrame.from_dict(min_worth_sent_coupon, orient='index', columns=['min_worth_sent_coupon']).reset_index().rename(columns={'index':'member_id'})
    max_worth_sent_coupon    = pd.DataFrame.from_dict(max_worth_sent_coupon, orient='index', columns=['max_worth_sent_coupon']).reset_index().rename(columns={'index':'member_id'})

    # Merge the tables into one
    tables_to_join = [avg_worth_sent_coupon, median_worth_sent_coupon, min_worth_sent_coupon, max_worth_sent_coupon]
    summary_worth_sent_coupons = join_tables(tables_to_join, ['member_id'])

    # Merge all scores into one df
    scores = scores.merge(total_utility_per_member, on='member_id', how='left').fillna(0)
    scores = scores.merge(summary_worth_sent_coupons, on='member_id', how='left').fillna(0)

    # Merge remaining member_ids and put to zero
    member_ids = pd.DataFrame(member_ids.values, columns=['member_id'])
    scores = pd.merge(member_ids, scores, 'left').fillna(0)

    return scores.convert_dtypes()


def calc_utility(utility_df, member_id, accepted_offers):
    utility_scores = list(map(lambda offer_id: utility_df.loc[member_id, offer_id], accepted_offers))
    return utility_scores


def perform_timeline_sense_check(events_df, issues):
    # print("")
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
    simulation_results = join_tables(list_of_tables, ['issue_id'], issues[['issue_id']]).fillna(0)
    simulation_results = simulation_results.sort_values(by='issue_id').reset_index(drop=True)
    # print(simulation_results)

    to_compare = simulation_results.merge(issues.add_suffix('_baseline').rename(columns={'issue_id_baseline':'issue_id'}), on='issue_id')

    # print(to_compare)
    if np.any(to_compare['nr_first_available'] > 0):
        assert np.all(to_compare['amount_baseline'] == to_compare['nr_first_available'])
    diff = (to_compare['nr_accepted'] - to_compare['nr_accepted_baseline']).abs()
    diff
    # sys.exit()
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
