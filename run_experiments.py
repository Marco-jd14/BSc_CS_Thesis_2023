# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 15:23:26 2023

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
plt.style.use("seaborn")

# Local imports
import database.connect_db as connect_db
import database.query_db as query_db
from database.query_db import Event
from IssueStreamSimulator import IssueStreamSimulator
from allocator_algorithms import greedy, max_sum_utility, maximin_utility

# Set seed if results have to be repeatable
# np.random.seed(0)


# Different supported allocation procedures with name
allocation_procedures_mapping = {'greedy':   greedy,
                                 'max_sum':  max_sum_utility,
                                 'maximin':  maximin_utility,
                                 }
# Historical context types definition
class Historical_Context_Type(enum.Enum):
    full_historical     = 0
    partial_historical  = 1
    time_discounted     = 2


EXPORT_FOLDER = './timelines/'


def main():
    # Retrieve and prepare the required data for the simulator
    USE_DB = True
    if USE_DB:
        # Make a connection to the database
        conn = connect_db.establish_host_connection()
        db   = connect_db.establish_database_connection(conn)
        print("Successfully connected to database '%s'"%str(db.engine).split("/")[-1][:-1])

        # Retrieve data and close connection
        utility_df, P_accept_df, coupon_prep, member_prep = prepare_simulation_data(db)
        db.close()
    else:
        # Not using a database, simply retrieve data from locally saved files
        utility_df, P_accept_df, coupon_prep, member_prep = prepare_simulation_data(None)

    # Initialize the simulator
    simulator = IssueStreamSimulator()
    simulator.set_resource_data(utility_df, *coupon_prep)
    simulator.set_agent_data(P_accept_df, *member_prep)

    # Set the number of simulations
    NR_SIMULATIONS = 1
    simulator.set_simulation_properties(NR_SIMULATIONS, EXPORT_FOLDER)

    # Define the types of allocation policies to simulate
    ALLOCATOR_PROCEDURES     = ['greedy']
    HISTORICAL_CONTEXT_TYPES = [Historical_Context_Type.full_historical, Historical_Context_Type.partial_historical]
    MINIMUM_BATCH_SIZES      = [1, 50]

    # Do all specified types of allocation policies
    for alloc_proc_name in ALLOCATOR_PROCEDURES:
        for min_batch_size in MINIMUM_BATCH_SIZES:
            for historical_context_type in HISTORICAL_CONTEXT_TYPES:

                # Set the new properties and start simulations
                allocation_procedure = {alloc_proc_name: allocation_procedures_mapping[alloc_proc_name]}
                simulator.set_alloc_policy_properties(min_batch_size, allocation_procedure, historical_context_type)
                simulator.start(verbose=False)



def prepare_simulation_data(db=None):
    if db is not None:
        # Retrieve from database
        result = query_db.retrieve_from_sql_db(db, 'filtered_coupons', 'filtered_issues', 'filtered_offers')
        filtered_coupons, filtered_issues, filtered_offers = result
        result = query_db.retrieve_from_sql_db(db, 'member', 'member_category', 'member_family_member')
        all_members, all_member_categories, all_family_members = result

        # By dumping the required data from the database to a local file, the need for the database is removed.
        # This enables the simulation to run on other machines without access to the database, if all of these local files are provided
        DUMP_TO_PICKLE = False
        if DUMP_TO_PICKLE:
            filtered_coupons.     to_pickle('./database/data/filtered_coupons.pkl')
            filtered_issues.      to_pickle('./database/data/filtered_issues.pkl')
            filtered_offers.      to_pickle('./database/data/filtered_offers.pkl')
            all_members.          to_pickle('./database/data/all_members.pkl')
            all_member_categories.to_pickle('./database/data/all_member_categories.pkl')
            all_family_members.   to_pickle('./database/data/all_family_members.pkl')

    else:
        # Retrieve from local files
        with open('./database/data/filtered_coupons.pkl', 'rb') as fp:      filtered_coupons        = pickle.load(fp)
        with open('./database/data/filtered_issues.pkl', 'rb') as fp:       filtered_issues         = pickle.load(fp)
        with open('./database/data/filtered_offers.pkl', 'rb') as fp:       filtered_offers         = pickle.load(fp)
        with open('./database/data/all_members.pkl', 'rb') as fp:           all_members             = pickle.load(fp)
        with open('./database/data/all_member_categories.pkl', 'rb') as fp: all_member_categories   = pickle.load(fp)
        with open('./database/data/all_family_members.pkl', 'rb') as fp:    all_family_members      = pickle.load(fp)

    # No functionality to incorporate 'aborted_at' has been made (so far)
    assert np.all(filtered_issues['aborted_at'].isna()), "The simulation can not yet take into account when an issue is aborted. Please drop the issue and corresponding coupons."
    filtered_issues = filtered_issues.sort_values(by='sent_at')

    # No functionality to incorporate 'reissue_unused_participants_of_coupons_enabled' has been made (so far)
    # print("%d out of %d offers enabled reissue"%(np.sum(filtered_offers['reissue_unused_participants_of_coupons_enabled']), len(filtered_offers)))

    relevant_offer_columns = ['id', 'category_id', 'accept_time', 'member_criteria_gender', 'member_criteria_min_age',
                              'member_criteria_max_age', 'family_criteria_min_count', 'family_criteria_max_count',
                              'family_criteria_child_age_range_min', 'family_criteria_child_age_range_max',
                              'family_criteria_child_stages_child_stages', 'family_criteria_child_gender',
                              'family_criteria_is_single', 'family_criteria_has_children']
    filtered_offers = copy.copy(filtered_offers[relevant_offer_columns])

    # TODO: low-priority --> use the following columns when determining eligible members:
        # 'receive_coupon_after', 'deactivated_at', 'archived_at', 'onboarded_at', 'created_at'
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
    decline_times = valid_decline_times.values.reshape(-1)


    # Optionally make two nice plots
    PLOT_DECLINE_TIMES = False
    if PLOT_DECLINE_TIMES:
        plt.figure(figsize=(8,6))
        plt.hist(valid_decline_times, bins=80, density=False, edgecolor='k')
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.ylabel("frequency", fontsize=15)
        plt.xlabel("fraction of given time to respond", fontsize=15)
        # plt.savefig("./plots/decline_time_histogram.png", bbox_inches='tight', pad_inches=0.05, dpi=150)
        plt.show()

        plt.figure(figsize=(8,5.2))
        quantiles = [0,1,10,25,50,50,75,90,99,100]
        data_quantiles = np.percentile(valid_decline_times, np.sort(np.unique(quantiles)))
        plt.plot(data_quantiles, 'ko-')
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.ylabel("fraction of given time to respond", fontsize=15)
        plt.xlabel("\ndata quantile", fontsize=15)

        def nr_to_label(nr):
            s = str(nr)+"%"
            if nr == 0:      s += "\n(min.)"
            elif nr == 50:   s += "\n(median)"
            elif nr == 100:  s += "\n(max.)"
            return s

        plt.xticks([0,1,2,3,4,5,6,7,8], list(map(nr_to_label, np.sort(np.unique(quantiles)))))
        # plt.savefig("./plots/decline_time_quantiles.png", bbox_inches='tight', pad_inches=0.05, dpi=150)
        plt.show()


    # Set to False when the accept probabilities have already been computed by a Recommender System
    CALCULATE_ACCEPT_PROBABILITIES = True
    if CALCULATE_ACCEPT_PROBABILITIES:
        P_accept_df = calc_accept_probabilities(all_members['id'], filtered_offers['id'], filtered_coupons)
    else: 
        P_accept_df = pd.read_pickle(os.path.join(EXPORT_FOLDER, "utility_df.pkl"))

    # Set the utility of a member receiving a coupon, equal to the probability of that member accepting the coupon
    utility_df = copy.copy(P_accept_df)

    # Group some variables in a tuple and return
    coupon_prep = (filtered_issues, filtered_offers)
    member_prep = (all_members, all_member_categories, all_family_members, P_let_expire_given_not_accepted, decline_times)
    return utility_df, P_accept_df, coupon_prep, member_prep



def calc_accept_probabilities(member_ids, offer_ids, filtered_coupons):

    # Calculate per member how many coupons have been accepted or not, and per offer how many coupons have been accepted or not
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
    P_accept_matrix = P_accept_members.values.reshape(-1,1) @ P_accept_offers.values.reshape(1,-1)
    P_accept_df = pd.DataFrame(P_accept_matrix, columns=P_accept_offers.index, index=P_accept_members.index)

    # Calculate average probability of accepting
    accepted_coupons = filtered_coupons[filtered_coupons['member_response'] == Event.member_accepted]
    declined_coupons = filtered_coupons[filtered_coupons['member_response'] == Event.member_declined]
    let_expire_coupons = filtered_coupons[filtered_coupons['member_response'] == Event.member_let_expire]
    avg_P_accept = len(accepted_coupons) / (len(accepted_coupons) + len(declined_coupons) + len(let_expire_coupons))

    # Scale all probabilities such that the average accept probability 
    ratio = np.log(avg_P_accept) / np.log(np.average(P_accept_df.values))
    P_accept_df = np.power(P_accept_df, ratio)

    # Check if all probabilities indeed lie between 0 and 1
    assert np.all(P_accept_df.values < 1) and np.all(P_accept_df.values > 0)

    # Optionally make a nice plot
    PLOT_P_ACCEPT_DISTRIBUTION = False
    if PLOT_P_ACCEPT_DISTRIBUTION:
        print("Average P_accept is %.3f"%np.mean(P_accept_df.values))
        print("Percentage P_accept > 0.5 is %.1f"%(100*np.sum(P_accept_df.values.reshape(-1)>0.5) / len(P_accept_df.values.reshape(-1))))

        plt.figure()
        plt.hist(P_accept_df.values.reshape(-1), bins=50, density=False, edgecolor='k')
        plt.xlabel("P_accept", fontsize=15)
        plt.ylabel("frequency", fontsize=15)
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=15)
        plt.title("Distribution of P_accept")
        # plt.savefig("./plots/P_accept_dist", bbox_inches='tight', pad_inches=0.05, dpi=150)
        plt.show()

    return P_accept_df

if __name__ == '__main__':
    main()
