# -*- coding: utf-8 -*-
"""
Created on Thu May 18 18:19:02 2023

@author: Marco
"""


import sys
import copy
import enum
import numpy as np
import pandas as pd
import datetime as dt
from pprint import pprint
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
from database.lib.tracktime import TrackTime, TrackReport

import database.connect_db as connect_db
import database.query_db as query_db
Event = query_db.Event

COMMUNITY_ID = 1 # Tilburg

def main():
    TrackTime("Connect to db")
    conn = connect_db.establish_host_connection()
    db   = connect_db.establish_database_connection(conn)
    print("Successfully connected to database '%s'"%str(db.engine).split("/")[-1][:-1])

    TrackTime("Retrieve from db")
    result = query_db.retrieve_from_sql_db(db, 'filtered_coupons', 'filtered_issues', 'filtered_offers')
    filtered_coupons, filtered_issues, filtered_offers = result

    # print_table_info(db)


    query = "select * from member_category"
    all_member_categories = pd.read_sql_query(query, db)
    query = "select * from member_family_member where type='child'"
    all_children = pd.read_sql_query(query, db)

    eligible_members_basic = get_eligible_members_basic(db, verbose=False)

    # query = "select * from member"
    # all_members = pd.read_sql_query(query, db)
    # plot_timeline_active_coupons(all_members,'onboarded_at')
    # plot_timeline_active_coupons(all_members,'receive_coupon_after')

    print("Nr coupons to handle:",len(filtered_coupons))

    for i, coupon in filtered_coupons.iterrows():
        TrackTime("print")
        if i%100 == 0:
            print("\rHandling coupon nr %d (%.1f%%)"%(i,100*i/len(filtered_coupons)), end='')

        # print(eligible_members_basic.shape)
        TrackTime("get matching_context")
        matching_context = filtered_offers[filtered_offers['id'] == coupon['offer_id']].squeeze()
        timestamp = coupon['created_at']

        eligible_members = eligible_members_basic
        eligible_members = get_eligible_members_static(        db, eligible_members, matching_context, all_member_categories, verbose=False)
        eligible_members = get_eligible_members_time_dependent(db, eligible_members, matching_context, timestamp, all_children, verbose=False)
        # TrackTime('copy')
        # eligible_members = copy.copy(eligible_members)
        # print(eligible_members.shape)
        # print("nr eligible_members:", len(eligible_members))
        # TODO: eligibility based on coupon history

        if i > 0:
            break

    db.close()

    TrackReport()


def plot_timeline_active_coupons(df, col_name):
    timestamps = df[col_name]
    timestamps = timestamps.sort_values(ascending=True)
    assert timestamps.is_monotonic

    start_date = timestamps.iloc[0] - dt.timedelta(seconds=1)
    end_date   = timestamps.iloc[-1] + dt.timedelta(seconds=1)
    delta = {'days':1}
    intervals = datetime_range(start_date, end_date, delta)

    res = timestamps.groupby(pd.cut(timestamps, intervals)).count()
    res.name = "nr_coupons_per_interval"
    print(res)
    res = res.reset_index()

    interval_ends = list(map(lambda interval: interval.right, res[col_name]))
    plt.plot(interval_ends, res.nr_coupons_per_interval, '-o')
    plt.xticks(fontsize=8)

def datetime_range(start_date, end_date, delta):
    result = []
    nxt = start_date
    delta = relativedelta(**delta)

    while nxt <= end_date:
        result.append(nxt)
        nxt += delta

    result.append(end_date)
    return result


def child_age_to_stage(age):
    assert age >= 0
    for stage, age_range in child_stages_def.items():
        if age_range[0] <= age and age <= age_range[1]:
            return stage
    raise ValueError("Age not part of a child stage")

child_stages_def = {
    'baby':         [0,1],
    'dreumes':      [1,2],
    'peuter':       [2,4],
    'kleuter':      [4,6],
    'schoolkind':   [6,11],
    'puber':        [11,18],
    'volwassene':   [18,200],
    }

def calculate_age_at(date_born, date_at):
    return date_at.year - date_born.year - ((date_at.month, date_at.day) < (date_born.month, date_born.day))


def get_eligible_members_time_dependent(db, eligible_members, matching_context, timestamp, all_children=None, verbose=False):
    # TODO: inactivated_at, active, onboarded_at, receive_coupons_after, 'created_at'

    TrackTime("calc age")
    # Calculate age of members
    min_age = matching_context['member_criteria_min_age']
    max_age = matching_context['member_criteria_max_age']
    if not pd.isna(min_age) or not pd.isna(max_age):
        members_age = eligible_members['date_of_birth'].apply(lambda born: calculate_age_at(born, timestamp.date()))

    TrackTime("age criteria")
    # Minimum and maximum age criteria
    if not pd.isna(min_age) and not pd.isna(max_age):
        age_in_range = np.logical_and((members_age >= min_age).values, (members_age <= max_age).values)
        eligible_members = eligible_members[age_in_range]
    elif not pd.isna(min_age):
        eligible_members = eligible_members[(members_age >= min_age).values]
    elif not pd.isna(max_age):
        eligible_members = eligible_members[(members_age <= max_age).values]
    if verbose: print("nr eligible_members age range:", len(eligible_members))

    TrackTime("setting up family criteria")
    # Family criteria
    fam_min_count       = matching_context['family_criteria_min_count']
    fam_max_count       = matching_context['family_criteria_max_count']
    fam_has_children    = matching_context['family_criteria_has_children']
    fam_child_min_age   = matching_context['family_criteria_child_age_range_min']
    fam_child_max_age   = matching_context['family_criteria_child_age_range_max']
    fam_child_stages    = matching_context['family_criteria_child_stages_child_stages']
    fam_child_gender    = matching_context['family_criteria_child_gender']

    fam_child_stages = None if fam_child_stages.strip("[] ") == "" else fam_child_stages
    child_criteria = [fam_child_min_age, fam_child_max_age, fam_child_stages, fam_child_gender]

    # If none of the remaining criteria are filled, immediately return the current set of eligible members
    if np.all(list(map(pd.isna, [fam_min_count, fam_max_count, fam_has_children] + child_criteria ))):
        return eligible_members

    # Check whether the criteria make sense
    if not np.all(list(map(pd.isna, child_criteria))):
        assert not pd.isna(fam_has_children) and fam_has_children==1, "Criteria on a child, but not that the member is required to have a child?"

    if not pd.isna(fam_has_children) and fam_has_children==0:
        assert np.all(list(map(pd.isna, child_criteria))), "Member is not allowed to have children, but there are other criteria on the children?"

    TrackTime("Retrieve children")
    # Filter on existing children (at the time of 'timestamp')
    if all_children is None:
        query = "select * from member_family_member where type='child'"
        all_children = pd.read_sql_query(query, db)

    TrackTime("calc children age")
    rel_children = copy.copy(all_children[all_children['user_id'].isin(eligible_members['id'])])
    rel_children['age'] = rel_children['date_of_birth'].apply(lambda born: calculate_age_at(born, timestamp.date()))
    rel_children = rel_children[rel_children['age'] >= 0]

    TrackTime("calc children count")
    # Count number of children with age >= 0
    children_counts = rel_children.groupby('user_id').aggregate(children_count=('id','count')).reset_index().rename(columns={'user_id':'id'})
    # Merge children_count column into members table, and put members without chilldren on zero count
    eligible_members = pd.merge(eligible_members, children_counts, how='left', on='id')
    eligible_members['children_count'] = eligible_members['children_count'].fillna(0).astype(int)

    TrackTime("calc family_count")
    # Calculate family count
    if not pd.isna(fam_min_count) or not pd.isna(fam_max_count):
        query = "select * from member_family_member where type='partner'"
        all_partners = pd.read_sql_query(query, db)
        rel_partners = all_partners[all_partners['user_id'].isin(eligible_members['id'])]
        partner_counts = rel_partners.groupby('user_id').aggregate(partner_count=('id','count')).reset_index().rename(columns={'user_id':'id'})
        # Merge partner_count column into members table, and put members without partner on zero count
        eligible_members = pd.merge(eligible_members, partner_counts, how='left', on='id')
        eligible_members['partner_count'] = eligible_members['partner_count'].fillna(0).astype(int)
        eligible_members['family_count'] = eligible_members['children_count'] + eligible_members['partner_count']

    TrackTime("family_count criteria")
    # Family count criteria
    if not pd.isna(fam_min_count):
        eligible_members = eligible_members[eligible_members['family_count'] >= fam_min_count]
    if not pd.isna(fam_max_count):
        eligible_members = eligible_members[eligible_members['family_count'] <= fam_max_count]
    if verbose: print("nr eligible_members fam_count range:", len(eligible_members))

    TrackTime("has_children criteria")
    # Has children criteria
    if not pd.isna(fam_has_children):
        if fam_has_children:
            eligible_members = eligible_members[eligible_members['children_count'] > 0]
        else:
            eligible_members = eligible_members[eligible_members['children_count'] == 0]
            # Do not further check children criteria, since members are not allowed to have children
            return eligible_members
    if verbose: print("nr eligible_members has_children:", len(eligible_members))

    TrackTime("other children criteria")
    # Children criteria
    if not pd.isna(fam_child_min_age):
        rel_children = rel_children[rel_children['age'] >= fam_child_min_age]
    if not pd.isna(fam_child_max_age):
        rel_children = rel_children[rel_children['age'] <= fam_child_max_age]
    if not pd.isna(fam_child_gender):
        rel_children = rel_children[rel_children['gender'] == fam_child_gender]
    if not pd.isna(fam_child_stages):
        rel_children['stage'] = rel_children['age'].apply(child_age_to_stage)
        accepted_child_stages = list(map(lambda string: string.strip('"'), fam_child_stages.strip("[]").split(",")))
        rel_children = rel_children[rel_children['stage'].isin(accepted_child_stages)]

    # Filter members with children that fit within the children criteria
    eligible_members = eligible_members[eligible_members['id'].isin(rel_children['user_id'])]
    if verbose: print("nr eligible_members children criteria:", len(eligible_members))

    return eligible_members


def get_eligible_members_static(db, eligible_members, matching_context, all_member_categories=None, verbose=False):
    """ Function that applies 'static' criteria.
    'static' in the sense that the result of the criteria should not change
    over (a short) time. In this case, because I only received a snapshot of the
    database at one point in time, these properties can physically not change.
    Things like: community, subscribed categories, gender, partners
    """

    TrackTime("gender criteria")
    # Gender criteria
    required_gender = matching_context['member_criteria_gender']
    if not pd.isna(required_gender):
        eligible_members = eligible_members[eligible_members['gender'] == required_gender]
    if verbose: print("nr eligible_members gender:", len(eligible_members))

    TrackTime("is_single criteria")
    # Single or has partner criteria
    has_to_be_single = matching_context['family_criteria_is_single']
    if not pd.isna(has_to_be_single):
        query = "select * from member_family_member where type='partner'"
        all_partners = pd.read_sql_query(query, db)

        if has_to_be_single:
            eligible_members = eligible_members[~eligible_members['id'].isin(all_partners['user_id'])]
        else:
            eligible_members = eligible_members[eligible_members['id'].isin(all_partners['user_id'])]
    if verbose: print("nr eligible_members has_to_be_single:", len(eligible_members))

    TrackTime("category_id criteria")
    # Member must be subscribed to category
    coupon_category_id = matching_context['category_id']
    if not pd.isna(coupon_category_id):
        TrackTime("Retrieve category table")
        if all_member_categories is None:
            query = "select * from member_category where category_id=%d"%coupon_category_id
            members_subscribed_to_coupon_cat = pd.read_sql_query(query, db)
        else:
            members_subscribed_to_coupon_cat = all_member_categories[all_member_categories['category_id'] == coupon_category_id]

        TrackTime("category_id criteria")
        eligible_members = eligible_members[eligible_members['id'].isin(members_subscribed_to_coupon_cat['member_id'])]
    if verbose: print("nr eligible_members category:", len(eligible_members))

    return eligible_members


def get_eligible_members_basic(db, verbose=False):
    TrackTime("Retrieve member table")
    query = "select * from member"
    all_members = pd.read_sql_query(query, db)
    if verbose: print("nr all_members:", len(all_members))

    TrackTime("eligible basic")
    # email and phone number criteria
    members_with_email    = all_members[~all_members['email'].isna()]
    members_with_phone_nr = members_with_email[~members_with_email['mobile'].isna()]

    # member must be active to be eligible
    active_members = members_with_phone_nr[members_with_phone_nr['active'] == 1]
    assert np.all(active_members['member_state'] == 'active')
    if verbose: print("nr active_members:", len(active_members))

    # Community criteria
    eligible_members = active_members[active_members['community_id'] == COMMUNITY_ID]

    return eligible_members


def print_table_info(db):
    # all_tables = pd.read_sql_query("show tables", db).squeeze().values
    # print(all_tables)

    # tables = ['filtered_coupons', 'filtered_issues', 'filtered_offers']

    tables = ['member', 'member_category', 'member_family_member', 'filtered_offers', 'category']
    for table_name in tables:

        print("\n\nTABLE", table_name)
        query = "SELECT * FROM %s"%table_name
        df = pd.read_sql_query(query, db)

        for col in df.columns:
            unique_values = pd.unique(df[col].values)
            if len(unique_values) <= 10:
                print("\t", col, type(unique_values[0]), len(unique_values), unique_values, unique_values[0])
            else:
                print("\t", col, type(unique_values[0]), len(unique_values), unique_values[0])

    # query = "select * from category"
    # category_summary = pd.read_sql_query(query, db)
    # print(category_summary[['id','title','enabled_at']])
    # print(len(category_summary))


if __name__ == '__main__':
    main()