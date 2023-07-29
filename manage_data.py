# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 22:47:23 2023

@author: Marco
"""

import os
import re
import json
import pickle

EXPORT_FOLDER = './timelines/'

def main():
    update_overview_json()

    source_run_nr = 23
    dest_run_nr = 63
    last_sim_nr = None
    folder_nr = None
    # rename_data_run_and_sim_nrs(source_run_nr, dest_run_nr, last_sim_nr, folder_nr)


def update_overview_json():
    overview = {}
    for run_nr in range(200):
        if os.path.exists(os.path.join(EXPORT_FOLDER, "run_%d"%run_nr)):
            print(run_nr)

            f = open(os.path.join(EXPORT_FOLDER, "run_%d"%run_nr, "%d_info.json"%run_nr))
            run_info = json.load(f)
            f.close()

            try:
                with open(os.path.join(EXPORT_FOLDER, "run_%d"%run_nr, '%d_run_name.pkl'%run_nr), 'rb') as fp:
                    run_name = pickle.load(fp)
                with open(os.path.join(EXPORT_FOLDER, "run_%d"%run_nr, '%d_run_data.pkl'%run_nr), 'rb') as fp:
                    run_data = pickle.load(fp)
                computed = True

                run_info['run_name'] = run_name
                member_stats_all_sims, _, _ = run_data
                nr_simulations = len(member_stats_all_sims['tot_worth_accepted_coupon'].columns)

            except:
                computed = False
                nr_simulations = 0


            run_info['computed'] = computed
            run_info['nr_sims_computed'] = nr_simulations
            run_info['completely_computed'] = nr_simulations == 50 or nr_simulations == 100

            overview[run_nr] = run_info

    with open(os.path.join(EXPORT_FOLDER, 'overview.json'), 'w') as fp:
        json.dump(overview, fp)


def rename_data_run_and_sim_nrs(source_run_nr, dest_run_nr, last_sim_nr=None, folder_nr=None):
    if folder_nr is None:
        folder_nr = source_run_nr

    contents = os.listdir(os.path.join(EXPORT_FOLDER, "run_%d"%folder_nr))
    timeline_files = list(filter(lambda name: re.search("^%d.[0-9]+_events_df.pkl"%source_run_nr, name), contents))
    timeline_nrs = sorted(list(map(lambda name: int(name.split('_')[0].split('.')[-1]), timeline_files)))

    if last_sim_nr is None:
        last_sim_nr = len(timeline_nrs) - 1
    
    for i, sim_nr in enumerate(timeline_nrs[::-1]):
        print("\r%d"%sim_nr, "\t\t", end="\t")
        source = os.path.join(EXPORT_FOLDER, "run_%d"%folder_nr, "%d.%d_events_df.pkl"%(source_run_nr, sim_nr))
        dest = os.path.join(EXPORT_FOLDER, "run_%d"%folder_nr, '%d.%d_events_df.pkl'%(dest_run_nr, last_sim_nr - i))
        os.rename(source, dest)
        source = os.path.join(EXPORT_FOLDER, "run_%d"%folder_nr, "%d.%d_sim_info.json"%(source_run_nr, sim_nr))
        dest = os.path.join(EXPORT_FOLDER, "run_%d"%folder_nr,'%d.%d_sim_info.json'%(dest_run_nr, last_sim_nr - i))
        os.rename(source, dest)



if __name__ == '__main__':
    main()