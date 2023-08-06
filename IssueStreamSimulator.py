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
from collections import Counter
from database.lib.tracktime import TrackTime, TrackReport


TIME_DISCOUNT_RATIO = 0.1 ** (1/30)  # After 30 days, 10% (=0.1) of utility remains


CHILD_STAGES = {
    'baby':         [0,1],
    'dreumes':      [1,2],
    'peuter':       [2,4],
    'kleuter':      [4,6],
    'schoolkind':   [6,11],
    'puber':        [11,18],
    'volwassene':   [18,200],
    }


class IssueStreamSimulator:
    def __init__(self):
        global Event, Historical_Context_Type
        from database.query_db import Event
        from run_experiments import Historical_Context_Type


    def set_resource_data(self, utility_df, issues, offers):
        """ Function that saves data on issues and offers as attribute
        It splits up the utility_df into its values, columns, and index into separate attributes

        Parameters
        ----------
        utility_df : pd.DataFrame
            A dataframe with columns the offer-ids, and the index are the member-ids. Each combination of
            member x offer has a value, corresponding to the increase in utility of that member accepting a coupon from that offer
        issues : pd.DataFrame
            A list of issues over a certain timeline, which are to be allocated by the simulation
        offers : pd.DataFrame
            Each issue from 'filtered_issues' corresponds to an offer with a set of criteria, i.e. 'minimum_age >= 18'
            Also includes which category the offer is a part of
        """
        self.utility_df = copy.copy(utility_df)
        self.issues     = copy.copy(issues)
        self.offers     = copy.copy(offers)

        member_indices = np.arange(self.utility_df.shape[0])
        member_id_to_index = {self.utility_df.index[member_index]: member_index for member_index in member_indices}

        offer_indices = np.arange(self.utility_df.shape[1])
        offer_id_to_index = {self.utility_df.columns[offer_index]: offer_index for offer_index in offer_indices}

        utility_values = self.utility_df.values
        utility_indices = (member_id_to_index, offer_id_to_index)
        self.utility_values = copy.copy(utility_values)
        self.utility_indices = copy.copy(utility_indices)


    def set_agent_data(self, P_accept_df, members, member_categories, family_members,
                        P_let_expire_given_not_accepted, decline_times):
        """ Function that saves data on members as attribute. It splits up family members into children and partners

        Parameters
        ----------
        P_accept_df : pd.DataFrame
            A dataframe with columns the offer-ids, and the index are the member-ids. Each combination of
            member x offer has a value, corresponding to the probability of that member accepting a coupon from that offer
        members : pd.DataFrame
            Information mapping a member-id on date of birth, gender, active, etc.
        member_categories : pd.DataFrame
            A table mapping which members are 'subscribed' to which category
        family_members : pd.DataFrame
            A table mapping which members have registered a partner or child
        P_let_expire_given_not_accepted : float
            A single value, to be interpreted as the probability that a randomly selected member
            does not respond to an offer, given that we already know the member will not accept the offer
        decline_times : np.array, 1 dimensional
            A list of decline times, which expresses how long a member has taken before declining an offer
            as a percentage of the total given time to respond to the offer
        """
        self.members           = copy.copy(members)
        self.member_categories = copy.copy(member_categories)
        self.children          = copy.copy(family_members[family_members['type'] == 'child'].reset_index(drop=True))
        self.partners          = copy.copy(family_members[family_members['type'] == 'partner'].reset_index(drop=True))
        self.P_accept_df       = copy.copy(P_accept_df)
        self.P_let_expire_given_not_accepted = copy.copy(P_let_expire_given_not_accepted)
        self.decline_times     = copy.copy(decline_times)


    def set_alloc_policy_properties(self, min_batch_size=None, alloc_procedure=None,
                                    historical_context_type=None):
        """ This function allows tweaking the three properties which can vary between simulation
        These properties include:

        Parameters
        ----------
        min_batch_size : int
            The minimum number of coupons which have to available for sending out before the simulation can send out the next batch
        alloc_procedure : dict[str --> function]
            A dictionary with a single key, which is the name of the allocation procedure, and the function
            that is an offline allocation procedure, operating on a single batch of coupons. The only input to the function
            is a rectangular matrix of utilities
        historical_context_type : run_experiments.Historical_Context_Type
            A Historical_Context_Type enum.Enum instance, which declares how information about member's
            historically received coupons should be incorporated into the allocation policy
        """
        self.min_batch_size          = min_batch_size
        self.alloc_procedure_name    = list(alloc_procedure.keys())[0]
        self.alloc_procedure         = list(alloc_procedure.values())[0]
        self.historical_context_type = historical_context_type


    def set_simulation_properties(self, nr_simulations, export_folder):
        """ Function that sets how many simulations are to be run with the same setting,
        and where the results should be exported to
        """
        self.nr_simulations = nr_simulations

        export_folder = export_folder.replace("/", "\\")
        if not export_folder.endswith("\\"):
            export_folder += '\\'
        self.export_folder = export_folder


    def start(self, extra_run_info={}, verbose=False):
        """ After having called the functions 'set_resource_data', 'set_agent_data',
        'set_alloc_policy_properties' and 'set_simulation_properties', this function starts the simulation
        
        Parameters
        ----------
        extra_run_info : dict
            Gives the possibility to provide some more information on the type of simulation being run.
            The dictionary will be saved under the 'run_info.json' file
        """
        run_nr = self.get_next_available_run_nr()
        all_run_info = self.get_run_info(extra_run_info)

        run_export_folder = os.path.join(self.export_folder, "run_%d"%run_nr)
        os.makedirs(run_export_folder, exist_ok=True)
        # Export run info and utility values
        self.export_run_info(all_run_info, run_nr)

        print("\nStarting a total of %d simulation(s) with minimum batch size %d, '%s' allocation procedure"%(self.nr_simulations,self.min_batch_size,self.alloc_procedure_name) + \
              ", and '%s' way of incorporating members' historical context"%str(self.historical_context_type), end='')

        prev_start_time = None
        for sim_nr in range(self.nr_simulations):
            sim_start_time = dt.datetime.now().replace(microsecond=0)

            sim_info_str = "\nStarting simulation %d (run %d) at %s"%(sim_nr, run_nr, sim_start_time)
            sim_info_str += " (%s later)"%(str(sim_start_time - prev_start_time).split('.')[0] ) if prev_start_time else ""
            print(sim_info_str)

            events_df = self.simulate_once(verbose)
            events_df.to_pickle(os.path.join(run_export_folder, '%d.%d_events_df.pkl'%(run_nr, sim_nr)))

            sim_end_time = dt.datetime.now().replace(microsecond=0)
            sim_info = {'runtime':str(sim_end_time - sim_start_time).split('.')[0],
                        'start': str(sim_start_time), 'end': str(sim_end_time)}
            with open(os.path.join(run_export_folder, '%d.%d_sim_info.json'%(run_nr, sim_nr)), 'w') as fp:
                json.dump(sim_info, fp)

            prev_start_time = sim_start_time

        sim_info_str = "\nFinished all simulations at %s"%(dt.datetime.now().replace(microsecond=0))
        sim_info_str += " (%s later)"%(str(dt.datetime.now() - prev_start_time).split('.')[0] ) if prev_start_time else ""
        print(sim_info_str)

        print("")
        TrackReport()
        print("\n")


    def print_progress(self, issue_counter, batch_counter):
        print("\rissue nr %d (%.1f%%)     batch nr %d"%(issue_counter,100*issue_counter/len(self.issues), batch_counter), end='')


    def simulate_once(self, verbose=False):
        """ Function that performs a single simulation

        Returns
        -------
        events_df : pd.DataFrame
            A list of events that have happened in the simulation, sorted chronologically.
            Columns include ['event','timestamp','coupon_id','coupon_follow_id','issue_id','offer_id','member_id','batch_id']
        """
        self.init_new_sim()

        prev_batch_unsent_coupons = []
        prev_batch_day_int = None

        # Loop over all issues to release
        batch_counter = 0
        for issue_counter, (issue_id, issue) in enumerate(self.issues.iterrows()):
            TrackTime("Releasing new issue")
            if issue_counter%20 == 0:
                self.print_progress(issue_counter, batch_counter)

            # Release the new issue
            add_to_queue, came_available_coupons = self.release_new_issue(issue)
            # Add the new issue to the queue and events list
            self.unsorted_queue_of_coupons.extend(add_to_queue)
            self.events_list.extend(came_available_coupons)

            # Send out the next batch while we have enough coupons
            while self.is_batch_ready_to_be_sent(issue_counter):
                TrackTime("Sending out batch")
                batch_counter += 1
                if batch_counter%10 == 0:
                    self.print_progress(issue_counter, batch_counter)

                # Generate events for the next batch
                result = self.send_out_new_batch(batch_counter, prev_batch_unsent_coupons, prev_batch_day_int, verbose)
                prev_batch_unsent_coupons, prev_batch_day_int = result


        # Send out last batch
        if len(self.unsorted_queue_of_coupons) > 0:
            self.send_out_new_batch(batch_counter+1, prev_batch_unsent_coupons, prev_batch_day_int, verbose)

        # Turn the events_list into a dataframe and sort it
        TrackTime("Make events df")
        events_df = pd.DataFrame(self.events_list)[self.events_df.columns]
        events_df['event_enum_nr'] = events_df['event'].apply(lambda event: event.value)
        events_df['timestamp'] = events_df['timestamp'].apply(lambda time: time.replace(microsecond=0))
        events_df = events_df.sort_values(by=['timestamp','coupon_follow_id','event_enum_nr']).reset_index(drop=True)
        events_df = events_df.drop(columns=['event_enum_nr'])
        self.events_df = copy.copy(events_df)
        return events_df


    def send_out_new_batch(self, batch_counter, prev_batch_unsent_coupons, prev_batch_day_int, verbose):
        """ Function that handles a batch:
            - Extracts batch of coupons from the queue
            - Filters the full matrix of utilities to the batch coupons and their eligible members
            - Decides upon an allocation of the batch: which coupon is sent to which eligible member
            - Simulate the member responses to the offered coupons
            - Add not-accepted coupons back to the queue of coupons for allocation

        Parameters
        ----------
        batch_counter : int
        prev_batch_unsent_coupons : list[dict]
        prev_batch_day_int : int

        Returns
        -------
        unsent_coupons : list[dict]
        batch_day_int : int
        """
        # Determine which coupons are to be allocated in the current batch
        batch_to_send, batch_sent_at = self.extract_batch_from_queue(verbose)

        # Determine whether this batch is sent out on a new day (i.e. prev batch before midnight, current batch after midnight)
        batch_day_int = int((batch_sent_at - self.first_issue_sent_at).total_seconds() / 60 / 60 / 24)
        days_passed_since_prev_batch = 0 if prev_batch_day_int is None else (batch_day_int - prev_batch_day_int)

        # Check if we can try to resend coupons, or if they have already expired
        unsent_coupons_to_retry, unsent_coupons_now_expired = self.check_expiry_unsent_coupons(batch_sent_at, prev_batch_unsent_coupons, verbose)
        # Add expired coupons to events_list
        self.events_list.extend(unsent_coupons_now_expired)
        # Add non-expired coupons that were not allocated last batch to this batch
        batch_to_send = unsent_coupons_to_retry + batch_to_send

        # Prepare the matrix of utilities for the allocation-algorithm
        utility_to_allocate, actual_utility, batch_indices = self.filter_relevant_utilities(batch_to_send, batch_sent_at)

        # Zero eligible members
        if utility_to_allocate.shape[0] == 0:
            unsent_coupons = batch_to_send
            if verbose: print("\nCould not find any eligible members to send the batch to")
            return unsent_coupons, batch_day_int

        X_a_r = self.determine_coupon_allocation(utility_to_allocate, verbose)
        allocated_member_coupon_pairs = np.nonzero(X_a_r)
        assert X_a_r.shape == (len(batch_indices[0]), len(batch_indices[1]))
        assert len(set(allocated_member_coupon_pairs[0])) == len(allocated_member_coupon_pairs[0]), "One member got more than one coupon?"

        TrackTime("Send out batch & simulate")

        # Send out the coupons that were allocated to the right members
        unsent_coupons, sent_coupons = self.send_out_chosen_coupons(allocated_member_coupon_pairs, batch_indices,
                                                                    batch_to_send, batch_sent_at, batch_counter)


        # Simulate the member responses of those coupons that were sent out
        accepted_coupons, not_accepted_coupons = self.simulate_member_responses(allocated_member_coupon_pairs, batch_indices,
                                                                                batch_to_send, batch_sent_at, verbose)


        # Check if not-accepted coupons can be re-allocated, or have expired
        came_available_coupons, expired_coupons, coupons_for_re_release_to_queue = self.re_release_non_accepted_coupons(not_accepted_coupons)

        if verbose:
            print("%d coupons not sent out"%len(unsent_coupons), end='\t')
            print("%d coupons sent out"%len(sent_coupons), end='\t')
            print("%d coupons accepted"%len(accepted_coupons), end='\t')
            print("%d coupons not accepted"%len(not_accepted_coupons), end='\t')
            print("%d coupons came newly available"%len(came_available_coupons), end='\t')
            print("%d coupons expired"%len(expired_coupons))

        # Add some coupons back to the queue
        self.unsorted_queue_of_coupons.extend(coupons_for_re_release_to_queue)

        # Add all the events to the events_list
        self.events_list.extend(sent_coupons)
        self.events_list.extend(accepted_coupons)
        self.events_list.extend(not_accepted_coupons)
        self.events_list.extend(came_available_coupons)
        self.events_list.extend(expired_coupons)

        assert len(batch_to_send) == len(sent_coupons) + len(unsent_coupons), "%d != %d (%d + %d)"%(len(batch_to_send), len(sent_coupons) + len(unsent_coupons), len(sent_coupons), len(unsent_coupons))
        assert len(sent_coupons) == len(accepted_coupons) + len(not_accepted_coupons), "%d != %d (%d + %d)"%(len(sent_coupons), len(accepted_coupons) + len(not_accepted_coupons), len(accepted_coupons), len(not_accepted_coupons))
        assert len(not_accepted_coupons) == len(came_available_coupons) + len(expired_coupons), "%d != %d (%d + %d)"%(len(not_accepted_coupons), len(came_available_coupons) + len(expired_coupons), len(came_available_coupons), len(expired_coupons))

        TrackTime("Update historical context")
        self.historical_context.update_with_batch_results(sent_coupons, accepted_coupons, not_accepted_coupons, 
                                                          days_passed_since_prev_batch, self.utility_values, 
                                                          self.utility_indices, self.first_issue_sent_at)

        if len(unsent_coupons) == len(batch_to_send):
            print("\nDid not allocate any coupons from last batch")
        if verbose:
            if len(unsent_coupons) > 0:
                print("\nCould not allocate %d out of %d coupons"%(len(unsent_coupons),len(batch_to_send)))

        return unsent_coupons, batch_day_int



    def re_release_non_accepted_coupons(self, not_accepted_coupons):
        """ Determine whether the not-accepted coupons have now expired, or are ready to be added back to the queue

        Parameters
        ----------
        not_accepted_coupons : list[dict]

        Returns
        -------
        expired_coupons : list[dict]
            A list of dictionaries, with each dictionary representing an event with the keys
            ['event','timestamp','coupon_id','coupon_follow_id','issue_id','offer_id','member_id','batch_id']
        came_available_coupons : list[dict]
            A list of dictionaries, with each dictionary representing an event with the keys
            ['event','timestamp','coupon_id','coupon_follow_id','issue_id','offer_id','member_id','batch_id']
        add_to_queue : list[dict]
            A list of dictionaries, with each dictionary representing a coupon to allocate with the keys
            [        'timestamp','coupon_id','coupon_follow_id','issue_id','offer_id']
        """
        # Check if non-accepted coupons can be re-allocated to new members
        came_available_coupons = []
        expired_coupons = []
        add_to_queue = []

        for not_accepted_coupon in not_accepted_coupons:
            # Do a +1 here because the event column in included in the not_accepted_coupon-object
            coupon_expires_at = self.issues.loc[not_accepted_coupon['issue_id'],'expires_at']
            # coupon_expires_at = expire_times.loc[not_accepted_coupon[ISSUE_ID_COLUMN+1]]
            coupon_came_available_at = not_accepted_coupon['timestamp']

            if coupon_came_available_at < coupon_expires_at:
                # Coupon is not expired yet, add back to queue of coupons
                new_coupon_id = self.max_existing_coupon_id + 1
                self.max_existing_coupon_id += 1

                event = {'event': Event.coupon_available, 'timestamp':coupon_came_available_at, 'coupon_id':new_coupon_id, 'coupon_follow_id':not_accepted_coupon['coupon_follow_id'],
                         'issue_id': not_accepted_coupon['issue_id'], 'offer_id':not_accepted_coupon['offer_id'], 'member_id':np.nan, 'batch_id':np.nan}
                came_available_coupons.append(event)

                new_coupon = {'timestamp':coupon_came_available_at, 'coupon_id':new_coupon_id, 'coupon_follow_id':not_accepted_coupon['coupon_follow_id'],
                              'issue_id': not_accepted_coupon['issue_id'], 'offer_id':not_accepted_coupon['offer_id']}
                add_to_queue.append(new_coupon)
            else:
                # Coupon is expired
                event = {'event': Event.coupon_expired, 'timestamp':coupon_came_available_at, 'coupon_id':np.nan, 'coupon_follow_id':not_accepted_coupon['coupon_follow_id'],
                         'issue_id': not_accepted_coupon['issue_id'], 'offer_id':not_accepted_coupon['offer_id'], 'member_id':np.nan, 'batch_id':np.nan}
                expired_coupons.append(event)

        return came_available_coupons, expired_coupons, add_to_queue

    @staticmethod
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


    def simulate_member_responses(self, allocated_member_coupon_pairs, batch_indices, batch_to_send, batch_sent_at, verbose):
        """ Now that the allocation policy has chosen which coupons will be allocated to which members, let the members
        choose whether they accept the allocated coupons. The reactions are simulated.

        Parameters
        ---------
        allocated_member_coupon_pairs : tuple[list[int], list[int]]
        batch_indices : tuple[dict[int --> int], dict[int --> int]]
        batch_to_send : list[dict]
        batch_sent_at : dt.datetime

        Returns
        -------
        accepted_coupons : list[dict]
            A list of dictionaries, with each dictionary representing an event with the keys
            ['event','timestamp','coupon_id','coupon_follow_id','issue_id','offer_id','member_id','batch_id']
        not_accepted_coupons : list[dict]
            A list of dictionaries, with each dictionary representing an event with the keys
            ['event','timestamp','coupon_id','coupon_follow_id','issue_id','offer_id','member_id','batch_id']
        """
        # Unpack tuples
        batch_member_index_to_id, batch_coupon_index_to_id = batch_indices
        members_with_coupon_indices, allocated_coupon_indices = allocated_member_coupon_pairs

        # Retrieve probability of accepting a coupon for each member
        all_member_ids_to_index, all_offer_ids_to_index = self.utility_indices
        batch_member_ids = list(map(lambda batch_member_index: batch_member_index_to_id[batch_member_index], members_with_coupon_indices))
        all_member_index = list(map(lambda member_id: all_member_ids_to_index[member_id], batch_member_ids))
        batch_offer_ids = list(map(lambda batch_coupon_index: batch_to_send[batch_coupon_index]['offer_id'], allocated_coupon_indices))
        all_offer_index = list(map(lambda offer_id: all_offer_ids_to_index[offer_id], batch_offer_ids))
        accept_probabilites = self.P_accept_df.values[all_member_index, all_offer_index]


        # Simulate if coupon will be accepted or not
        coupon_accepted = np.random.uniform(0, 1, size=len(accept_probabilites)) < accept_probabilites
        # Draw from realistic decline times to simulate accept time
        random_indices = np.random.randint(len(self.decline_times), size=np.sum(coupon_accepted))

        accepted_coupons = []
        for i, (batch_member_index, batch_coupon_index) in enumerate(zip(members_with_coupon_indices[coupon_accepted], allocated_coupon_indices[coupon_accepted])):
            assert batch_coupon_index_to_id[batch_coupon_index] == batch_to_send[batch_coupon_index]['coupon_id']
            coupon = batch_to_send[batch_coupon_index]

            accept_time = self.offers.loc[coupon['offer_id'], 'accept_time']
            accept_time = batch_sent_at + dt.timedelta(days=float(accept_time)) * self.decline_times[random_indices[i]]

            accepted_coupon = {'event': Event.member_accepted, 'timestamp':accept_time, 'coupon_id':coupon['coupon_id'], 'coupon_follow_id':coupon['coupon_follow_id'],
                               'issue_id': coupon['issue_id'], 'offer_id':coupon['offer_id'], 'member_id':batch_member_index_to_id[batch_member_index], 'batch_id':np.nan}
            accepted_coupons.append(accepted_coupon)


        # Simulate if coupon will be declined or no response (expire)
        # TODO: improve upon P_let_expire_given_not_accepted --> make it member dependent instead of one global value?
        P_let_expire = self.P_let_expire_given_not_accepted
        coupon_let_expire = np.random.uniform(0, 1, size=np.sum(~coupon_accepted)) < P_let_expire
        # Draw from realistic decline times
        random_indices = np.random.randint(len(self.decline_times), size=np.sum(~coupon_accepted))

        not_accepted_coupons = []
        for i, (batch_member_index, batch_coupon_index) in enumerate(zip(members_with_coupon_indices[~coupon_accepted], allocated_coupon_indices[~coupon_accepted])):
            assert batch_coupon_index_to_id[batch_coupon_index] == batch_to_send[batch_coupon_index]['coupon_id']
            coupon = batch_to_send[batch_coupon_index]
            if coupon_let_expire[i]:
                accept_time = self.offers.loc[coupon['offer_id'], 'accept_time']
                expire_time = self.determine_coupon_checked_expiry_time(batch_sent_at, float(accept_time))
                expire_time = expire_time[0] # Take first of suggested possible expiry times
                event = {'event': Event.member_let_expire, 'timestamp':expire_time, 'coupon_id':coupon['coupon_id'], 'coupon_follow_id':coupon['coupon_follow_id'],
                         'issue_id': coupon['issue_id'], 'offer_id':coupon['offer_id'], 'member_id':batch_member_index_to_id[batch_member_index], 'batch_id':np.nan}
            else:
                accept_time = self.offers.loc[coupon['offer_id'], 'accept_time']
                decline_time = batch_sent_at + dt.timedelta(days=float(accept_time)) * self.decline_times[random_indices[i]]
                event = {'event': Event.member_declined, 'timestamp':decline_time, 'coupon_id':coupon['coupon_id'], 'coupon_follow_id':coupon['coupon_follow_id'],
                         'issue_id': coupon['issue_id'], 'offer_id':coupon['offer_id'], 'member_id':batch_member_index_to_id[batch_member_index], 'batch_id':np.nan}

            not_accepted_coupons.append(event)

        return accepted_coupons, not_accepted_coupons


    def send_out_chosen_coupons(self, allocated_member_coupon_pairs, batch_indices, batch_to_send, batch_sent_at, batch_ID):
        """ Make events of which coupon was sent to which member. Some coupons might not have been allocated to
        any member, so these can be readded to the queue
        
        Parameters
        ---------
        allocated_member_coupon_pairs : tuple[list[int], list[int]]
        batch_indices : tuple[dict[int --> int], dict[int --> int]]
        batch_to_send : list[dict]
        batch_sent_at : dt.datetime
        batch_ID : int
        
        Returns
        -------
        add_to_queue : list[dict]
            A list of dictionaries, with each dictionary representing a coupon to allocate with the keys
            [        'timestamp','coupon_id','coupon_follow_id','issue_id','offer_id']
        sent_coupons : list[dict]
            A list of dictionaries, with each dictionary representing an event with the keys
            ['event','timestamp','coupon_id','coupon_follow_id','issue_id','offer_id','member_id','batch_id']
        """
        # Unpack tuples
        batch_member_index_to_id, batch_coupon_index_to_id = batch_indices
        members_with_coupon_indices, allocated_coupon_indices = allocated_member_coupon_pairs

        # Add the coupons, that were not allocated, back to the queue
        add_to_queue = []
        non_allocated_coupons = set(np.arange(len(batch_to_send))) - set(allocated_coupon_indices)
        for batch_coupon_index in non_allocated_coupons:
            assert batch_coupon_index_to_id[batch_coupon_index] == batch_to_send[batch_coupon_index]['coupon_id']
            add_to_queue.append(batch_to_send[batch_coupon_index])

        sent_coupons = []
        for batch_member_index, batch_coupon_index in zip(members_with_coupon_indices, allocated_coupon_indices):
            assert batch_coupon_index_to_id[batch_coupon_index] == batch_to_send[batch_coupon_index]['coupon_id']
            coupon = batch_to_send[batch_coupon_index]
            sent_coupon = {'event': Event.coupon_sent, 'timestamp':batch_sent_at, 'coupon_id':coupon['coupon_id'], 'coupon_follow_id':coupon['coupon_follow_id'],
                           'issue_id': coupon['issue_id'], 'offer_id':coupon['offer_id'], 'member_id':batch_member_index_to_id[batch_member_index], 'batch_id':batch_ID}
            sent_coupons.append(sent_coupon)

        return add_to_queue, sent_coupons


    def determine_coupon_allocation(self, utility_to_allocate, verbose):
        """ Based on utility matrix, determine which members get which coupons

        Parameters
        ----------
        utility_to_allocate : np.array
            Each row x column is a combination of member x coupon, corresponding to the 
            increase in utility of that member accepting that coupon

        Returns
        -------
        X_a_r : np.array
            A matrix of booleans, indicating which combinations of member x coupon has 
            been chosen in the allocation. Rows are the agents / members, colums the resources / coupons
        """
        original_utility_shape = utility_to_allocate.shape

        # Remove coupons without any eligible members from batch_utility for a speed-up
        coupons_without_eligible_members = np.all(utility_to_allocate == -1, axis=0)
        if np.any(coupons_without_eligible_members):
            utility_to_allocate = utility_to_allocate[:,~coupons_without_eligible_members]

        # Determine allocation of coupons based on utilities
        TrackTime("Determining optimal allocation")
        start = dt.datetime.now()
        X_a_r = self.alloc_procedure(utility_to_allocate, verbose)
        if verbose: print("Took", (dt.datetime.now() - start), "to determine allocation")

        # Add the coupons without eligible members back
        if np.any(coupons_without_eligible_members):
            new_X_a_r = np.zeros(shape=original_utility_shape)
            new_X_a_r[:,~coupons_without_eligible_members] = X_a_r[:,:]
            X_a_r = new_X_a_r

        assert X_a_r.shape == original_utility_shape
        return X_a_r


    def filter_relevant_utilities(self, batch_to_send, batch_sent_at):
        """ Function that filters from the full utility matrix, the utilities relevant to the batch
        This depends both on the offer corresponding to coupons included in the batch,
        and the set of eligible members per offer
        The function can also adjust some of the utilities, dependent on the historical_context_type

        Parameters
        ----------
        batch_to_send : list[dict]
        batch_sent_at : dt.datetime

        Returns
        -------
        utility_to_allocate : np.array
            Filtered utilities with some utilities possibly adjusted based on historical context
        actual_utility : np.array
            Filtered utilities, with none of the utilities adjusted (all original values)

        batch_indices : tuple containing:
            member_index_to_id : dict[int --> int]
                Mapping the row numbers of the utility matrices to member id
            coupon_index_to_id : dict[int --> int]
                Mapping the column numbers of the utility matrices to coupon id
        """
        TrackTime("Filtering relevant utilies")
        member_id_to_index, offer_id_to_index = self.utility_indices

        # Decrease nr columns of utility_values based on relevant offers
        offer_ids_to_send = list(map(lambda coupon_dict: coupon_dict['offer_id'], batch_to_send))
        offer_indices_to_send = list(map(lambda offer_id: offer_id_to_index[offer_id], offer_ids_to_send))
        utility_to_allocate = self.utility_values[:, offer_indices_to_send]
        actual_utility = copy.copy(utility_to_allocate)

        if self.historical_context_type.name == Historical_Context_Type.time_discounted.name:
            TrackTime("Adjusting utilities (time-discounted)")
            utility_to_allocate -= self.historical_context.time_discounted_utilities.reshape(-1,1)
            utility_to_allocate[utility_to_allocate<0] = 0

        TrackTime("Get all eligible members")
        # Determine for every offer, the set of eligible members
        offer_id_to_eligible_members = self.get_all_eligible_members(batch_to_send, batch_sent_at)

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


        coupon_index_to_id = list(map(lambda coupon_dict: coupon_dict['coupon_id'], batch_to_send))
        member_index_to_id = all_eligible_member_ids

        assert utility_to_allocate.shape == actual_utility.shape
        assert utility_to_allocate.shape == (len(member_index_to_id), len(coupon_index_to_id)), "%s != %s"%(str(utility_to_allocate.shape), str((len(member_index_to_id), len(coupon_index_to_id))))
        return utility_to_allocate, actual_utility, (member_index_to_id, coupon_index_to_id)


    def extract_batch_from_queue(self, verbose):
        """ Sort the queue of coupons, then extract the coupons that are part of the next batch

        Returns
        -------
        batch_to_send : list[dict]
        batch_sent_at : dt.datetime
        """
        if len(self.unsorted_queue_of_coupons) <= self.min_batch_size:
            actual_batch_size = len(self.unsorted_queue_of_coupons)
            batch_sent_at = self.unsorted_queue_of_coupons[actual_batch_size-1]['timestamp']
        else:
            actual_batch_size = self.min_batch_size
            batch_sent_at = self.unsorted_queue_of_coupons[actual_batch_size-1]['timestamp']
            while True:
                if len(self.unsorted_queue_of_coupons) <= actual_batch_size:
                    break
                if self.unsorted_queue_of_coupons[actual_batch_size]['timestamp'] - batch_sent_at > dt.timedelta(seconds=3):
                    break
                actual_batch_size += 1

        # Extract 'actual_batch_size' number of coupons from the queue
        batch_to_send = self.unsorted_queue_of_coupons[:actual_batch_size]
        self.unsorted_queue_of_coupons = self.unsorted_queue_of_coupons[actual_batch_size:]

        return batch_to_send, batch_sent_at


    def check_expiry_unsent_coupons(self, batch_sent_at, prev_batch_unsent_coupons, verbose):
        """ Checks whether coupons that we NOT sent in the previous batch, can be retried in this
        next batch, or whether the coupons have expired by now

        Parameters
        ----------
        batch_sent_at : dt.datetime
        prev_batch_unsent_coupons : list[dict]

        Returns
        -------
        unsent_coupons_to_retry : list[dict]
        unsent_coupons_now_expired : list[dict]
        """
        # TODO: evaluate coupon expire also based on 'redeem_till' if 'redeem_type' in ['on_date', 'til_date']
        unsent_coupons_to_retry, unsent_coupons_now_expired = [], []

        for coupon in prev_batch_unsent_coupons:
            if abs(self.issues.loc[coupon['issue_id'],'expires_at'] - self.issues.loc[coupon['issue_id'],'sent_at']) < dt.timedelta(seconds=3):
                # If the issue expires the moment the coupons are sent out, we tolerate one round of sending out coupons, even if it is at most a few days after expiry
                if abs(batch_sent_at - self.issues.loc[coupon['issue_id'],'expires_at']) < dt.timedelta(days=3):
                    unsent_coupons_to_retry.append(coupon)
                else:
                    event = {'event': Event.coupon_expired, 'timestamp':self.issues.loc[coupon['issue_id'],'expires_at'], 'coupon_id':np.nan, 'coupon_follow_id':coupon['coupon_follow_id'],
                             'issue_id': coupon['issue_id'], 'offer_id':coupon['offer_id'], 'member_id':np.nan, 'batch_id':np.nan}
                    unsent_coupons_now_expired.append(event)
            else:
                if batch_sent_at < self.issues.loc[coupon['issue_id'],'expires_at']:
                    unsent_coupons_to_retry.append(coupon)
                else:
                    event = {'event': Event.coupon_expired, 'timestamp':self.issues.loc[coupon['issue_id'],'expires_at'], 'coupon_id':np.nan, 'coupon_follow_id':coupon['coupon_follow_id'],
                             'issue_id': coupon['issue_id'], 'offer_id':coupon['offer_id'], 'member_id':np.nan, 'batch_id':np.nan}
                    unsent_coupons_now_expired.append(event)

        if len(unsent_coupons_now_expired) > 0 and verbose:
            offer_ids = list(map(lambda coupon: coupon['offer_id'], unsent_coupons_now_expired))
            if verbose: print("\n%d coupons expired while waiting to send out next batch (offer-ids: %s)"%(len(unsent_coupons_now_expired), str(list(set(offer_ids)))))

        return unsent_coupons_to_retry, unsent_coupons_now_expired


    def is_batch_ready_to_be_sent(self, issue_counter):
        """ Returns True or False if the minimum batch size has been reached,
        also looking at whether all issues that should have been released before
        the batch timestamp, were actually released.

        Parameters
        ----------
        issue_counter : int

        Returns
        -------
         : bool
        """
        if len(self.unsorted_queue_of_coupons) < self.min_batch_size:
            return False

        self.unsorted_queue_of_coupons.sort(key=lambda my_dict: my_dict['timestamp'])

        time_of_sending_next_batch = self.unsorted_queue_of_coupons[self.min_batch_size-1]['timestamp']
        if issue_counter+1 < len(self.issues):
            time_of_next_issue = self.issues['sent_at'].iloc[issue_counter+1]
            if time_of_next_issue - time_of_sending_next_batch < dt.timedelta(seconds=3):
                # Even though we have enough coupons to reach minimum of batch_size,
                # We first have to process another issue, to include in this next batch
                return False

        return True


    def release_new_issue(self, issue):
        """ Add a new issue to the queue of coupons, and make 'coupon_available' events

        Parameters
        ----------
        issue : pd.Series
            Should have the indice ['amount','sent_at','id','offer_id']

        Returns
        -------
        add_to_queue : list[dict]
            A list of dictionaries, with each dictionary representing a coupon to allocate with the keys
            [        'timestamp','coupon_id','coupon_follow_id','issue_id','offer_id']
        came_available_coupons : list[dict]
            A list of dictionaries, with each dictionary representing an event with the keys
            ['event','timestamp','coupon_id','coupon_follow_id','issue_id','offer_id','member_id','batch_id']
        """
        new_coupon_ids        = np.arange(issue['amount']) + self.max_existing_coupon_id + 1
        new_coupon_follow_ids = np.arange(issue['amount']) + self.max_existing_coupon_follow_id + 1
        self.max_existing_coupon_id        += issue['amount']
        self.max_existing_coupon_follow_id += issue['amount']

        add_to_queue = []
        for new_id, new_follow_id in zip(new_coupon_ids, new_coupon_follow_ids):
            new_coupon = {'timestamp': issue['sent_at'], 'coupon_id': new_id, 'coupon_follow_id': new_follow_id,
                          'issue_id': issue['id'], 'offer_id':issue['offer_id']}
            add_to_queue.append(new_coupon)

        to_add = {'event': Event.coupon_available, 'member_id':np.nan, 'batch_id':np.nan}
        came_available_coupons = list(map(lambda my_dict: {**my_dict, **to_add}, add_to_queue))

        return add_to_queue, came_available_coupons


    def get_all_eligible_members(self, batch_to_send, batch_sent_at):
        """ Function that gets for each coupon in the batch, a set of eligible members to which
        the coupon can be sent to

        Parameters
        ---------
        batch_to_send : list[dict]
        batch_sent_at : dt.datetime

        Returns
        -------
        offer_id_to_eligible_members : dict[int --> np.array]
             A dictionary mapping the offer-id to a (time-dependent) set of eligible members
        """
        members = self.members

        # Phone nr and email criteria already in effect
        # members = members[~members['email'].isna()]
        # members = members[~members['mobile'].isna()]
        # member must be active to be eligible
        members = members[members['active'] == 1]

        offer_ids_to_send = Counter(list(map(lambda coupon_dict: coupon_dict['offer_id'], batch_to_send)))
        offer_id_to_eligible_members = {}

        for offer_id, nr_coupons_to_send in offer_ids_to_send.items():
            offer = self.offers.loc[offer_id, :].squeeze()
            eligible_members = members

            TrackTime("Get all eligible members static")
            eligible_members = self.get_eligible_members_static(eligible_members, offer)

            TrackTime("Get all eligible members time")
            eligible_members = self.get_eligible_members_time_dependent(eligible_members, offer, batch_sent_at)

            TrackTime("Get all eligible members history")
            eligible_members = self.get_eligible_members_historical_context(eligible_members, offer_id, nr_coupons_to_send, batch_sent_at)

            TrackTime("Get all eligible members")
            offer_id_to_eligible_members[offer_id] = eligible_members['id'].values

        return offer_id_to_eligible_members


################# GET ALL ELIGIBLE MEMBERS BASED ON HISTORICALLY RECEIVED COUPONS ########################################

    def get_eligible_members_historical_context(self, eligible_members, offer_id, nr_coupons_to_send, batch_sent_at, tracktime=False):
        """ Function that filters out more ineligible members by utilizing historical context

        Parameters
        ----------
        eligible_members : pd.DataFrame
            Set of eligible members so far
        offer_id : int
        nr_coupons_to_send : int
            The number of coupons in the same batch as this coupon
        batch_sent_at : dt.datetime
            To compute the numbers of days that have passed since the start of the simulation
            Comparison of day_nr (float) is much faster than comparison of datetime objects

        Returns
        ------
        eligible_members : pd.DataFrame
            Set of eligible members for this particular offer and timestamp
        """

        batch_sent_at_day_nr = (batch_sent_at - self.first_issue_sent_at).total_seconds() / 60 / 60 / 24

        def let_coupon_expire_last_month(member_id):
            return batch_sent_at_day_nr - self.historical_context.member_context[member_id]['let_last_c_expire_at'] < 30

        def accepted_coupon_last_month(member_id):
            return batch_sent_at_day_nr - self.historical_context.member_context[member_id]['accepted_last_c_at'] < 30

        def has_outstanding_coupon(member_id):
            accepted_last_coupon_at = self.historical_context.member_context[member_id]['accepted_last_c_at']
            let_last_coupon_expire_at = self.historical_context.member_context[member_id]['let_last_c_expire_at']
            # If last_accepted or last_let_expire is in the future, the member is yet to respond to the outstanding coupon with that response
            return batch_sent_at_day_nr < accepted_last_coupon_at or batch_sent_at_day_nr < let_last_coupon_expire_at

        members_who_already_received_this_offer = set(self.historical_context.offer_context[offer_id])

        if self.historical_context_type.name == Historical_Context_Type.full_historical.name:
            if tracktime: TrackTime("Recently let coupon expire")
            members_who_let_coupon_expire_in_last_month = \
                set(filter(let_coupon_expire_last_month, eligible_members['id'].values))
            members_to_filter = list(members_who_already_received_this_offer.union(members_who_let_coupon_expire_in_last_month))
        else:
            members_to_filter = list(members_who_already_received_this_offer)

        if tracktime: TrackTime("Filter members")
        eligible_members = eligible_members[~eligible_members['id'].isin(members_to_filter)]

        if len(eligible_members) <= nr_coupons_to_send:
            return eligible_members # No point in filtering as we already have fewer eligible members than coupons to send out


        if tracktime: TrackTime("Has outstanding coupon")
        members_with_outstanding = \
            set(filter(has_outstanding_coupon, eligible_members['id'].values))

        if self.historical_context_type.name == Historical_Context_Type.full_historical.name:
            if tracktime: TrackTime("Recently accepted coupon")
            members_who_accepted_coupon_in_last_month = \
                set(filter(accepted_coupon_last_month, eligible_members['id'].values))
            members_to_filter = list(members_with_outstanding.union(members_who_accepted_coupon_in_last_month))
        else:
            members_to_filter = list(members_with_outstanding)

        if tracktime: TrackTime("Filter members")
        filtered_eligible_members= eligible_members[~eligible_members['id'].isin(members_to_filter)]
        if len(filtered_eligible_members) >= nr_coupons_to_send:
            # Only filter if the filtering does not lead to a shortage of eligible members
            return filtered_eligible_members
        else:
            return eligible_members

################# GET ALL ELIGIBLE MEMBERS BASED ON TIME-DEPENDENT CRITERIA ########################################

    @staticmethod
    def child_age_to_stage(age):
        assert age >= 0
        for stage, age_range in CHILD_STAGES.items():
            if age_range[0] <= age and age <= age_range[1]:
                return stage
        raise ValueError("Age not part of a child stage")


    def get_eligible_members_time_dependent(self, eligible_members, offer, batch_sent_at, verbose=False, tracktime=False):
        """ Function that filters out more ineligible members by utilizing time-based criteria
        and the timestamp at which the coupon will be sent.

        Parameters
        ----------
        eligible_members : pd.DataFrame
            Set of eligible members so far
        offer : pd.DataFrame
            Criteria that have to be filtered on
        batch_sent_at : dt.datetime
            The time at which the criteria have to be evaluated against (i.e. age at this timestamp to check 'min_age >= 18')

        Returns
        ------
        eligible_members : pd.DataFrame
            Set of eligible members for this particular offer and timestamp
        """
        # TODO: use the columns 'inactivated_at', 'active', 'onboarded_at', 'receive_coupons_after', 'created_at' to determine eligible members

        date_at = batch_sent_at.date()
        def calculate_age_at(date_born):
            return date_at.year - date_born.year - ((date_at.month, date_at.day) < (date_born.month, date_born.day))


        if tracktime: TrackTime("calc age")
        # Calculate age of members
        min_age = offer['member_criteria_min_age']
        max_age = offer['member_criteria_max_age']
        if not pd.isna(min_age) or not pd.isna(max_age):
            members_age = eligible_members['date_of_birth'].apply(lambda born: calculate_age_at(born))
            members_age = members_age.values

        if tracktime: TrackTime("age criteria")
        # Minimum and maximum age criteria
        if not pd.isna(min_age) and not pd.isna(max_age):
            age_in_range = np.logical_and(members_age >= min_age, members_age <= max_age)
            eligible_members = eligible_members[age_in_range]
        elif not pd.isna(min_age):
            eligible_members = eligible_members[members_age >= min_age]
        elif not pd.isna(max_age):
            eligible_members = eligible_members[members_age <= max_age]
        if verbose: print("nr eligible_members age range:", len(eligible_members))

        if tracktime: TrackTime("setting up family criteria")
        # Family criteria
        fam_min_count       = offer['family_criteria_min_count']
        fam_max_count       = offer['family_criteria_max_count']
        fam_has_children    = offer['family_criteria_has_children']
        fam_child_min_age   = offer['family_criteria_child_age_range_min']
        fam_child_max_age   = offer['family_criteria_child_age_range_max']
        fam_child_stages    = offer['family_criteria_child_stages_child_stages']
        fam_child_gender    = offer['family_criteria_child_gender']

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

        if tracktime: TrackTime("calc children age")
        rel_children = copy.copy(self.children[self.children['user_id'].isin(eligible_members['id'])])
        rel_children['age'] = rel_children['date_of_birth'].apply(lambda born: calculate_age_at(born))
        rel_children = rel_children[rel_children['age'] >= 0]

        TrackTime("Calc children count")
        # Count number of children with age >= 0
        children_counts = Counter(rel_children['user_id'])
        children_counts = pd.DataFrame(children_counts.items(), columns=['id','children_count'])#.sort_values('id')
        # children_counts = rel_children.groupby('user_id').aggregate(children_count=('id','count')).reset_index().rename(columns={'user_id':'id'})
        # Merge children_count column into members table, and put members without chilldren on zero count
        eligible_members = pd.merge(eligible_members, children_counts, how='left', on='id').fillna(0)
        eligible_members['children_count'] = eligible_members['children_count'].astype(int)
        TrackTime("Get all eligible members time")

        if tracktime: TrackTime("calc family_count")
        # Calculate family count
        if not pd.isna(fam_min_count) or not pd.isna(fam_max_count):
            # query = "select * from member_family_member where type='partner'"
            # all_partners = pd.read_sql_query(query, db)
            rel_partners = self.partners[self.partners['user_id'].isin(eligible_members['id'])]
            partner_counts = rel_partners.groupby('user_id').aggregate(partner_count=('id','count')).reset_index().rename(columns={'user_id':'id'})
            # Merge partner_count column into members table, and put members without partner on zero count
            eligible_members = pd.merge(eligible_members, partner_counts, how='left', on='id')
            eligible_members['partner_count'] = eligible_members['partner_count'].fillna(0).astype(int)
            eligible_members['family_count'] = eligible_members['children_count'] + eligible_members['partner_count']

        if tracktime: TrackTime("family_count criteria")
        # Family count criteria
        if not pd.isna(fam_min_count):
            eligible_members = eligible_members[eligible_members['family_count'] >= fam_min_count]
        if not pd.isna(fam_max_count):
            eligible_members = eligible_members[eligible_members['family_count'] <= fam_max_count]
        if verbose: print("nr eligible_members fam_count range:", len(eligible_members))

        if tracktime: TrackTime("has_children criteria")
        # Has children criteria
        if not pd.isna(fam_has_children):
            if fam_has_children:
                eligible_members = eligible_members[eligible_members['children_count'] > 0]
            else:
                eligible_members = eligible_members[eligible_members['children_count'] == 0]
                # Do not further check children criteria, since members are not allowed to have children
                return eligible_members
        if verbose: print("nr eligible_members has_children:", len(eligible_members))

        if tracktime: TrackTime("other children criteria")
        # Children criteria
        if not pd.isna(fam_child_min_age):
            rel_children = rel_children[rel_children['age'] >= fam_child_min_age]
        if not pd.isna(fam_child_max_age):
            rel_children = rel_children[rel_children['age'] <= fam_child_max_age]
        if not pd.isna(fam_child_gender):
            rel_children = rel_children[rel_children['gender'] == fam_child_gender]
        if not pd.isna(fam_child_stages):
            rel_children['stage'] = rel_children['age'].apply(self.child_age_to_stage)
            accepted_child_stages = list(map(lambda string: string.strip('"'), fam_child_stages.strip("[]").split(",")))
            rel_children = rel_children[rel_children['stage'].isin(accepted_child_stages)]

        # Filter members with children that fit within the children criteria
        eligible_members = eligible_members[eligible_members['id'].isin(rel_children['user_id'])]
        if verbose: print("nr eligible_members children criteria:", len(eligible_members))

        return eligible_members

    ################# GET ALL ELIGIBLE MEMBERS BASED ON NON-TIME-DEPENDENT CRITERIA ########################################

    def get_eligible_members_static(self, eligible_members, offer, verbose=False, tracktime=False):
        """ Function that applies 'static' criteria.
        'static' in the sense that the result of the criteria should not change
        over (a short) time. In this case, because I only received a snapshot of the
        database at one point in time, these properties can physically not change.
        Things like: community, subscribed categories, gender, partners

        Parameters
        ----------
        eligible_members : pd.DataFrame
            Set of eligible members so far
        offer : pd.DataFrame
            Criteria that have to be filtered on

        Returns
        ------
        eligible_members : pd.DataFrame
            Set of eligible members for this particular offer
        """

        if tracktime: TrackTime("gender criteria")
        # Gender criteria
        required_gender = offer['member_criteria_gender']
        if not pd.isna(required_gender):
            eligible_members = eligible_members[eligible_members['gender'] == required_gender]
        if verbose: print("nr eligible_members gender:", len(eligible_members))

        if tracktime: TrackTime("is_single criteria")
        # Single or has partner criteria
        has_to_be_single = offer['family_criteria_is_single']
        if not pd.isna(has_to_be_single):

            if has_to_be_single:
                eligible_members = eligible_members[~eligible_members['id'].isin(self.partners['user_id'])]
            else:
                eligible_members = eligible_members[eligible_members['id'].isin(self.partners['user_id'])]
        if verbose: print("nr eligible_members has_to_be_single:", len(eligible_members))

        if tracktime: TrackTime("category_id criteria")
        # Member must be subscribed to category
        coupon_category_id = offer['category_id']
        if not pd.isna(coupon_category_id):
            members_subscribed_to_coupon_cat = self.member_categories[self.member_categories['category_id'] == coupon_category_id]
            eligible_members = eligible_members[eligible_members['id'].isin(members_subscribed_to_coupon_cat['member_id'])]
        if verbose: print("nr eligible_members category:", len(eligible_members))

        return eligible_members



    def init_new_sim(self):
        """ Resets attributes for the start of a new simulation.
        """
        # To be able to easily generate new unique coupon (follow) ids
        self.max_existing_coupon_id = -1
        self.max_existing_coupon_follow_id = -1

        # The list / df of events which will be exported in the end
        self.events_list = []
        self.events_df = pd.DataFrame(columns=['event','timestamp','coupon_id','coupon_follow_id','issue_id','offer_id','member_id','batch_id'])

        # Initialize queue and define the columns of an element in the queue
        self.unsorted_queue_of_coupons = []

        # Initialize historical context
        self.historical_context = Historical_Context(self.offers, self.members)

        self.first_issue_sent_at = self.issues['sent_at'].iloc[0].replace(hour=0,minute=0,second=0)


    def get_run_info(self, extra_run_info={}):
        """
        Returns
        -------
         : dict
             Information about the properties of the simulation currently being run
        """
        all_run_info = {'min_batch_size':       self.min_batch_size,
                        'alloc_procedure':      self.alloc_procedure_name,
                        'historical_context':   str(self.historical_context_type).replace('Historical_Context_Type.',''),
                        'version_info':         '',
                        'version_tag':          '',
                        }

        assert isinstance(extra_run_info, dict)
        all_run_info.update(extra_run_info)

        return all_run_info


    @staticmethod
    def convert_utility_indices_and_values_to_dataframe(utility_indices, utility_values):
        """ Function that can be used to combine a matrix of values, and member-ids and offer-ids into one dataframe
        The dataframe is not used by the simulation, as using a numpy array for the df.values and
        dictionaries for the columns and index works faster (since we are also going to be changing the df.values which goes faster with a np.array than pd.DataFrame)

        Returns
        -------
        utility_df : pd.DataFrame
            A dataframe with columns the offer-ids, and the index are the member-ids. Each combination of
            member x offer has a value, corresponding to the increase in utility of that member accepting a coupon from that offer
        """
        member_id_to_index, offer_id_to_index = utility_indices
        member_id_to_index = pd.DataFrame.from_dict(member_id_to_index, orient='index', columns=['index'])
        member_id_to_index = member_id_to_index.reset_index().rename(columns={'level_0':'member_id'}).sort_values(by='index')
        offer_id_to_index = pd.DataFrame.from_dict(offer_id_to_index, orient='index', columns=['index'])
        offer_id_to_index = offer_id_to_index.reset_index().rename(columns={'level_0':'offer_id'}).sort_values(by='index')
        assert np.all(offer_id_to_index['index'].values == np.arange(len(offer_id_to_index)))
        assert np.all(member_id_to_index['index'].values == np.arange(len(member_id_to_index)))
        utility_df = pd.DataFrame(utility_values, index=member_id_to_index['member_id'].values, columns=offer_id_to_index['offer_id'].values)
        return utility_df


    def export_run_info(self, all_run_info, run_nr):
        """ Export info about the properties of the simulation, and the matrix of accept proabilities being used by the simulation
        """
        run_export_folder = os.path.join(self.export_folder, "run_%d"%run_nr)
        if not os.path.exists(run_export_folder):
            os.makedirs(run_export_folder)

        # utility_df = self.convert_utility_indices_and_values_to_dataframe(self.utility_indices, self.utility_values)
        # utility_df.to_pickle(os.path.join(run_export_folder, '%d_utility_df.pkl'%run_nr))

        self.P_accept_df.to_pickle(os.path.join(run_export_folder, '%d_P_accept_df.pkl'%run_nr))
        with open(os.path.join(run_export_folder, '%d_info.json'%run_nr), 'w') as fp:
            json.dump(all_run_info, fp)


    def get_next_available_run_nr(self):
        """ Read which run_nrs are already taken, and calculate next available run_nr

        Returns
        ------
        run_nr : int
        """
        subfolders_in_export_folder = list(filter(lambda name: os.path.isdir(self.export_folder + name), os.listdir(self.export_folder)))
        def extract_run_nr(folder_name):
            try:    return int(folder_name.split("_")[1])
            except: return 0
        run_nr = max(list(map(extract_run_nr, subfolders_in_export_folder))) + 1 if len(subfolders_in_export_folder) > 0 else 1
        return run_nr



class Historical_Context:
    def __init__(self, offers, members):
        # Initialize historical context and the 3 values based on member-id (key)
        self.offer_context = {offer_id: set() for offer_id in offers['id'].values}
        self.member_context = {member_id: {'accepted_last_c_at':-np.inf, 
                                           'let_last_c_expire_at':-np.inf} for member_id in members['id'].values}
        # time_discounted_total_U_per_member
        self.time_discounted_utilities = np.zeros(len(members))


    def update_with_batch_results(self, sent_coupons, accepted_coupons, not_accepted_coupons, days_passed_since_prev_batch,
                                  utility_values, utility_indices, first_issue_sent_at):
        """ Function to update the historical context based on batch results
        The batch results include which members have been offered which coupons,
        which members accepted, which did not respond at all, and the times at which the events happened
        
        Parameters
        ----------
        sent_coupons : list[dict]
            A list stating which coupons have been allocated to which members (not necessarily accepted)
        accepted_coupons : list[dict]
            A list stating which members ended up accepting their allocated coupons
        not_accepted_coupons : list[dict]
            A list stating which members did not accept their allocated coupons
        days_passed_since_prev_batch : int
            Whether the previous batch was sent on the previous day (i.e. before midnights).
            More than one midnight could have passed since the previous batch
        utility_values : np.array
            Matrix of utility values for every member x offer combination
        utility_indices : tuple[dict[int --> int]]
            Mapping member-id and offer-id to utility_values index
        first_issue_sent_at : dt.datetime
            To compute the day number since start of simulation
        """
        # Update time-discounted utilities with the number of days_passed_since_prev_batch
        self.time_discounted_utilities[:] = self.time_discounted_utilities * (TIME_DISCOUNT_RATIO ** days_passed_since_prev_batch)
        member_id_to_index, offer_id_to_index = utility_indices

        # Update historical offer context
        for sent_coupon in sent_coupons:
            member_id = sent_coupon['member_id']
            offer_id = sent_coupon['offer_id']
            self.offer_context[offer_id].add(member_id)

        # Update historical member context: last_accepted_at
        for accepted_coupon in accepted_coupons:
            member_id = accepted_coupon['member_id']
            accepted_at_day_nr = (accepted_coupon['timestamp'] - first_issue_sent_at).total_seconds() / 60 / 60 / 24
            self.member_context[member_id]['accepted_last_c_at'] = accepted_at_day_nr

            # Add utility of accepted coupon to time-discounted sum of utilities
            offer_id = accepted_coupon['offer_id']
            member_index, offer_index = member_id_to_index[member_id], offer_id_to_index[offer_id]
            coupon_utility = utility_values[member_index, offer_index]

            self.time_discounted_utilities[member_index] += coupon_utility

        # Update historical member context: last_let_expire_at
        for not_accepted_coupon in not_accepted_coupons:
            if not_accepted_coupon['event'] == Event.member_let_expire:
                member_id = not_accepted_coupon['member_id']
                let_expire_at_day_nr = (not_accepted_coupon['timestamp'] - first_issue_sent_at).total_seconds() / 60 / 60 / 24
                self.member_context[member_id]['let_last_c_expire_at'] = let_expire_at_day_nr

                # Add utility of let_expire coupon to time-discounted sum of utilities (as form of punishment)
                offer_id = not_accepted_coupon['offer_id']
                member_index, offer_index = member_id_to_index[member_id], offer_id_to_index[offer_id]
                coupon_utility = utility_values[member_index, offer_index]

                self.time_discounted_utilities[member_index] += coupon_utility


