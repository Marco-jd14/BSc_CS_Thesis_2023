# -*- coding: utf-8 -*-
"""
Created on Mon May 29 16:51:51 2023

@author: Marco
"""

import sys
import numpy as np
from pulp import LpMaximize, LpProblem, LpStatus, lpSum, LpVariable
import datetime as dt


def greedy(Uar, verbose=False):
    nr_A, nr_R = Uar.shape
    if nr_A < nr_R:
        if verbose: print("\nLess eligible members than resources: %d < %d"%(nr_A,nr_R))
    Xar = np.zeros_like(Uar, dtype=int)

    nr_non_eligible_members = np.sum(Uar < 0)
    nr_eligible_options = np.prod(Uar.shape) - nr_non_eligible_members

    # sort the utility-matrix with highest utility first
    row_indices, col_indices = np.unravel_index(np.flip(np.argsort(Uar, axis=None)), Uar.shape)

    row_list, col_list = [], []
    for i, (a, r) in enumerate(zip(row_indices, col_indices)):
        if i >= nr_eligible_options:
            # Do not allocate coupons when utilities are negative (aka members not eligible)
            if verbose: print("Not all resources could be allocated after %d iterations"%i)
            break

        if len(col_list) == nr_R:
            if verbose: print("All resources allocated after %d iterations"%i)
            break # Every resource is already allocated

        if len(row_list) == nr_A:
            if verbose: print("All members have a resource after %d iterations"%i)
            break # Every member already as a resource

        if a in row_list or r in col_list:
            continue
        else:
            row_list.append(a)
            col_list.append(r)

    Xar[row_list, col_list] = 1
    return Xar




def max_sum_utility(Uar, verbose=False, min_utility=0):
    model = LpProblem(name="max_sum_utility", sense=LpMaximize)
    nr_A, nr_R = Uar.shape

    if nr_A < nr_R:
        if verbose: print("\nLess eligible members than resources: %d < %d"%(nr_A,nr_R))



    ##### Defining the variables ####################################
    Xar = [["" for r in range(Uar.shape[-1])] for a in range(Uar.shape[0])]

    for a in range(nr_A):
        for r in range(nr_R):
            Xar[a][r] = LpVariable(name="x_%d,%d"%(a,r),cat="Binary")
    Xar = np.array(Xar)
    assert Uar.shape == Xar.shape

    # Ensure ineligible utilities do not get allocated
    Xar[Uar < min_utility] = 0


    ###### Objective Function ########################################
    obj_function = lpSum(Uar * Xar)
    model += obj_function


    ###### Constraints ##############################################
    # each agent...
    for a in range(nr_A):
        # ... gets at most 1 resource (but can also get 0 resources)
        constraint_max = lpSum(Xar[a,:]) <= 1
        model += (constraint_max, "Agent %d max 1 resource"%a)

    # each resource...
    for r in range(nr_R):
        # ... can only be allocated to 1 agent or 0 agents
        constraint = lpSum(Xar[:,r]) <= 1
        model += (constraint, "Resource %d allocated to at most 1 agent"%r)


    ####### Solve the model #########################################
    start = dt.datetime.now()
    if verbose: print(start.time())
    model.solve()
    if verbose: print(dt.datetime.now() - start)


    ####### Results ##################################################
    if model.status != 1:
        print(LpStatus[model.status])
        return

    if verbose:
        print(LpStatus[model.status])
        print(f"objective: {model.objective.value()}")

        allocated_resources = [None for r in range(nr_R)]
        for var in model.variables():
            if var.name.startswith("x_") and var.value() == 1:
                r = int(var.name.split(",")[-1])
                allocated_resources[r] = var
            elif var.value() > 0:
                print(f"{var.name}: {var.value()}")

        for resource in allocated_resources:
            if resource is not None:
                print(f"{resource.name}: {resource.value()}")

    # print(Xar.shape)
    # print(_convert_vars_to_value(Xar))
    # sys.exit()
    return _convert_vars_to_value(Xar)


def _convert_vars_to_value(Xar):
    for i, row in enumerate(Xar):
        for j, element in enumerate(row):
            try:
                Xar[i,j] = element.value()
            except:
                pass
    assert np.all(np.logical_or(Xar == 0, Xar == 1)), "Decision variable is not boolean?"
    return Xar.astype(int)







def main():
    np.random.seed(0)
    nr_agents = 10
    nr_unique_resources = 5
    Uar = np.random.uniform(0,0.7,size=(nr_agents, nr_unique_resources))

    Uar = [[  -1,  -1,  -1, 0.7,  -1],
           [  -1,  -1,  -1, 0.7,  -1],
           [ 0.1, 0.2, 0.3,  -1, 0.7],
           [0.15,0.25,0.35,  -1, 0.7]]
    Uar = np.array(Uar)
    print(Uar)

    # X_a_r = greedy(Uar, verbose=True)
    # print("\n", X_a_r, "\n")
    # X_a_r = max_sum_utility(Uar, verbose=True)
    # print("\n", X_a_r, "\n")
    X_a_r = maximin_utility(Uar, verbose=True)
    print("\n", X_a_r, "\n")

if __name__ == '__main__':
    main()