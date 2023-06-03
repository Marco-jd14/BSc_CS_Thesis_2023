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
    rows, cols = np.unravel_index(np.flip(np.argsort(Uar, axis=None)), Uar.shape)
    for i, (a, r) in enumerate(zip(rows,cols)):
        if i >= nr_eligible_options:
            # Do not allocate coupons when utilities are negative (aka members not eligible)
            if verbose: print("Not all resources could be allocated after %d iterations"%i)
            return Xar

        nr_members_assigned_to_resource = np.sum(Xar, axis=0)
        if np.all(nr_members_assigned_to_resource > 0):
            if verbose: print("All resources allocated after %d iterations"%i)
            return Xar # Every resource is already allocated

        nr_resources_allocated_to_members = np.sum(Xar, axis=1)
        if np.all(nr_resources_allocated_to_members > 0):
            if verbose: print("All members have a resource after %d iterations"%i)
            return Xar # Every member already as a resource

        if nr_members_assigned_to_resource[r] > 0:
            continue

        if nr_resources_allocated_to_members[a] == 0:
            Xar[a,r] = 1

    if verbose: print("Last coupon allocated on last iteration")
    assert i == np.prod(Uar.shape) - 1, "iteration %d != %d (shape=%s)"%(i, np.prod(Uar.shape) - 1, str(Uar.shape))
    assert np.all(np.sum(Xar, axis=0) > 0) or np.all(np.sum(Xar, axis=1) > 0), "Not all resources allocated and not all members got a resource"
    return Xar




def max_utility(Uar, verbose=False, min_utility=-1):
    model = LpProblem(name="max_utilty", sense=LpMaximize)
    nr_A, nr_R = Uar.shape

    if nr_A < nr_R:
        if verbose: print("\nLess eligible members than resources: %d < %d"%(nr_A,nr_R))

    Xar = [["" for r in range(Uar.shape[-1])] for a in range(Uar.shape[0])]

    # Define the variables
    for a in range(nr_A):
        for r in range(nr_R):
            Xar[a][r] = LpVariable(name="x_%d,%d"%(a,r),cat="Binary")
    Xar = np.array(Xar)
    assert Uar.shape == Xar.shape

    # Ensure ineligible utilities do not get allocated
    Xar[Uar <= min_utility] = 0

    # Make the objective function (utilitarian)
    obj_function = lpSum(Uar * Xar)
    model += obj_function

    # Add constraints
    # each agent...
    for a in range(nr_A):
        # ... gets at most 1 resource ...
        constraint_max = lpSum(Xar[a,:]) <= 1
        # ... but can also get 0 resources
        constraint_min = lpSum(Xar[a,:]) >= 0 # Can be left out (implied by binary nature of decision variables)
        model += (constraint_max, "Agent %d max 1 resource"%a)
        model += (constraint_min, "Agent %d min 0 resources"%a)

    # each resource...
    for r in range(nr_R):
        # ... can only be allocated to 1 agent or 0 agents
        constraint = lpSum(Xar[:,r]) <= 1
        model += (constraint, "Resource %d allocated to at most 1 agent"%r)

    # Solve the model
    start = dt.datetime.now()
    if verbose: print(start.time())
    model.solve()
    if verbose: print(dt.datetime.now() - start)

    # Print the results
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
            assert Xar[i,j] == 0 or Xar[i,j] == 1, "Decision variable is not boolean? '%s'"%str(Xar[i,j])
    return Xar.astype(int)
