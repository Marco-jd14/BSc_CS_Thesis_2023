# -*- coding: utf-8 -*-
"""
Created on Mon May 29 16:51:51 2023

@author: Marco
"""

import numpy as np


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

