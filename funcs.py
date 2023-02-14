import pickle
import numpy as np
from numba import jit

#%% all functions

##function to load pickles
def loadall(filename):
    '''Takes a filename, loads pickles'''
    with open(filename, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break


# function to load pickles in form of an array
def pic_to_ar(name):
    '''Takes a filename, returns unpickled array'''
    items = loadall(name)
    c = list(items)
    c_ar = np.array(c)
    c_ar_sq = np.squeeze(c_ar)

    return c_ar_sq


# function to retrieve coactivation matrices
def coact(c_ar_sq):
    '''Takes unpickled array, returns coactivation matrix'''
    tr_for_high = c_ar_sq

    matr = np.zeros((len(tr_for_high), len(tr_for_high[0]), len(tr_for_high[0])))

    for numb in range(len(tr_for_high)):

        tr_for_high_first = tr_for_high[numb]

        cor_mat = np.zeros((len(tr_for_high_first), len(tr_for_high_first)))

        for i in range(len(tr_for_high_first)):
            for j in range(len(tr_for_high_first)):
                cor_mat[i, j] = (
                    np.sum(
                        np.logical_and(
                            tr_for_high_first[i, :] == 1, tr_for_high_first[j, :] == 1
                        )
                    )
                    / np.shape(tr_for_high_first)[1]
                )

        matr[numb] = cor_mat

    mean_matr_coactiv = np.mean(matr, axis=0)

    return mean_matr_coactiv


# function to retrieve coactivation matrices with transient step 40
def coact40(c_ar_sq):
    '''Takes unpickled array, returns coactivation matrix without 40 step transient'''
    tr_for_high = c_ar_sq[:, :, 40:]

    matr = np.zeros((len(tr_for_high), len(tr_for_high[0]), len(tr_for_high[0])))

    for numb in range(len(tr_for_high)):

        tr_for_high_first = tr_for_high[numb]

        cor_mat = np.zeros((len(tr_for_high_first), len(tr_for_high_first)))

        for i in range(len(tr_for_high_first)):
            for j in range(len(tr_for_high_first)):
                cor_mat[i, j] = (
                    np.sum(
                        np.logical_and(
                            tr_for_high_first[i, :] == 1, tr_for_high_first[j, :] == 1
                        )
                    )
                    / np.shape(tr_for_high_first)[1]
                )

        matr[numb] = cor_mat

    mean_matr_coactiv = np.mean(matr, axis=0)

    return mean_matr_coactiv


# function to retrieve shifted coactivation matrices
def coact_sh(c_ar_sq):
    '''Takes unpickled array, returns shifted coactivation matrix'''
    tr_for_high = c_ar_sq

    matr = np.zeros((len(tr_for_high), len(tr_for_high[0]), len(tr_for_high[0])))

    for numb in range(len(tr_for_high)):

        tr_for_high_first = tr_for_high[numb]
        tr_for_high_first_sh = np.zeros_like(tr_for_high_first)

        cor_mat = np.zeros((len(tr_for_high_first), len(tr_for_high_first)))

        for i in range(len(tr_for_high_first)):
            for j in range(len(tr_for_high_first)):
                tr_for_high_first_sh[j, :] = np.roll(tr_for_high_first[j, :], -1)

                cor_mat[i, j] = (
                    np.sum(
                        np.logical_and(
                            tr_for_high_first[i, :] == 1,
                            tr_for_high_first_sh[j, :] == 1,
                        )
                    )
                    / np.shape(tr_for_high_first)[1]
                )

        matr[numb] = cor_mat

    mean_matr_coactiv = np.mean(matr, axis=0)

    return mean_matr_coactiv


# function to get the repeating ends from the initial conditions
@jit(nopython=True, cache=True)
def attrs(chunk):
    '''Gets the repeating ends of the array'''
    c_ar_sq_us = chunk  # comment in and out for different conditions
    # nost = np.empty(1)

    steps_with_effects = 40  # transient period
    at_s = 3  # attractor size
    attrs = np.ones((np.shape(c_ar_sq_us)[0], 12, 3))

    for i in range(np.shape(c_ar_sq_us)[0]):
        loop_step = steps_with_effects
        while loop_step <= (np.shape(c_ar_sq_us)[2] - 2 * at_s):
            attractor = c_ar_sq_us[i][:, loop_step : loop_step + at_s]
            attractor_shift = c_ar_sq_us[i][:, loop_step + at_s : loop_step + 2 * at_s]
            loop_test = attractor == attractor_shift
            if loop_test.all() != 1:
                print("No stable attractor for condition ", i)
                break
            if loop_step == (np.shape(c_ar_sq_us)[2] - 2 * at_s):
                at = attractor
                attrs[i] = at
            loop_step += 1
    return attrs


# remove duplicates
def Extract(lst):
    '''Extracts first appearances from list'''
    return [item[0] for item in lst]


# find uniques
def Unique(lst):
    '''Finds unique items'''
    return [list(set(item)) for item in lst]


def in_list(c, classes):
    '''Returns -1 if c is contained in list'''
    for f, sublist in enumerate(classes):
        if c in sublist:
            return f
    return -1


# function to get all unique attractors
def un_roll(atr_list):
    '''Takes ends of an unpickled array, returns attractors of the system, their amount
    and the indices of the limit cycles'''
    no_at = 0  # no attractor of size 3 is defined
    fix_p0 = 0  # fixed point 0
    other = 0  # attractor of size 3
    indices_help = []

    for i in range(np.shape(atr_list)[0]):
        if ((atr_list[i] == 1).all()) == 1:
            no_at += 1
        if ((atr_list[i] == 0).all()) == 1:
            fix_p0 += 1
        if (((atr_list[i] == 1).all()) != 1) and ((atr_list[i] == 0).all()) != 1:
            other += 1
            indices_help.append(i)

    # create an arrray of all the attractors of size 3
    all_attractors = atr_list[indices_help]

    # calculate the number of unique attractors
    u_a = np.unique(all_attractors, axis=0)

    # calculate the counts of this attractors
    u_a_counts = np.unique(all_attractors, axis=0, return_counts=True)[-1]

    rollers = [[]]

    # need to check if in u_a any attractors which are just rolled versions of themselves
    for i in range(len(u_a)):
        for j in range(len(u_a)):
            if np.sum(u_a[i] == np.roll(u_a[j], 1, axis=1)) == 36:  # check rolled array
                k = in_list(i, rollers)
                q = in_list(j, rollers)
                if (k == -1) and (q == -1):
                    rollers.append([i, j])
                if (k == -1) and (q != -1):
                    rollers[q].append(i)
                    rollers.append([])
                if (k != -1) and (q == -1):
                    rollers[k].append(j)
                    rollers.append([])
                if (k != -1) and (q != -1):
                    # rollers[q].append(i)
                    # rollers[k].append(j)
                    rollers.append([])
        t = in_list(i, rollers)
        if t == -1:
            rollers.append([i])

    # remove empty lists
    rolled = [x for x in rollers if x != []]

    attractor_space = u_a[Extract(rolled), :, :]

    attractor_counts = np.zeros(len(attractor_space))

    for i in range(len(attractor_space)):
        attractor_counts[i] = np.sum(u_a_counts[Unique(rolled)[i]])

    return attractor_space, attractor_counts, indices_help
