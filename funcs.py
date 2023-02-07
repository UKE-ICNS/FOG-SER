# To create numerical fifure of how close stimulations are 
# to healthy or PD states

import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
from matplotlib import pyplot
import pickle
from numba import jit

#%% all functions

##function to load pickles
def loadall(filename):
    with open(filename, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break

#function to load pickles in form of an array
def pic_to_ar(name):

    items = loadall(name)   
    c = list(items)
    c_ar=np.array(c)
    c_ar_sq = np.squeeze(c_ar)

    return c_ar_sq

#function to convert pickles to sum of spikes
def pers(c_ar_sq,T,NN):

    per=np.zeros((len(c_ar_sq), np.shape(c_ar_sq)[1])) #empty array for 
    #activation percentage in region depending on the initial state

    #calculate overall activity
    for i in range(len(c_ar_sq)):
        for reg in range(np.shape(c_ar_sq)[1]):
            reg_state = c_ar_sq[i,reg,:]
            #take all neurons with 1 in all time
            reg_activ = (reg_state==1)
            #number of spikes
            reg_activ_tot = np.sum(reg_activ)
            #get percents of activity
            reg_per = np.round((reg_activ_tot/(T*NN))*100, 2)
            #fill an array
            per[i,reg]=reg_per

    return per

#function to retrieve coactivation matrices
def coact(c_ar_sq):
    tr_for_high = c_ar_sq 

    matr = np.zeros((len(tr_for_high), len(tr_for_high[0]),len(tr_for_high[0])))

    for numb in range(len(tr_for_high)):

        tr_for_high_first = tr_for_high[numb]

        cor_mat = np.zeros((len(tr_for_high_first), len(tr_for_high_first)))

        for i in range(len(tr_for_high_first)):
            for j in range(len(tr_for_high_first)):
                #cor_mat[i,j]=np.sum(tr_for_high_first[i,:]==tr_for_high_first[j,:])/np.shape(tr_for_high_first)[1]
                cor_mat[i,j]=np.sum(np.logical_and(tr_for_high_first[i,:]==1,tr_for_high_first[j,:]==1))/np.shape(tr_for_high_first)[1]
                #cor_mat[i,j]=np.sum(np.logical_and(tr_for_high_first[i,:]==1,tr_for_high_first[j,:]==1))/np.sum(tr_for_high_first[i,:]==1) #for weird spikes

        matr[numb] = cor_mat

    mean_matr_coactiv = np.mean(matr, axis=0)

    print('done 1')
    
    return mean_matr_coactiv

#function to retrieve coactivation matrices
def coact40(c_ar_sq):
    tr_for_high = c_ar_sq[:,:,40:] 

    matr = np.zeros((len(tr_for_high), len(tr_for_high[0]),len(tr_for_high[0])))

    for numb in range(len(tr_for_high)):

        tr_for_high_first = tr_for_high[numb]

        cor_mat = np.zeros((len(tr_for_high_first), len(tr_for_high_first)))

        for i in range(len(tr_for_high_first)):
            for j in range(len(tr_for_high_first)):
                #cor_mat[i,j]=np.sum(tr_for_high_first[i,:]==tr_for_high_first[j,:])/np.shape(tr_for_high_first)[1]
                cor_mat[i,j]=np.sum(np.logical_and(tr_for_high_first[i,:]==1,tr_for_high_first[j,:]==1))/np.shape(tr_for_high_first)[1]
                #cor_mat[i,j]=np.sum(np.logical_and(tr_for_high_first[i,:]==1,tr_for_high_first[j,:]==1))/np.sum(tr_for_high_first[i,:]==1) #for weird spikes

        matr[numb] = cor_mat

    mean_matr_coactiv = np.mean(matr, axis=0)

    print('done 1')
    
    return mean_matr_coactiv

#function to retrieve shifted coactivation matrices
def coact_sh(c_ar_sq):
    tr_for_high = c_ar_sq 

    matr = np.zeros((len(tr_for_high), len(tr_for_high[0]),len(tr_for_high[0])))

    for numb in range(len(tr_for_high)):

        tr_for_high_first = tr_for_high[numb]
        tr_for_high_first_sh = np.zeros_like(tr_for_high_first)

        cor_mat = np.zeros((len(tr_for_high_first), len(tr_for_high_first)))

        for i in range(len(tr_for_high_first)):
            for j in range(len(tr_for_high_first)):
                tr_for_high_first_sh[j,:] = np.roll(tr_for_high_first[j,:], -1)
                #cor_mat[i,j]=np.sum(tr_for_high_first[i,:]==tr_for_high_first[j,:])/np.shape(tr_for_high_first)[1]
                cor_mat[i,j]=np.sum(np.logical_and(tr_for_high_first[i,:]==1,tr_for_high_first_sh[j,:]==1))/np.shape(tr_for_high_first)[1]
                #cor_mat[i,j]=np.sum(np.logical_and(tr_for_high_first[i,:]==1,tr_for_high_first[j,:]==1))/np.sum(tr_for_high_first[i,:]==1) #for weird spikes

        matr[numb] = cor_mat

    mean_matr_coactiv = np.mean(matr, axis=0)

    print('done 1')
    
    return mean_matr_coactiv

@jit(nopython=True, cache=True)
def attrs(chunk):
    c_ar_sq_us = chunk #comment in and out for different conditions
    #nost = np.empty(1)

    steps_with_effects = 40 #transient period
    at_s=3 #attractor size
    attrs = np.ones((np.shape(c_ar_sq_us)[0],12,3))

    for i in range(np.shape(c_ar_sq_us)[0]):
        loop_step = steps_with_effects
        while loop_step<=(np.shape(c_ar_sq_us)[2]-2*at_s):
            attractor = c_ar_sq_us[i][:,loop_step:loop_step+at_s]
            attractor_shift = c_ar_sq_us[i][:,loop_step+at_s:loop_step+2*at_s]
            loop_test = (attractor==attractor_shift)
            if loop_test.all()!=1: 
                print("No stable attractor for condition ", i)
                break
            if loop_step == (np.shape(c_ar_sq_us)[2]-2*at_s):
                at = attractor
                attrs[i] = at
            loop_step += 1
    return attrs

#remove duplicates           
def Extract(lst):
    return [item[0] for item in lst]

def Unique(lst):
    return [list(set(item)) for item in lst]

def in_list(c, classes):
    for f, sublist in enumerate(classes):
        if c in sublist:
            return f
    return -1

def un_roll(atr_list):
    no_at = 0 #no attractor of size 3 is defined
    fix_p0 = 0 #fixed point 0
    other = 0 #attractor of size 3
    indices_help = []

    for i in range(np.shape(atr_list)[0]):
        if ((atr_list[i]==1).all())==1: no_at += 1
        if ((atr_list[i]==0).all())==1: fix_p0 += 1
        if (((atr_list[i]==1).all())!=1) and ((atr_list[i]==0).all())!=1: 
            other +=1
            indices_help.append(i)

    #create an arrray of all the attractors of size 3
    all_attractors = atr_list[indices_help]

    #calculate the number of unique attractors
    u_a = np.unique(all_attractors,axis=0)

    #calculate the counts of this attractors
    u_a_counts = np.unique(all_attractors,axis=0,return_counts=True)[-1]

    rollers = [[]]

    #need to check if in u_a any attractors which are just rolled versions of themselves
    for i in range(len(u_a)):
        for j in range(len(u_a)):
            if np.sum(u_a[i] == np.roll(u_a[j],1,axis=1))==36: #check rolled array
                k=in_list(i, rollers)
                q=in_list(j, rollers)
                if (k==-1) and (q==-1):
                    rollers.append([i,j])
                if (k==-1) and (q!=-1):
                    rollers[q].append(i)
                    rollers.append([])
                if (k!=-1) and (q==-1):
                    rollers[k].append(j)
                    rollers.append([])
                if (k!=-1) and (q!=-1):
                    # rollers[q].append(i)
                    # rollers[k].append(j)
                    rollers.append([])
        t=in_list(i, rollers)
        if t==-1:
            rollers.append([i])

    #remove empty lists
    rolled = [x for x in rollers if x != []]           

    attractor_space = u_a[Extract(rolled),:,:]

    attractor_counts = np.zeros(len(attractor_space))

    for i in range(len(attractor_space)):
        attractor_counts[i] = np.sum(u_a_counts[Unique(rolled)[i]])

    return attractor_space, attractor_counts, indices_help


# #%%

# #%% projection graphs
# # for i in range(len(h_del)):
# #     for j in range(len(h_del)):
# #         if h_del[i,j]<=borderline_del[i,j]:
# #             if h_stn_del[i,j]<=borderline_del[i,j]:
# #                 mn[i,j]=1
# #             else:
# #                 mn[i,j]=2
# #         else:
# #             if h_stn_del[i,j]<=borderline_del[i,j]:
# #                 mn[i,j]=3
# #             else:
# #                 mn[i,j]=4

# # for i in range(len(A1)):
# #     for j in range(len(A1)):
# #         if A1[i,j]!=1:
# #             mn1[i,j]=None

# # import matplotlib.pyplot as plt
# # from mpl_toolkits.axes_grid1 import make_axes_locatable

# # figure = pyplot.figure()
# # ax = sns.heatmap(mn1, cbar_kws={'label': 'Property'}, xticklabels=plot_state, yticklabels=plot_state, cmap=plt.get_cmap("Dark2", 4), vmin=1, vmax=4, linewidth=0.5, square=True, cbar=False)#.set_title("Distance to healthy")
# # plt.ylabel("Region") 
# # plt.xlabel("Region state on the next step")
# # divider = make_axes_locatable(ax) 
# # cax = divider.append_axes("right", size="5%", pad=0.1) 
# # cbar = plt.colorbar(ax.collections[0], cax=cax)
# # cbar.set_ticks([1.5, 2.25, 3, 3.75])
# # cbar.ax.set_yticklabels(['STN- and STN+SNr-DBS \n are closer to \n the healthy state', 'STN+SNr-DBS is closer \n to the healthy state', 'STN-DBS is closer \n to the healthy state','STN- and STN+SNr-DBS \n are further from \n the healthy state'], size=10)
# # cbar.ax.tick_params(axis='y', which='major', length=0, pad=15)
# # #cbar.outline.set_edgecolor('black')
# # #cbar.outline.set_linewidth(2)
# # ax.set_title('Distance to healthy')
# # plt.tight_layout()
# # plt.show()