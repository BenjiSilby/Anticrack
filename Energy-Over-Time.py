# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 11:05:17 2025

@author: jack3
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statistics as st 
from mpl_toolkits import mplot3d
from sklearn.linear_model import LinearRegression
from matplotlib.animation import FuncAnimation
#from scipy.stats import skew, kurtosis
#import time
def argmedian(x):
  return np.argpartition(x, len(x) // 2)[len(x) // 2]
#%% Get Data
# Files names
# 'DEID_Particle_2025-03-31_15-05-01.csv'
# 'DEID_Particle_mar1325_total.csv'
# 'DEID_Particle_mar0325_total.csv'
# 'DEID_Particle_feb19_total.csv'
# 'DEID_Particle_2025-02-13_08-44-19.csv'
# 'DEID_Particle_2024-02-07_12-39-54.csv'

storm_list = ['DEID_Particle_2025-03-31_15-05-01.csv',
              'DEID_Particle_2025-03-31_15-05-01.csv',
              'DEID_Particle_2025-03-13_09-08-55.csv',
              'DEID_Particle_2025-03-03_18-47-33.csv',
              'DEID_Particle_feb19_total.csv',
              'DEID_Particle_2025-02-13_08-44-19.csv',
              'DEID_Particle_2024-02-07_12-39-54.csv']

# verify the cutoff time for 04-01-evening
cutoff_times = ['2025-04-01 17:00',
                '2025-04-01 06:47',
                '2025-03-13 17:41',
                '2025-03-04 20:42',
                '2025-02-20 18:30',
                '2025-02-14 12:23',
                '2025-02-07 16:34']

storm_dates = ['2025-04-01 PM',
                '2025-04-01 AM',
                '2025-03-13',
                '2025-03-04',
                '2025-02-20',
                '2025-02-14',
                '2025-02-07']


# Feb 19 has a different column name for FBF Snow Accum
snow_accum_col_name = ['FBF Snow Accum (mm)',
                       'FBF Snow Accum (mm)',
                       'FBF Snow Accum (mm)',
                       'FBF Snow Accum (mm)',
                       'Snow_Accum_mm',
                       'FBF Snow Accum (mm)',
                       'FBF Snow Accum (mm)']

# Change u in order to change the storm analyzed
u = 2

# read in the particle data
particleData = pd.read_csv(storm_list[u])

# Define the cutoff time
cutoff_time = pd.to_datetime(cutoff_times[u])

# convert time column to useable dates rather than timestamps
particleData["Time"] =  pd.to_datetime(particleData["Time"])

# filter the March 4th using height

# startTime is generally the beginning except for the April 1st storm
startTime = particleData["Time"][0]
if u == 0:
    startTime = pd.to_datetime('2025-04-01 07:18')
if u == 1:
    startTime = pd.to_datetime('2025-03-31 19:23')
if u ==2:
    startTime = pd.to_datetime('2025-03-12 15:56')
if u == 3:
    snowAccum = particleData[snow_accum_col_name[u]]
    snowAccum = snowAccum.to_numpy()
    lowerSnowAccum = max(snowAccum) - 24*10
    particleData = particleData[(particleData[snow_accum_col_name[u]] >= lowerSnowAccum)]
    #startTime = pd.to_datetime('2025-03-04 13:10')

# Filter the DataFrame
particleData = particleData[(particleData["Time"] < cutoff_time) & (particleData["Time"] >= startTime)]

#%% Variable Declaration

# use a fixed layer height of 5 mm
mm = 10

# create a column to store densities
mass = particleData['Mass']
volume = particleData['Volume HFD']
snowAccum = particleData[snow_accum_col_name[u]]
snowAccum = snowAccum.to_numpy()
SDI = particleData['SDI']
complexity = particleData['Complexity']
comlexity = complexity.to_numpy()
time = particleData['Time']
time = time.to_numpy()
failureLayer = 13*10; # cm to mm for the layer that fails in tap test

# filter complexity by setting less than/= 1 equal to nan these 
# particles hit the resolution limit of the thermal camera
complexity = np.array([float('nan') if x <= 1 else x for x in complexity])

# create storage variables for minimum density method
Hs = np.zeros(len(time))                  # slab height
rcs = np.zeros(len(time))                 # crack length
SIs = np.zeros(len(time))                 # Stability index
Wfs1 = np.zeros(len(time))                 # stability index / fracture energy non-dimensional
Wfs2 = np.zeros(len(time))                 # stability index / fracture energy non-dimensional
Wfs3 = np.zeros(len(time))                 # stability index / fracture energy non-dimensional
Wgs = np.zeros(len(time))                 # slab weight
min_vals = np.zeros((len(time), 3))           # array of minimum density values
min_idxs = np.zeros((len(time), 3))           # minimum density index values
max_min_idxs = np.zeros((len(time), 3))       # minimum density index values
predictedFails = np.zeros((len(time), 3))     # location of predicted fails

# create storage variables for density difference method
Hs_diff = np.zeros(len(time))                 # slab height
rcs_diff = np.zeros(len(time))               # crack length
SIs_diff = np.zeros(len(time))                # stability index
Wfs_diff1 = np.zeros(len(time))                # fracture energy non-dimensional
Wfs_diff2 = np.zeros(len(time))                # fracture energy non-dimensional
Wfs_diff3 = np.zeros(len(time))                # fracture energy non-dimensional
Wgs_diff = np.zeros(len(time))                # slab weight
max_vals = np.zeros((len(time), 3))               # array of max density differnces values
max_idxs = np.zeros((len(time), 3))               # max density difference index values
next_nonzero_index = np.zeros((len(time), 3))     # value that is the top of the layer index values
predictedFails_diff = np.zeros((len(time), 3))    # location of minimum densitys in cm

#%% Find largest jump

# Divisor is eyeballed for about 120 data points gathered
divisor = 120
d = (len(time)-2)//divisor
print(d)

int_range = range(1, (len(time)-1), d)

#%% Loop for getting variables
for q in int_range:
    
    layerNum = 1
    startVar = 0
    
    currentData = particleData[(particleData["Time"] <= time[q])]
    densCol = np.ones(len(currentData)) 
    mean_SDI = np.ones(len(currentData)) 
    mean_Complexity = np.ones(len(currentData)) 
    min_densVals = densCol
    dens_diff = np.zeros(densCol.shape)             # density difference value
    
    for i in range(0, len(currentData)):
        if snowAccum[i] - snowAccum[0] > layerNum*mm:
            # calculate the density
            densCol[startVar:i] = sum(mass[startVar:i])/sum(volume[startVar:i])
            
            # reset looping variables
            layerNum += 1
            startVar = i
        
        if i == len(particleData)-1:
            #densCol[startVar:-1, j] = sum(mass[startVar:-1])/sum(volume[startVar:-1])
            #mean_SDI[startVar:-1, j] = np.mean(SDI[startVar:-1])
            #mean_Complexity[startVar:-1, j] = np.nanmean(complexity[startVar:-1])
            densCol[startVar:-1] = float('nan')
    
    # assign last row to the correct density
    # densCol[-1, j] = sum(mass[startVar:-1])/sum(volume[startVar:-1])
    # mean_SDI[-1, j] = np.mean(SDI[startVar:-1])
    # mean_Complexity[-1, j] = np.nanmean(complexity[startVar:-1])
    densCol[-1] = float('nan')
    
    if layerNum > 10:
        # minimum density method
        min_densVals_col = densCol[:].copy()
        for k in range(3):
            # find the current minimum value
            min_vals[q, k] = np.nanmin(min_densVals_col)
            # get and store the index of the minimum value
            min_idxs_total = [idx for idx, val in enumerate(min_densVals_col) if val == min_vals[q, k]]  
            min_idxs[q, k] = min_idxs_total[0]
            max_min_idxs[q, k] = min_idxs_total[-1]
            # set values in the array equal to zero if they are the minimum
            min_densVals_col[min_densVals_col == min_vals[q, k]] = 1000

            # find the minimum layer height
            predictedFails[q, k] = snowAccum[int(min_idxs[q, k])]/10
        
        # get index of median using argmedian
        median_idx = argmedian(predictedFails_diff[q, :])
        
        # initialize FIXED anti-crack variables (DESIRE TO HAVE UPDATE BASED ON DEID DATA) 
        h = mm/1000      # assume weak layer is as thick as the average layer height in meters
        g = 9.81            # gravitation constant
        rho_ice = 917       # fixed ice denisty (kg m^-3) 
        
        # Get average slab density
        startLayer = int(min_idxs[q, median_idx])
        endLayer = int(max_min_idxs[q, median_idx]+1)
        # corrected on 5/28/25 to be sum mass over sum volume
        rho_slab = sum(mass[endLayer:q])/sum(volume[endLayer:q])
        
        # Fracture Toughness from McClung 2006 these are in kPa
        # K_Ic = 50*pow((rho_slab/rho_ice), 2.4)*1000          #kPa*m^0.5 to Pa*m^0.5
        # E = (9.68*10**5)*pow((rho_slab/rho_ice), 2.94)*1000  #kPa to Pa
        
        # # use Irwin's law to get fracture energy in Pa
        # Wf = ((K_Ic**2)/E)
        Wf1 = np.nanmean(complexity[endLayer:q])*(rho_slab/917)**(2*np.nanmean(complexity[endLayer:q]))
        Wf2 = (10**(-np.nanmean(complexity[endLayer:q])))*(rho_slab/917)**(np.nanmean(complexity[endLayer:q]))
        Wf3 = 0.07061183868583236*(rho_slab/917)**1.49608036286816
        
        # find slab height in meters
        H = (snowAccum[q] - snowAccum[endLayer])/1000
        
        # Move on to Heierli model
        # find slab weight 
        Wg = rho_slab*g*H*h/1000
        
        # non-dimensionalize Wf (If this is less than 1 we get failure)
        SI = Wf1/Wg
        
        # Get a crack length
        Rc = (0.75*SI)**0.25
        
        # value from Heirli model
        ro = 1.0            # assume touchdown length of a meter
        rc_avg = ro*Rc
        
        Hs[q] = H
        rcs[q] = rc_avg  
        SIs[q] = SI
        Wfs1[q] = Wf1
        Wfs2[q] = Wf2
        Wfs3[q] = Wf3
        Wgs[q] = Wg
            
        # density difference method
        for i in range(0, len(densCol)-1):
            dens_diff[i] = densCol[i+1] - densCol[i]
        
        dens_diff_col = dens_diff[:].copy()
        for k in range(3):
            # get the maximum of the density difference
            max_vals[q, k] = np.nanmax(dens_diff_col)
            # get the maximum index
            max_idxs[q, k] = np.nanargmax(dens_diff_col)
            # replace the value with 0
            dens_diff_col[dens_diff_col == max_vals[q, k]] = 0
            
            # find the minimum layer height
            predictedFails_diff[q, k] = snowAccum[int(max_idxs[q, k])]/10
            
        # initialize FIXED anti-crack variables (DESIRE TO HAVE UPDATE BASED ON DEID DATA) 
        h = mm/1000      # assume weak layer is as thick as the average layer height in meters
        g = 9.81            # gravitation constant
        rho_ice = 917       # fixed ice denisty (kg m^-3) 
        
        # get index of median using argmedian
        median_idx = argmedian(predictedFails_diff[q, :])
        
        # Get average slab density
        startLayer = int(max_idxs[q, median_idx])
        
        # find the end of the layer
        subarray = dens_diff_col[startLayer+1:]
        nonzero_indices = np.nonzero(subarray)[0]
        if nonzero_indices.size > 0:
            next_nonzero_index[q, median_idx] = max_idxs[q, median_idx] + 1 + nonzero_indices[0]
        
        # assign index for the end layer
        endLayer = int(next_nonzero_index[q, median_idx])
        # slab density updated 05/28/25
        rho_slab = sum(mass[endLayer:q])/sum(volume[endLayer:q])
        
        # Fracture Toughness from McClung 2006 these are in kPa
        # K_Ic = 50*pow((rho_slab/rho_ice), 2.4)*1000          #kPa*m^0.5 to Pa*m^0.5
        # E = (9.68*10**5)*pow((rho_slab/rho_ice), 2.94)*1000  #kPa to Pa
        
        # # use Irwin's law to get fracture energy in Pa
        # Wf = ((K_Ic**2)/E)
        Wf1 = np.nanmean(complexity[endLayer:q])*(rho_slab/917)**(2*np.nanmean(complexity[endLayer:q]))
        Wf2 = (10**(-np.nanmean(complexity[endLayer:q])))*(rho_slab/917)**(np.nanmean(complexity[endLayer:q]))
        Wf3 = 0.07061183868583236*(rho_slab/917)**1.49608036286816
        
        # find slab height in meters
        H = (snowAccum[q] - snowAccum[endLayer])/1000
        
        # Move on to Heierli model
        # find slab weight 
        Wg = rho_slab*g*H*h/1000
        
        # non-dimensionalize Wf (If this is less than 1 we get failure)
        SI = Wf1/Wg
        
        # Get a crack length
        Rc = (0.75*SI)**0.25
        
        # value from Heirli model
        ro = 1.0            # assume touchdown length of a meter
        rc_avg = ro*Rc
        Hs_diff[q] = H
        rcs_diff[q] = rc_avg  
        Wfs_diff1[q] = Wf1
        Wfs_diff2[q] = Wf2
        Wfs_diff3[q] = Wf3
        SIs_diff[q] = SI
        Wgs_diff[q] = Wg

#%% Plotting

# replace values that are zero with NaN because we skip many indicies using the looping method above
Wfs1[Wfs1 == 0] = np.nan
Wfs2[Wfs2 == 0] = np.nan
Wfs3[Wfs3 == 0] = np.nan

Wgs[Wgs == 0] = np.nan
SIs[SIs == 0] = np.nan

Wfs_diff1[Wfs_diff1 == 0] = np.nan
Wfs_diff2[Wfs_diff2 == 0] = np.nan
Wfs_diff3[Wfs_diff3 == 0] = np.nan

Wgs_diff[Wgs_diff == 0] = np.nan
SIs_diff[SIs_diff == 0] = np.nan

# convert time to useable time
time = pd.to_datetime(time)




plt.figure(dpi = 400)
plt.plot(time, Wfs1, '.', label = r"$G_{I_c}$")
#plt.plot(time, Wfs1, '.', label = r"$Complexity(\rho_s/\rho_{ice})^{2*Complexity}$")
#plt.plot(time, Wfs2, '.', label = r"$(10^{-Complexity})(\rho_s/\rho_{ice})^{Complexity}$")
#plt.plot(time, Wfs3, '.', label = r"$0.071(\rho_s/\rho_{ice})^{1.50}$")
plt.plot(time, Wgs, '.', label = r"$W_g\,h$, h = 10 mm")
plt.title("Fracture Energy over Time using Minimum Density Method (" + storm_dates[u] + ")")
plt.xlabel("Time (MM-DD HH)")
plt.ylabel(r"$kPa\,m$")
plt.xticks(rotation = 45)
plt.legend()

plt.figure(dpi = 400)
#plt.plot(time, Wfs2, '.', label = r"$G_{I_c}$")
#plt.plot(time, Wfs_diff1, '.', label = r"$Complexity(\rho_s/\rho_{ice})^{2*Complexity}$")
plt.plot(time, Wfs_diff2, '.', label = r"$(10^{-Complexity})(\rho_s/\rho_{ice})^{Complexity}$")
# plt.plot(time, Wfs_diff3, '.', label = r"$0.071(\rho_s/\rho_{ice})^{1.50}$")
plt.plot(time, Wgs_diff, '.', label = r"$W_g\,h$, h = 5 mm")
plt.title("Fracture Energy over Time using Maximum Density Difference Method (" + storm_dates[u] + ")")
plt.xlabel("Time (MM-DD HH)")
plt.ylabel(r"$kPa\,m$")
plt.xticks(rotation = 45)
plt.legend()

plt.figure(dpi = 400)
plt.plot(time, SIs, '.', label = r"$\rho_{min}$ Stability Index")
plt.title("Stability Indices over Time (" + storm_dates[u] + ")")
plt.xlabel("Time (MM-DD HH)")
plt.ylabel("Stability Index (SI)")
plt.xticks(rotation = 45)
plt.legend()

plt.figure(dpi = 400)
plt.plot(time, SIs_diff, '.', label = r"$\Delta \rho_{max}$ Stability Index")
plt.title("Stability Indices over Time (" + storm_dates[u] + ")")
plt.xlabel("Time (MM-DD HH)")
plt.ylabel("Stability Index (SI)")
plt.xticks(rotation = 45)
plt.legend()

#%% error bar plot
# Stack them together for easy computation
models_min_rho = np.vstack([Wfs1, Wfs2, Wfs3])  # shape: (3, 100)

# Compute median across models
median_model = np.nanmedian(models_min_rho, axis=0)

# Compute bounds (min and max) across the *other two* models
# If you want to visualize deviation from the median:
errors = np.abs(models_min_rho - median_model)
error_min = np.nanmin(errors, axis=0)
error_max = np.nanmax(errors, axis=0)

def interpolate_nans(y):
    x = np.arange(len(y))
    mask = ~np.isnan(y)
    return np.interp(x, x[mask], y[mask])

# Example
median_filled = interpolate_nans(median_model)
x = median_filled[0]
median_filled[median_filled == x] = np.nan

models_min_rho_filled = np.zeros(np.shape(models_min_rho))
hold = interpolate_nans(models_min_rho[0, :])
x = hold[0]
hold[hold == x] = np.nan
models_min_rho_filled[0, :] = hold

hold = interpolate_nans(models_min_rho[1, :])
x = hold[0]
hold[hold == x] = np.nan
models_min_rho_filled[1, :] = hold

hold = interpolate_nans(models_min_rho[2, :])
x = hold[0]
hold[hold == x] = np.nan
models_min_rho_filled[2, :] = hold

plt.figure(dpi = 400)

# Compute proper lower and upper bounds
lower_bound = np.nanmin(models_min_rho_filled, axis=0)
upper_bound = np.nanmax(models_min_rho_filled, axis=0)

# Then plot like this:
plt.fill_between(time, lower_bound, upper_bound, 
                 color='lightgrey', alpha=0.5, label='Model Range')

# Plot median
plt.plot(time, median_filled, color='black', label='Median')

# Optional: plot individual models
plt.plot(time, Wfs1, '.', alpha=0.5, label = r"$Complexity(\rho_s/\rho_{ice})^{2*Complexity}$")
plt.plot(time, Wfs2, '.', alpha=0.5, label = r"$(10^{-Complexity})(\rho_s/\rho_{ice})^{Complexity}$")
plt.plot(time, Wfs3, '.', alpha=0.5, label = r"$0.071(\rho_s/\rho_{ice})^{1.50}$")
plt.plot(time, Wgs, '.', label = r"$W_g*h$, h = 5mm")

plt.legend()
#plt.title("Fracture Energy over Time using Minimum Density Method (" + storm_dates[u] + ")")
plt.xlabel("Time (MM-DD HH)")
plt.ylabel(r"kPa\,m")
plt.xticks(rotation = 45)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(True)
plt.show()

models_max_rho = np.vstack([Wfs_diff1, Wfs_diff2, Wfs_diff3])  # shape: (3, 100)

# Compute median across models
median_model = np.nanmedian(models_max_rho, axis=0)

# Compute bounds (min and max) across the *other two* models
# If you want to visualize deviation from the median:
errors = np.abs(models_max_rho - median_model)
error_min = np.nanmin(errors, axis=0)
error_max = np.nanmax(errors, axis=0)

# Example
median_filled = interpolate_nans(median_model)
x = median_filled[0]
median_filled[median_filled == x] = np.nan

models_max_rho_filled = np.zeros(np.shape(models_max_rho))
hold = interpolate_nans(models_max_rho[0, :])
x = hold[0]
hold[hold == x] = np.nan
models_max_rho_filled[0, :] = hold

hold = interpolate_nans(models_max_rho[1, :])
x = hold[0]
hold[hold == x] = np.nan
models_max_rho_filled[1, :] = hold

hold = interpolate_nans(models_max_rho[2, :])
x = hold[0]
hold[hold == x] = np.nan
models_max_rho_filled[2, :] = hold

plt.figure(dpi = 400)

# Compute proper lower and upper bounds 
lower_bound = np.nanmin(models_max_rho_filled, axis=0)
upper_bound = np.nanmax(models_max_rho_filled, axis=0)

# Then plot like this:
plt.fill_between(time, lower_bound, upper_bound, 
                 color='lightgrey', alpha=0.5, label='Model Range')

# Plot median
plt.plot(time, median_filled, color='black', label='Median')

# Optional: plot individual models
plt.plot(time, Wfs_diff1, '.', alpha=0.5, label = r"$Complexity(\rho_s/\rho_{ice})^{2*Complexity}$")
plt.plot(time, Wfs_diff2, '.', alpha=0.5, label = r"$(10^{-Complexity})(\rho_s/\rho_{ice})^{Complexity}$")
plt.plot(time, Wfs_diff3, '.', alpha=0.5, label = r"$0.071(\rho_s/\rho_{ice})^{1.50}$")
plt.plot(time, Wgs_diff, '.', label = r"$W_g*h$, h = 5mm")

plt.legend()
#plt.title("Fracture Energy over Time using Density Difference Method (" + storm_dates[u] + ")")
plt.xlabel("Time (MM-DD HH)")
plt.ylabel(r"kPa\,m")
plt.xticks(rotation = 45)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(True)
plt.show()
