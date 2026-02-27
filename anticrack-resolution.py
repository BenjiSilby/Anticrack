# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 12:32:16 2025

@author: jack3
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import statistics as st
#from mpl_toolkits import mplot3d
#from sklearn.linear_model import LinearRegression
#from matplotlib.animation import FuncAnimation
#from scipy.stats import skew, kurtosis
#import time
#%%
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
u = 4

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

#%% Analysis cell

# list of snow heights we would like to investigate
mm = np.arange(1, 11, 1)
weak_layers = 5

# create a column to store densities
densCol = np.ones([len(particleData), len(mm)])  
mean_SDI = np.ones([len(particleData), len(mm)]) 
mean_Complexity = np.ones([len(particleData), len(mm)]) 
particle_counts = np.ones([len(particleData), len(mm)])
resolution = np.zeros((len(particleData),  len(mm)))
min_densVals = densCol

mass = particleData['Mass']
volume = particleData['Volume HFD']
snowAccum = particleData[snow_accum_col_name[u]]
snowAccum = snowAccum.to_numpy()
SDI = particleData['SDI']
complexity = particleData['Complexity']
comlexity = complexity.to_numpy()
failureLayer = 13*10; # cm to mm for the layer that fails in tap test

# filter complexity by setting less than/= 1 equal to nan these 
# particles hit the resolution limit of the thermal camera
complexity = np.array([float('nan') if x <= 1 else x for x in complexity])

# create storage variables
Hs = np.ones((len(mm), 3))                  # slab height
rcs = np.ones((len(mm), 3))                 # crack length
wfs = np.ones((len(mm), 3))                 # stability index / fracture energy non-dimensional
Wgs = np.ones((len(mm), 3))                 # slab weight
min_vals = np.zeros((len(mm), 3))           # array of minimum density values
min_idxs = np.zeros((len(mm), 3))           # minimum density index values
max_min_idxs = np.zeros((len(mm), 3))       # minimum density index values
predictedFails = np.zeros((len(mm), 3))     # location of minimum densitys in cm

for j in range(0, len(mm)):
    layerNum = 1
    startVar = 0
    
    for i in range(0, len(particleData)):
        if snowAccum[i] - snowAccum[0] > layerNum*mm[j]:
            # calculate the density
            densCol[startVar:i, j] = sum(mass[startVar:i])/sum(volume[startVar:i])
            mean_SDI[startVar:i, j] = np.mean(SDI[startVar:i])
            mean_Complexity[startVar:i, j] = np.nanmean(complexity[startVar:i])
            
            # resolution calculation
            particle_count = len(densCol[startVar:i, j])
            min_size = min(volume[startVar:i])
            particle_counts[startVar:i, j] = particle_count
            resolution[startVar:i, j] = min_size/particle_count
            
            # reset looping variables
            layerNum += 1
            startVar = i
        
        if i == len(particleData)-1:
            #densCol[startVar:-1, j] = sum(mass[startVar:-1])/sum(volume[startVar:-1])
            #mean_SDI[startVar:-1, j] = np.mean(SDI[startVar:-1])
            #mean_Complexity[startVar:-1, j] = np.nanmean(complexity[startVar:-1])
            densCol[startVar:-1, j] = float('nan')
            mean_SDI[startVar:-1, j] = float('nan')
            mean_Complexity[startVar:-1, j] = float('nan')
            particle_counts[startVar:-1, j] = float('nan')
            resolution[startVar:-1, j] = float('nan')
    
    # assign last row to the correct density
    # densCol[-1, j] = sum(mass[startVar:-1])/sum(volume[startVar:-1])
    # mean_SDI[-1, j] = np.mean(SDI[startVar:-1])
    # mean_Complexity[-1, j] = np.nanmean(complexity[startVar:-1])
    densCol[-1, j] = float('nan')
    mean_SDI[-1, j] = float('nan')
    mean_Complexity[-1, j] = float('nan')
    resolution[-1, j] = float('nan')
    particle_counts[-1, j] = float('nan')
    
#%% minimum density method
for j in range(0, len(mm)):
    # create a dummy variable to find three minimum density values
    min_densVals_col = densCol[:, j].copy()
    
     # Find minimum density and associated layer
    for k in range(3):
        # find the current minimum value
        min_vals[j, k] = np.nanmin(min_densVals_col)
        # get and store the index of the minimum value
        min_idxs_total = [idx for idx, val in enumerate(min_densVals_col) if val == min_vals[j, k]]  
        min_idxs[j, k] = min_idxs_total[0]
        max_min_idxs[j, k] = min_idxs_total[-1]
        # set values in the array equal to zero if they are the minimum
        min_densVals_col[min_densVals_col == min_vals[j, k]] = 1000

        # find the minimum layer height
        predictedFails[j, k] = snowAccum[int(min_idxs[j, k])]/10
        
        # initialize FIXED anti-crack variables (DESIRE TO HAVE UPDATE BASED ON DEID DATA) 
        h = mm[j]/1000      # assume weak layer is as thick as the average layer height in meters
        g = 9.81            # gravitation constant
        rho_ice = 917       # fixed ice denisty (kg m^-3) 
        
        # Get average slab density
        startLayer = int(min_idxs[j, k])
        endLayer = int(max_min_idxs[j, k]+1)
        # corrected on 5/28/25 to be sum mass over sum volume
        rho_slab = sum(mass[endLayer:-1])/sum(volume[endLayer:-1])
        
        # Fracture Toughness from McClung 2006 these are in kPa
        K_Ic = 50*pow((rho_slab/rho_ice), 2.4)*1000          #kPa*m^0.5 to Pa*m^0.5
        E = (9.68*10**5)*pow((rho_slab/rho_ice), 2.94)*1000  #kPa to Pa
        
        # use Irwin's law to get fracture energy in Pa
        Wf = ((K_Ic**2)/E)
        
        # find slab height in meters
        H = (snowAccum[len(snowAccum)-1] - snowAccum[endLayer])/1000
        
        # Move on to Heierli model
        # find slab weight 
        Wg = rho_slab*g*H*h
        
        # non-dimensionalize Wf (If this is less than 1 we get failure)
        wf = Wf/Wg
        
        # Get a crack length
        Rc = (0.75*wf)**0.25
        
        # value from Heirli model
        ro = 1.0            # assume touchdown length of a meter
        rc_avg = ro*Rc
        
        Hs[j, k] = H
        rcs[j, k] = rc_avg  
        wfs[j, k] = wf
        Wgs[j, k] = Wg
        
#%% density difference method

# create storage variables
Hs_diff = np.ones((len(mm), 3))                 # slab height
rcs_diff = np.ones((len(mm), 3))                # crack length
wfs_diff = np.ones((len(mm), 3))                # stability index / fracture energy non-dimensional
Wgs_diff = np.ones((len(mm), 3))                # slab weight
max_vals = np.zeros((len(mm), 3))               # array of max density differnces values
max_idxs = np.zeros((len(mm), 3))               # max density difference index values
next_nonzero_index = np.zeros((len(mm), 3))     # value that is the top of the layer index values
predictedFails_diff = np.zeros((len(mm), 3))    # location of minimum densitys in cm
dens_diff = np.zeros(densCol.shape)             # density difference value

for j in range(len(densCol[0,:])):
    for i in range(len(densCol[:, 0])-1):
        dens_diff[i, j] = densCol[i+1, j] - densCol[i, j]
    
    dens_diff_col = dens_diff[:, j].copy()
    for k in range(3):
        # get the maximum of the density difference
        max_vals[j, k] = np.nanmax(dens_diff_col)
        # get the maximum index
        max_idxs[j, k] = np.nanargmax(dens_diff_col)
        # replace the value with 0
        dens_diff_col[dens_diff_col == max_vals[j, k]] = 0
        
        # find the minimum layer height
        predictedFails_diff[j, k] = snowAccum[int(max_idxs[j, k])]/10
        
        # initialize FIXED anti-crack variables (DESIRE TO HAVE UPDATE BASED ON DEID DATA) 
        h = mm[j]/1000      # assume weak layer is as thick as the average layer height in meters
        g = 9.81            # gravitation constant
        rho_ice = 917       # fixed ice denisty (kg m^-3) 
        
        # Get average slab density
        startLayer = int(max_idxs[j, k])
        
        # find the end of the layer
        subarray = dens_diff_col[startLayer+1:]
        nonzero_indices = np.nonzero(subarray)[0]
        if nonzero_indices.size > 0:
            next_nonzero_index[j, k] = max_idxs[j, k] + 1 + nonzero_indices[0]
        
        # assign index for the end layer
        endLayer = int(next_nonzero_index[j, k])
        # slab density updated 05/28/25
        rho_slab = sum(mass[endLayer:-1])/sum(volume[endLayer:-1])
        
        # Fracture Toughness from McClung 2006 these are in kPa
        K_Ic = 50*pow((rho_slab/rho_ice), 2.4)*1000          #kPa*m^0.5 to Pa*m^0.5
        E = (9.68*10**5)*pow((rho_slab/rho_ice), 2.94)*1000  #kPa to Pa
        
        # use Irwin's law to get fracture energy in Pa
        Wf = ((K_Ic**2)/E)
        
        # find slab height in meters
        H = (snowAccum[len(snowAccum)-1] - snowAccum[endLayer])/1000
        
        # Move on to Heierli model
        # find slab weight 
        Wg = rho_slab*g*H*h
        
        # non-dimensionalize Wf (If this is less than 1 we get failure)
        wf = Wf/Wg
        
        # Get a crack length
        Rc = (0.75*wf)**0.25
        
        # value from Heirli model
        ro = 1.0            # assume touchdown length of a meter
        rc_avg = ro*Rc
        Hs_diff[j, k] = H
        rcs_diff[j, k] = rc_avg  
        wfs_diff[j, k] = wf
        Wgs_diff[j, k] = Wg
        
#%% SI by layer
layer_SI = np.ones([len(particleData), len(mm)])  
layer_Wg = np.ones([len(particleData), len(mm)])  
layer_rc = np.ones([len(particleData), len(mm)])  
layer_slab_density = np.ones([len(particleData), len(mm)])  
si_is_one_density = np.ones(len(mm))

layer_KI = np.ones([len(particleData), len(mm)])  
layer_E = np.ones([len(particleData), len(mm)])  

layer_H = np.ones([len(particleData), len(mm)])  

# non-dimensional groups
non_dim_Wf = np.ones([len(particleData), len(mm)])  
non_dim_E = np.ones([len(particleData), len(mm)])  
non_dim_Ki = np.ones([len(particleData), len(mm)])  
non_dim_rho = np.ones([len(particleData), len(mm)])

for j in range(0, len(mm)):
    layerNum = 1
    startVar = 0
    
    # initialize FIXED anti-crack variables (DESIRE TO HAVE UPDATE BASED ON DEID DATA) 
    h = mm[j]/1000      # assume weak layer is as thick as the average layer height in meters
    g = 9.81            # gravitation constant
    rho_ice = 917       # fixed ice denisty (kg m^-3) 
    
    for i in range(0, len(particleData)):
        if snowAccum[i] - snowAccum[0] > layerNum*mm[j]:
            # take the mean of all the snow above it 
            rho_slab = sum(mass[i:-1])/sum(volume[i:-1])
            
            # Fracture Toughness from McClung 2006 these are in kPa
            K_Ic = 50*pow((rho_slab/rho_ice), 2.4)*1000          #kPa*m^0.5 to Pa*m^0.5
            E = (9.68*10**5)*pow((rho_slab/rho_ice), 2.94)*1000  #kPa to Pa
            
            # use Irwin's law to get fracture energy in Pa
            Wf = ((K_Ic**2)/E)
            
            # find slab height in meters
            H = (snowAccum[len(snowAccum)-1] - snowAccum[i])/1000
            
            # Move on to Heierli model
            # find slab weight 
            Wg = rho_slab*g*H
            
            # non-dimensionalize Wf (If this is less than 1 we get failure)
            wf = Wf/(Wg*h)
            
            # Get a crack length
            Rc = (0.75*wf)**0.25
            
            # value from Heirli model
            ro = 1.0            # assume touchdown length of a meter
            rc = ro*Rc
            
            # assign values to the layer
            layer_SI[startVar:i, j] = wf
            layer_Wg[startVar:i, j] = Wg
            layer_rc[startVar:i, j] = rc
            layer_slab_density[startVar:i, j] = rho_slab
            
            layer_KI[startVar:i, j] = K_Ic
            layer_E[startVar:i, j] = E
            layer_H[startVar:i, j] = H
            
            # non dimensional groups
            non_dim_Wf[startVar:i, j] = Wf/(rho_slab*g*pow(H, 2))
            non_dim_Ki[startVar:i, j] = K_Ic/(rho_slab*g*pow(H, 3/2))
            non_dim_E[startVar:i, j] = E/(rho_slab*g*H)
            non_dim_rho[startVar:i, j] = rho_slab/rho_ice
            
            layerNum += 1
            startVar = i
            
        if i == len(particleData)-1:
            #densCol[startVar:-1, j] = sum(mass[startVar:-1])/sum(volume[startVar:-1])
            #mean_SDI[startVar:-1, j] = np.mean(SDI[startVar:-1])
            #mean_Complexity[startVar:-1, j] = np.nanmean(complexity[startVar:-1])
            layer_SI[startVar:-1, j] = float('nan')
            layer_Wg[startVar:-1, j] = float('nan')
            layer_rc[startVar:-1, j] = float('nan')
            layer_slab_density[startVar:-1, j] = float('nan')
            
            # non dimensional groups
            non_dim_Wf[startVar:-1, j] = float('nan')
            non_dim_Ki[startVar:-1, j] = float('nan')
            non_dim_E[startVar:-1, j] = float('nan')
            non_dim_rho[startVar:-1, j] = float('nan')
            
            layer_KI[startVar:-1, j] = float('nan')
            layer_E[startVar:-1, j] = float('nan')
            layer_H[startVar:-1, j] = float('nan')
            
    # ensure the last value is replaced with NAN
    layer_SI[-1, j] = float('nan')
    layer_Wg[-1, j] = float('nan')
    layer_rc[-1, j] = float('nan')
    layer_slab_density[-1, j] = float('nan')
    
    # non dimensional groups
    non_dim_Wf[-1, j] = float('nan')
    non_dim_Ki[-1, j] = float('nan')
    non_dim_E[-1, j] = float('nan')
    non_dim_rho[-1, j] = float('nan')
    
    layer_KI[-1, j] = float('nan')
    layer_E[-1, j] = float('nan')
    layer_H[-1, j] = float('nan')
    
    min_abs_diff = abs(layer_SI[:, j] - 1)
    abs_min_diff_loc = np.argmin(min_abs_diff)
    si_is_one_density[j] = densCol[abs_min_diff_loc, j]
    
#%% Plotting Hlines

layerHeight = 5
index = round((layerHeight - 1)/1)

plt.figure(dpi = 400)
plt.plot(densCol[:, index], snowAccum/10-min(snowAccum)/10, label = 'Density')
#plt.plot(dens_diff[:, index],  snowAccum/10, label = 'Density Diff')
plt.xlabel(r'Density ($kg\,m^{-3}$)')
plt.ylabel('Snow Height (cm)')
plt.title('Density across Snow Height ' + storm_dates[u])
plt.text(min(densCol[:, index]), max(snowAccum/10)-min(snowAccum)/10 - 0.5, f'Layer Height = {layerHeight} mm')
minLabels = ['min 1', 'min 2', 'min 3']
maxLabels = ['max 1', 'max 2', 'max 3']
minColors = ['r', 'darkred', 'salmon']
maxColors = ['b', 'navy', 'skyblue']

plt.hlines(np.median(max(snowAccum/10) - Hs[index, :]*100)-min(snowAccum)/10, min(densCol[:, index]), max(densCol[:, index]), label = 'Median of Min', color = 'cyan', linestyle = '-')
plt.hlines(np.median(max(snowAccum/10) - Hs_diff[index, :]*100)-min(snowAccum)/10, min(densCol[:, index]), max(densCol[:, index]), label = 'Median of Max', color = 'gold', linestyle = '-')
#plt.hlines(np.mean(max(snowAccum/10) - Hs[index, :]*100), min(densCol[:, index]), max(densCol[:, index]), label = 'Median of Min', color = 'purple', linestyle = '-')
#plt.hlines(np.mean(max(snowAccum/10) - Hs_diff[index, :]*100), min(densCol[:, index]), max(densCol[:, index]), label = 'Median of Max', color = 'orange', linestyle = '-')

for i in range(3):
    plt.hlines(max(snowAccum/10) - Hs[index, i]*100-min(snowAccum)/10, min(densCol[:, index]), max(densCol[:, index]), label = minLabels[i], color = minColors[i], linestyle = '--')
    plt.hlines(max(snowAccum/10) - Hs_diff[index, i]*100-min(snowAccum)/10, min(densCol[:, index]), max(densCol[:, index]), label = maxLabels[i], color = maxColors[i], linestyle = ':')

plt.legend(loc = 'lower left', bbox_to_anchor=(1, 0.25))

#%% Resolution test
layerHeight = 5
index = round((layerHeight - 1)/1)

plt.figure(dpi = 400)
plt.subplot(1, 2, 1)
plt.plot(densCol[:, index], snowAccum/10-min(snowAccum)/10, label = 'Density')
plt.ylabel(r'$HS_{test}$ (cm)')
plt.xlabel(r'Density ($kg\,m^{-3}$)')
plt.subplot(1, 2, 2)
plt.plot(resolution[:, index]*1e9, snowAccum/10-min(snowAccum)/10, label = 'Resolution')
plt.xlabel(r'Resolution ($mm^3/particle$)')
plt.xticks(rotation=45)
plt.suptitle('Density Proflie and Resolution')

plt.figure(dpi = 400)
plt.subplot(1, 2, 1)
plt.plot(densCol[:, index], snowAccum/10-min(snowAccum)/10, label = 'Density')
plt.ylabel(r'$HS_{test}$ (cm)')
plt.xlabel(r'Density ($kg\,m^{-3}$)')
plt.subplot(1, 2, 2)
plt.plot(particle_counts[:, index], snowAccum/10-min(snowAccum)/10, label = 'Resolution')
plt.xlabel(r'Particle Count')
plt.xticks(rotation=45)
plt.suptitle('Density Proflie and Particle Count')