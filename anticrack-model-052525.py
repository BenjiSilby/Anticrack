# -*- coding: utf-8 -*-
"""
Created on Sun May  4 15:28:16 2025

@author: Jack
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
    
    # assign last row to the correct density
    # densCol[-1, j] = sum(mass[startVar:-1])/sum(volume[startVar:-1])
    # mean_SDI[-1, j] = np.mean(SDI[startVar:-1])
    # mean_Complexity[-1, j] = np.nanmean(complexity[startVar:-1])
    densCol[-1, j] = float('nan')
    mean_SDI[-1, j] = float('nan')
    mean_Complexity[-1, j] = float('nan')
    
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
       
            
#%% Plotting
layerHeight = 1
index = round((layerHeight - 1)/1)

plt.figure(dpi = 400)
plt.plot(densCol[:, index], snowAccum/10, label = 'Density')
plt.xlabel(r'Density ($kg\,m^{-3}$)')
plt.ylabel('Snow Height (cm)')
plt.title('Density across Snow Height '  + storm_dates[u])
plt.text(min(densCol[:, index]), max(snowAccum/10)-min(snowAccum)/10 - 0.5, f'Layer Height = {layerHeight} mm')

plt.figure(dpi = 400)
plt.plot(mm, wfs[:, :], 'bo', label = r'Min $\rho$')
plt.plot(mm, wfs_diff[:, :], 'ro', label = r'Max $\Delta \rho$')
plt.xlabel('Layer Heights')
plt.ylabel('SI')
plt.title('SI vs. Layer Heights Comparison ' + storm_dates[u])
#plt.legend()

plt.figure(dpi = 400)
plt.plot(mm, Hs[:, :], 'bo', label = r'Min $\rho$')
plt.plot(mm, Hs_diff[:, :], 'ro', label = r'Max $\Delta \rho$')
plt.xlabel('Layer Heights')
plt.ylabel('Slab Location')
plt.title('Slab Location vs. Layer Heights Comparison '  + storm_dates[u])
#plt.legend()
plt.show()

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
#%% Plot Stability using density of layers
layerHeight = 1
index = round((layerHeight - 1)/1)

plt.figure(dpi = 400)
plt.plot(layer_SI[:, index], snowAccum/10-min(snowAccum)/10, label = 'Slab Density')
plt.xlabel('Stability Index')
plt.ylabel('Snow Height (cm)')
plt.title('Density across Snow Height ' + storm_dates[u])
plt.vlines(1, min(snowAccum)/10-min(snowAccum)/10, max(snowAccum)/10-min(snowAccum)/10, label = 'SI = 1', color = 'red')
plt.legend()

#%% Stability Modelling Continued

plt.figure(dpi = 400)
plt.loglog(densCol[:, index], layer_SI[:, index], '.')
plt.ylabel('Stability Index')
plt.xlabel('Density (kg/m^3)')
plt.title('SI vs. Density '  + storm_dates[u])

fig = plt.figure(dpi = 400)
ax = plt.axes(projection='3d')
ax.plot3D(np.log10(densCol[:, index]), snowAccum/10 - min(snowAccum)/10, np.log10(layer_SI[:, index]), '.')

plt.title('SI vs. Density and Height '  + storm_dates[u])
ax.set_xlabel('log(Density)')
ax.set_ylabel('Snow Accum (cm)')
ax.set_zlabel('log(SI)')


# Step 1: Transform input data
log_density = np.log10(densCol[:, index]/rho_ice)
snow = (snowAccum / 10 - min(snowAccum) / 10)/(max(snowAccum)/10)  # Normalize or leave as-is
log_SI = np.log10(layer_SI[:, index])

# Step 2: Create a mask of finite (non-NaN and non-Inf) values
valid_mask = np.isfinite(log_density) & np.isfinite(snow) & np.isfinite(log_SI)

# Step 3: Filter inputs
log_density_clean = log_density[valid_mask]
snow_clean = snow[valid_mask]
log_SI_clean = log_SI[valid_mask]

# Step 4: Fit regression
X = np.column_stack((log_density_clean, snow_clean))
model = LinearRegression().fit(X, log_SI_clean)

# Get coefficients
a, b = model.coef_
c = model.intercept_

print(f"Fitted model: log10(SI) = {a:.3f} * log10(Density) + {b:.3f} * SnowAccum + {c:.3f}")

# Step 4: Plot data
fig = plt.figure(dpi=200)
ax = plt.axes(projection='3d')
ax.scatter(log_density, snow, log_SI, alpha=0.6, label='Data')

# Step 5: Create grid and compute surface
log_density_range = np.linspace(min(log_density), max(log_density), 30)
snow_range = np.linspace(snow.min(), snow.max(), 30)
log_density_grid, snow_grid = np.meshgrid(log_density_range, snow_range)

# Predict log_SI values
log_SI_grid = a * log_density_grid + b * snow_grid + c

# Step 6: Plot surface
surf = ax.plot_surface(log_density_grid, snow_grid, log_SI_grid,
                alpha=0.5, cmap='plasma', edgecolor='none')

# Labels
ax.set_xlabel(r'$log(\rho/\rho_{ice})$')
ax.set_ylabel(r'$H_s/max(H_s)$')
ax.set_zlabel('log(SI)')
plt.title(r'$SI \approx F(\rho, H_s)$ ' + storm_dates[u])
plt.show()

#%% creates an animation and takes a long time
# def update(frame):
#     ax.view_init(elev=30, azim=frame)
#     return fig,

# # Create animation: frames from 0 to 359 degrees
# anim = FuncAnimation(fig, update, frames=np.arange(0, 360, 5), blit=False)

# # Save as GIF (needs Pillow)
# anim.save('rotating_surface1.gif', writer='pillow', fps=15)

# plt.close(fig)  # Close the figure after saving














#%% Repeat but as a function of slab density
layerHeight = 1
index = round((layerHeight - 1)/1)

plt.figure(dpi = 400)
plt.plot(layer_slab_density[:, index], snowAccum/10-min(snowAccum)/10, label = 'SI')
plt.xlabel(r'Slab Density (kg/m^3)')
plt.ylabel('Snow Height (cm)')
plt.title('Density across Snow Height ' + storm_dates[u])
plt.legend(loc = 'upper left')
#%% 
plt.figure(dpi = 400)
plt.loglog(layer_slab_density[:, index], layer_SI[:, index], '.')
plt.ylabel('Stability Index')
plt.xlabel('Slab Density (kg/m^3)')
plt.title('SI vs. Slab Density '  + storm_dates[u])

fig = plt.figure(dpi = 400)
ax = plt.axes(projection='3d')
ax.plot3D(np.log10(layer_slab_density[:, index]), snowAccum/10 - min(snowAccum)/10, np.log10(layer_SI[:, index]), '.')

plt.title('SI vs. Slab Density and Height '  + storm_dates[u])
ax.set_xlabel('log(Density)')
ax.set_ylabel('Snow Accum (cm)')
ax.set_zlabel('log(SI)')


# Step 1: Transform input data
log_density = np.log10(layer_slab_density[:, index]/rho_ice)        # Normalized vs rho_ice
snow = (snowAccum / 10 - min(snowAccum) / 10)/(max(snowAccum)/10)  # Normalize or leave as-is
log_SI = np.log10(layer_SI[:, index])

# Step 2: Create a mask of finite (non-NaN and non-Inf) values
valid_mask = np.isfinite(log_density) & np.isfinite(snow) & np.isfinite(log_SI)

# Step 3: Filter inputs
log_density_clean = log_density[valid_mask]
snow_clean = snow[valid_mask]
log_SI_clean = log_SI[valid_mask]

# Step 4: Fit regression
X = np.column_stack((log_density_clean, snow_clean))
model = LinearRegression().fit(X, log_SI_clean)

# Get coefficients
a, b = model.coef_
c = model.intercept_

print(f"Fitted model: log10(SI) = {a:.3f} * log10(Density) + {b:.3f} * SnowAccum + {c:.3f}")

# Step 4: Plot data
fig = plt.figure(dpi=200)
ax = plt.axes(projection='3d')
ax.scatter(log_density, snow, log_SI, alpha=0.6, label='Data')

# Step 5: Create grid and compute surface
log_density_range = np.linspace(min(log_density), max(log_density), 30)
snow_range = np.linspace(snow.min(), snow.max(), 30)
log_density_grid, snow_grid = np.meshgrid(log_density_range, snow_range)

# Predict log_SI values
log_SI_grid = a * log_density_grid + b * snow_grid + c

# Step 6: Plot surface
surf = ax.plot_surface(log_density_grid, snow_grid, log_SI_grid,
                alpha=0.5, cmap='plasma', edgecolor='none')

# Labels
ax.set_xlabel('log10(Density)')
ax.set_ylabel('Snow Accum (cm)')
ax.set_zlabel('log10(SI)')
plt.title('Regression Surface: log10(SI) ~ log10(Density) + SnowAccum')


#%%
# def update(frame):
#     ax.view_init(elev=30, azim=frame)
#     return fig,

# # Create animation: frames from 0 to 359 degrees
# anim = FuncAnimation(fig, update, frames=np.arange(0, 360, 5), blit=False)

# # Save as GIF (needs Pillow)
# anim.save('rotating_surface2.gif', writer='pillow', fps=15)

# plt.close(fig)  # Close the figure after saving
#%% Wg study
layerHeight = 1
index = round((layerHeight - 1)/1)

plt.figure(dpi = 400)
plt.plot(layer_slab_density[:, index]/rho_ice, layer_SI[:, index], '.')
plt.xlabel(r'$rho_{slab}/\rho_{ice}$')
plt.ylabel('SI')
plt.title('SI vs Slab Density')

plt.figure(dpi = 400)
plt.plot(layer_slab_density[:, index]/rho_ice, layer_Wg[:, index], '.')
plt.xlabel(r'$rho_{slab}/\rho_{ice}$')
plt.ylabel('Wg')
plt.title('Wg vs. Slab Density')


#%% Save Data

np.save('feb07density', densCol)
np.save('feb07si', layer_SI)
np.save('feb07snowAccum', snowAccum)

#%% plots from meeting 5/29/25
layerHeight = 1
index = round((layerHeight - 1)/1)

plt.figure(dpi = 400)
plt.plot(densCol[:, index], snowAccum-min(snowAccum), label = "Layer Density")
plt.plot(layer_slab_density[:, index], snowAccum-min(snowAccum), label = "Slab Density")
plt.plot(layer_Wg[:, index]*1000, snowAccum-min(snowAccum), label = "Wg*1000")
plt.xlabel('Density')
plt.ylabel('Snow Accum (mm)')
plt.title('Density/Weight across Snow Height '  + storm_dates[u])
plt.legend(loc = 'lower left')

# using particle density
log_density = np.log(densCol[:, index]/rho_ice)        # Normalized vs rho_ice
snow_norm = (snowAccum / 10 - min(snowAccum) / 10)/(max(snowAccum)/10)  # Normalize or leave as-is
log_SI = np.log(layer_SI[:, index])

plt.figure(dpi = 400)
plt.scatter(log_density, snow_norm, c = log_SI, cmap = 'plasma')
plt.colorbar(label=r'$log(SI)$')
plt.xlabel(r'$log(\rho/\rho_{ice})$')
plt.ylabel(r'$H_s/max(H_s)$')
plt.title(r'$\log(\rho/\rho_{ice})$ vs. $H_s/\max(H_s)$ Colored by $\log(\mathrm{SI})$ ' + storm_dates[u])

# using slab density
log_density = np.log(layer_slab_density[:, index]/rho_ice)        # Normalized vs rho_ice
snow_norm = (snowAccum / 10 - min(snowAccum) / 10)/(max(snowAccum)/10)  # Normalize or leave as-is
log_SI = np.log(layer_SI[:, index])

plt.figure(dpi = 400)
plt.scatter(log_density, snow_norm, c = log_SI, cmap = 'plasma')
plt.colorbar(label=r'$log(SI)$')
plt.xlabel(r'$log(\rho_{slab}/\rho_{ice})$')
plt.ylabel(r'$H_s/max(H_s)$')
plt.title(r'$\log(\rho_{slab}/\rho_{ice})$ vs. $H_s/\max(H_s)$ Colored by $\log(\mathrm{SI})$ ' + storm_dates[u])

# using particle density and Wg
log_density = np.log(densCol[:, index]/rho_ice)        # Normalized vs rho_ice
snow_norm = (snowAccum / 10 - min(snowAccum) / 10)/(max(snowAccum)/10)  # Normalize or leave as-is
log_Wg = np.log(layer_Wg[:, index])

plt.figure(dpi = 400)
plt.scatter(log_density, snow_norm, c = log_Wg, cmap = 'plasma')
plt.colorbar(label=r'$log(W_g)$')
plt.xlabel(r'$log(\rho/\rho_{ice})$')
plt.ylabel(r'$H_s/max(H_s)$')
plt.title(r'$\log(\rho/\rho_{ice})$ vs. $H_s/\max(H_s)$ Colored by $\log(\mathrm{W_g})$ ' + storm_dates[u])

# using slab density and Wg
log_density = np.log(layer_slab_density[:, index]/rho_ice)        # Normalized vs rho_ice
snow_norm = (snowAccum / 10 - min(snowAccum) / 10)/(max(snowAccum)/10)  # Normalize or leave as-is
log_Wg = np.log(layer_Wg[:, index])

plt.figure(dpi = 400)
plt.scatter(log_density, snow_norm, c = log_Wg, cmap = 'plasma')
plt.colorbar(label=r'$log(W_g)$')
plt.xlabel(r'$log(\rho_{slab}/\rho_{ice})$')
plt.ylabel(r'$H_s/max(H_s)$')
plt.title(r'$\log(\rho_{slab}/\rho_{ice})$ vs. $H_s/\max(H_s)$ Colored by $\log(\mathrm{W_g})$ ' + storm_dates[u])
#%%

grad_slab_denisty = np.gradient(layer_slab_density[:, index], snowAccum-min(snowAccum))
# Replace zeros with NaN
grad_slab_denisty = np.where(grad_slab_denisty == 0, np.nan, grad_slab_denisty)
plt.figure(dpi = 400)
plt.plot(layer_slab_density[:, index], snowAccum-min(snowAccum), label = r'$\rho_{slab}$')
plt.plot(grad_slab_denisty, snowAccum - min(snowAccum), label = r'$\nabla \rho_{slab}$')
plt.xlabel(r'Density (kg/m^3)')
plt.ylabel(r'Snow Accum (mm)')
plt.title('Gradient of Slab Density '+ storm_dates[u] )
plt.legend()

grad_Wg = np.gradient(layer_Wg[:, index], snowAccum-min(snowAccum))
# Replace zeros with NaN
grad_Wg = np.where(grad_Wg == 0, np.nan, grad_Wg)
plt.figure(dpi = 400)
plt.plot(layer_Wg[:, index], snowAccum-min(snowAccum), label = r'$W_g$')
plt.plot(grad_Wg, snowAccum - min(snowAccum), label = r'$\nabla W_g$')
plt.xlabel(r'Density (kg/m^3)')
plt.ylabel(r'Snow Accum (mm)')
plt.title('Gradient of Slab Weight ' + storm_dates[u])
plt.legend()

# Attempt smoothing data
smoothed_slabDens = pd.Series(layer_slab_density[:, index]).rolling(window=15000, center=True).mean()
grad_slab_denisty = np.gradient(smoothed_slabDens, snowAccum-min(snowAccum))
# Replace zeros with NaN
grad_slab_denisty = np.where(grad_slab_denisty == 0, np.nan, grad_slab_denisty)
plt.figure(dpi = 400)
plt.plot(layer_slab_density[:, index], snowAccum-min(snowAccum), label = r'$\rho_{slab}$')
plt.plot(grad_slab_denisty, snowAccum - min(snowAccum), label = r'$\nabla \rho_{slab}$')
plt.xlabel(r'Density (kg/m^3)')
plt.ylabel(r'Snow Accum (mm)')
plt.title('Gradient of Slab Density ' + storm_dates[u])
plt.legend()

smoothed_Wg = pd.Series(layer_Wg[:, index]).rolling(window=15000, center=True).mean()
grad_Wg = np.gradient(smoothed_Wg, snowAccum-min(snowAccum))
# Replace zeros with NaN
grad_Wg = np.where(grad_Wg == 0, np.nan, grad_Wg)
plt.figure(dpi = 400)
plt.plot(layer_Wg[:, index], snowAccum-min(snowAccum), label = r'$W_g$')
plt.plot(grad_Wg, snowAccum - min(snowAccum), '.', label = r'$\nabla W_g$')
plt.xlabel(r'Density (kg/m^3)')
plt.ylabel(r'Snow Accum (mm)')
plt.title('Gradient of Slab Weight ' + storm_dates[u])
plt.legend()

#%% try unique values
unique_slab_density, unique_idxs = np.unique(layer_slab_density[:, index], return_index=True)
grad_slab_denisty = np.gradient(unique_slab_density, snowAccum[unique_idxs])
plt.figure(dpi = 400)
plt.plot(layer_slab_density[:, index], snowAccum-min(snowAccum), label = r'$\rho_{slab}$')
plt.plot(grad_slab_denisty, snowAccum[unique_idxs] - min(snowAccum), label = r'$\nabla \rho_{slab}$')
plt.xlabel(r'Density (kg/m^3)')
plt.ylabel(r'Snow Accum (mm)')
plt.title('Gradient of Slab Density ' + storm_dates[u])
plt.legend()

unique_Wg, unique_idxs = np.unique(layer_Wg[:, index], return_index=True)
grad_Wg = np.gradient(unique_Wg, snowAccum[unique_idxs])
plt.figure(dpi = 400)
plt.plot(layer_Wg[:, index], snowAccum-min(snowAccum),  label = r'$W_g$')
plt.plot(grad_Wg, snowAccum[unique_idxs] - min(snowAccum), label = r'$\nabla W_g$')
plt.xlabel(r'Density (kg/m^3)')
plt.ylabel(r'Snow Accum (mm)')
plt.title('Gradient of Slab Weight ' + storm_dates[u])
plt.legend()

#%% Meeting 06/05/2025
layerHeight = 1
index = round((layerHeight - 1)/1)

plt.figure(dpi = 400)
plt.plot(snowAccum - min(snowAccum), layer_Wg[:, index], label = r'$W_g$ vs. $H_{snow}$')

# Get mask of unique values (keeping first occurrence)
_, idx = np.unique(layer_Wg[:, index], return_index=True)
idx = np.sort(idx)
layer_Wg_unique = layer_Wg[idx, index]
height_unique = snowAccum[idx] - min(snowAccum)

plt.plot(height_unique, layer_Wg_unique, label = r'Unique $W_g$ vs. $H_{snow}$')
plt.xlabel(r'Snow Accum (mm)')
plt.ylabel(r'$W_g$ (Pa*m)')
plt.title(r'$W_g$ vs. Snow Height')
plt.legend()


dy_dx = np.gradient(layer_Wg_unique, height_unique)
d2y_dx2 = np.gradient(dy_dx, height_unique)
plt.figure(dpi = 400)
# Original function
fig, axs = plt.subplots(3, 1, figsize=(8, 8), dpi=150, sharex=True)
axs[0].plot(height_unique, layer_Wg_unique, label= r'$W_g(x), x = H_{snow}$', color='blue')
axs[0].set_ylabel(r'$W_g$')
axs[0].legend()
axs[0].grid(True)
axs[0].set_title(r'$W_g$ vs. Snow Height')

# First derivative
axs[1].plot(height_unique, dy_dx, label= r'$dW_gh/dz$', color='orange')
axs[1].set_ylabel('First Derivative')
axs[1].legend()
axs[1].grid(True)

# Second derivative
axs[2].plot(height_unique, d2y_dx2, label=r'$d^2 W_gh/dz^2$', color='green')
axs[2].set_ylabel('Second Derivative')
axs[2].set_xlabel(r'Snow Height (mm)')
axs[2].legend(loc = "upper right")
axs[2].grid(True)

# Get mask of unique values (keeping first occurrence)
_, idx = np.unique(layer_slab_density[:, index], return_index=True)
idx = np.sort(idx)
layer_rho_unique = layer_slab_density[idx, index]
height_unique = snowAccum[idx] - min(snowAccum)

grad_layer_rho = np.gradient(layer_rho_unique, height_unique)

plt.figure(dpi = 400)
# Create subplots: 1 row, 2 columns
fig, axs = plt.subplots(1, 2, figsize=(12, 5), dpi=150, sharey=True)

# Plot 1: Slab Density vs Snow Accum
axs[0].plot(layer_slab_density[:, index], snowAccum - min(snowAccum), label=r'$\rho_{slab}$')
axs[0].set_xlabel(r'Density (kg/m$^3$)')
axs[0].set_ylabel(r'Snow Accum (mm)')
axs[0].set_title('Slab Density')
axs[0].legend()
axs[0].grid(True)

# Plot 2: Gradient of Slab Density vs Height
axs[1].plot(grad_layer_rho, height_unique, label=r'$\nabla \rho_{slab}$')
axs[1].set_xlabel(r'Density Gradient (kg/m$^3/mm$)')  # Adjust units if needed
axs[1].set_title('Gradient of Slab Density')
axs[1].legend()
axs[1].grid(True)

# Overall figure title
fig.suptitle('Slab Density and Its Gradient: ' + storm_dates[u], fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

#%% plot Pi groups
plt.figure(dpi = 400)
plt.plot(non_dim_rho[:, index], non_dim_Wf[:, index], '.')
plt.xlabel(r'$\rho_{slab}/\rho_{ice}$')
plt.ylabel(r'$Wf/(\rho_{ice}*g*H_l^2)$')
plt.title('Non-dim Fracture Energy vs Non-dim Density')

plt.figure(dpi = 400)
plt.plot(non_dim_rho[:, index], non_dim_E[:, index], '.')
plt.xlabel(r'$\rho_{slab}/\rho_{ice}$')
plt.ylabel(r'$E/(\rho_{ice}*g*H_l)$')
plt.title('Non-dim Elastic Modulus vs Non-dim Density')

plt.figure(dpi = 400)
plt.plot(non_dim_rho[:, index], non_dim_Ki[:, index], '.')
plt.xlabel(r'$\rho_{slab}/\rho_{ice}$')
plt.ylabel(r'$Ki/(\rho_{ice}*g*H_l^{3/2})$')
plt.title('Non-dim Fracture Toughness vs Non-dim Density')

plt.figure(dpi = 400)
plt.loglog(non_dim_rho[:, index], non_dim_Wf[:, index], '.')
plt.xlabel(r'$log \rho_{slab}/\rho_{ice}$')
plt.ylabel(r'$log Wf/(\rho_{ice}*g*H_l^2)$')
plt.title('Non-dim Fracture Energy vs Non-dim Density')

plt.figure(dpi = 400)
plt.loglog(non_dim_rho[:, index], non_dim_E[:, index], '.')
plt.xlabel(r'$log \rho_{slab}/\rho_{ice}$')
plt.ylabel(r'$log E/(\rho_{ice}*g*H_l)$')
plt.title('Non-dim Elastic Modulus vs Non-dim Density')

plt.figure(dpi = 400)
plt.loglog(non_dim_rho[:, index], non_dim_Ki[:, index], '.')
plt.xlabel(r'$log \rho_{slab}/\rho_{ice}$')
plt.ylabel(r'$log Ki/(\rho_{ice}*g*H_l^{3/2})$')
plt.title('Non-dim Fracture Toughness vs Non-dim Density')

#%%
plt.figure(dpi = 400)
plt.plot(layer_slab_density[:, index]/917, layer_KI[:, index])

plt.figure(dpi = 400)
plt.plot(layer_slab_density[:, index]/917, layer_E[:, index])

plt.figure(dpi = 400)
plt.plot(layer_slab_density[:, index]/917, pow(layer_KI[:, index], 2)/layer_E[:, index])

#%%
h = layerHeight/1000
plt.figure(dpi = 400)
plt.loglog(layer_slab_density[:, index]/917, layer_SI[:, index]*layer_Wg[:, index]*h)
plt.xlabel(r'$\rho_{slab}/\rho_{ice}$')
plt.ylabel(r'Fracture Energy $(W_f)$')
plt.title('Fracture Energy vs. Density')

# Step 1: Transform input data
log_density = np.log10(layer_slab_density[:, index]/rho_ice)        # Normalized vs rho_ice
log_SI = np.log10(layer_SI[:, index]*layer_Wg[:, index]*h)

# Step 2: Create a mask of finite (non-NaN and non-Inf) values
valid_mask = np.isfinite(log_density) & np.isfinite(log_SI)

# Step 3: Filter inputs
log_density_clean = log_density[valid_mask]
log_SI_clean = log_SI[valid_mask]

# Step 4: Fit regression
b, a = np.polyfit(log_density_clean, log_SI_clean, 1)

plt.figure(dpi = 400)
plt.loglog(layer_slab_density[:, index]/917, layer_SI[:, index]*layer_Wg[:, index]*h, label = "Calculated")
plt.loglog(layer_slab_density[:, index]/917, (10**a)*(layer_slab_density[:, index]/917)**b, '--', label = "Modeled")
plt.xlabel(r'$\rho_{slab}/\rho_{ice}$')
plt.ylabel(r'Fracture Energy $(W_f)$')
plt.title('Fracture Energy vs. Density')
plt.legend()

# plot Pi groups
mask = (non_dim_Wf < 0.2) & (non_dim_Wf > 0.027)
plt.figure(dpi = 400)
plt.loglog(non_dim_rho[:, index], non_dim_Wf[:, index], '.', label = "All Data")
plt.loglog(non_dim_rho[mask[:, index], index], non_dim_Wf[mask[:, index], index], '.', label = "Linear Portion")

# linear fit on log of data
b, a =  np.polyfit(np.log10(non_dim_rho[mask[:, index], index]), np.log10(non_dim_Wf[mask[:, index], index]), 1)
plt.plot(non_dim_rho[:, index], (10**a)*(non_dim_rho[:, index]**b), '-', label = "Modeled Values")

plt.xlabel(r'$\rho_{slab}/\rho_{ice}$')
plt.ylabel(r'$Wf/(\rho_{slab}*g*H_l^2)$')
plt.title('Non-dim Fracture Energy vs Non-dim Density')
x_pos = min(non_dim_rho[:, index])
y_pos = 10*np.nanmean(non_dim_Wf[:, index])
plt.text(x_pos, y_pos, f"Wf* = (10^{a:.4})*(rho/rho_ice)^{b:.4}", fontsize = 8)
plt.legend()

mask = (non_dim_E < 65000) & (non_dim_E > 17000)
plt.figure(dpi = 400)
plt.loglog(non_dim_rho[:, index], non_dim_E[:, index], '.', label = "All Data")
plt.loglog(non_dim_rho[mask[:, index], index], non_dim_E[mask[:, index], index], '.', label = "Linear Portion")

# linear fit on log of data
b, a =  np.polyfit(np.log10(non_dim_rho[mask[:, index], index]), np.log10(non_dim_E[mask[:, index], index]), 1)
plt.plot(non_dim_rho[:, index], (10**a)*(non_dim_rho[:, index]**b), '-', label = "Modeled Values")

plt.xlabel(r'$\rho_{slab}/\rho_{ice}$')
plt.ylabel(r'$E/(\rho_{slab}*g*H_l)$')
plt.title('Non-dim Elastic Modulus vs Non-dim Density')
x_pos = min(non_dim_rho[:, index])
y_pos = 10*np.nanmean(non_dim_E[:, index])
plt.text(x_pos, y_pos, f"E* = (10^{a:.4})*(rho/rho_ice)^{b:.4}", fontsize = 8)
plt.legend()

mask = (non_dim_Ki < 10**2) & (non_dim_Ki > 20)
plt.figure(dpi = 400)
plt.loglog(non_dim_rho[:, index], non_dim_Ki[:, index], '.', label = "All Data")
plt.loglog(non_dim_rho[mask[:, index], index], non_dim_Ki[mask[:, index], index], '.', label = "Linear Portion")

# linear fit on log of data
b, a =  np.polyfit(np.log10(non_dim_rho[mask[:, index], index]), np.log10(non_dim_Ki[mask[:, index], index]), 1)
plt.plot(non_dim_rho[:, index], (10**a)*(non_dim_rho[:, index]**b), '-', label = "Modeled Values")

plt.xlabel(r'$\rho_{slab}/\rho_{ice}$')
plt.ylabel(r'$Ki/(\rho_{slab}*g*H_l^{3/2})$')
plt.title('Non-dim Fracture Toughness vs Non-dim Density')
x_pos = min(non_dim_rho[:, index])
y_pos = 10*np.nanmean(non_dim_Ki[:, index])
plt.text(x_pos, y_pos, f"Ki* = (10^{a:.4})*(rho/rho_ice)^{b:.4}", fontsize = 8)
plt.legend()

#%% Sensitivity study

# create a range of possible densities
simulated_density = np.linspace(10, 200, 50)

alpha = 2.5826446280991737
beta = 1.86
# initial Wf
Wf = alpha*(simulated_density/917)**beta

plt.figure(dpi = 400)
plt.plot(simulated_density, Wf)
plt.plot(simulated_density, 1.3*(simulated_density/917)**(2*1.3), label = "complexity")
plt.xlabel("Density (Kg)")
plt.ylabel("Fracture Energy (kPa*m)")
plt.title("Fracture Energy Sensitity Study")
plt.text(np.mean(simulated_density), max(Wf), f"Alpha = {alpha:.3f}, beta = {beta}")
plt.legend()

alpha_range = np.linspace(1, alpha, 10)
Wf_VariedA = np.zeros([len(simulated_density), len(alpha_range)])
labels = []

plt.figure(dpi = 400)
for i in range(len(alpha_range)):
    Wf_VariedA[:, i] = alpha_range[i]*(simulated_density/917)**beta
    labels += [str(round(alpha_range[i], 3))]
    plt.plot(simulated_density, Wf_VariedA[:, i], label = labels[i])

plt.plot(simulated_density, 1.3*(simulated_density/917)**(2*1.3), label = "complexity")
plt.xlabel("Density (Kg)")
plt.ylabel("Fracture Energy (KPa*m)")
plt.title("Fracture Energy Alpha Sensitity Study")
plt.legend()
plt.text(np.mean(simulated_density), max(Wf), f"Beta = {beta}")

beta_range = np.linspace(beta, 3, 10)
Wf_VariedB = np.zeros([len(simulated_density), len(beta_range)])
labels = []

plt.figure(dpi = 400)
for i in range(len(beta_range)):
    Wf_VariedB[:, i] = alpha*(simulated_density/917)**beta_range[i]
    labels += [str(round(beta_range[i], 3))]
    plt.plot(simulated_density, Wf_VariedB[:, i], label = labels[i])

plt.plot(simulated_density, 1.3*(simulated_density/917)**(2*1.3), label = "complexity")
plt.xlabel("Density (Kg)")
plt.ylabel("Fracture Energy (kPa*m)")
plt.title("Fracture Energy Beta Sensitity Study")
plt.legend()
plt.text(np.mean(simulated_density), max(Wf), f"Alpha = {alpha:.3f}")

#all alpha and all beta
labels = []

# list of different line styles
styles = [
    '-',                # solid
    '--',               # dashed
    '-.',               # dash-dot
    ':',                # dotted
    (0, (1, 1)),        # densely dotted
    (0, (5, 5)),        # loosely dashed
    (0, (3, 5, 1, 5)),  # dash-dot with long gaps
    (0, (5, 1)),        # long dash + short gap
    (0, (3, 1, 1, 1)),  # dot-dash-dot-dash
    (0, (2, 4)),        # short dash + long gap
]

# colors
colors = [
    'blue',
    'orange',
    'green',
    'red',
    'purple',
    'brown',
    'pink',
    'gray',
    'olive',
    'cyan'
]

alpha_range = np.linspace(0, alpha, 5)
beta_range = np.linspace(beta, 4, 5)
Wf_Varied = np.zeros([len(simulated_density), len(alpha_range), len(beta_range)])

plt.figure(dpi = 400)
for j in range(len(alpha_range)):
    for i in range(len(beta_range)):
        Wf_Varied[:, j, i] = alpha_range[j]*(simulated_density/917)**beta_range[i]
        current = "alpha = " + str(round(alpha_range[j], 3))
        if i == 0:
            plt.plot(simulated_density, Wf_Varied[:, j, i], linestyle = styles[j], label = current, color = colors[j])
        else:
            plt.plot(simulated_density, Wf_Varied[:, j, i], linestyle = styles[j], color = colors[j])

plt.xlabel("Density (Kg)")
plt.ylabel("Fracture Energy (kPa*m)")
plt.title("Fracture Energy Alpha and Beta Sensitity Study")
plt.legend()
plt.tight_layout()

def power_function(alpha, beta, rho):
    Wf = alpha*(rho/917)**beta
    return Wf

A, B = np.meshgrid(alpha_range, beta_range)
Z = power_function(A, B, 80)

plt.figure(dpi = 400)
contour = plt.contourf(A, B, Z, levels=50, cmap='viridis')
plt.colorbar(contour, label= r'Fracture Energy, $\rho=80 kg/m^3$')
plt.xlabel('Alpha')
plt.ylabel('Beta')
plt.title('Fracture Energy Sensitivity to Alpha and Beta')
plt.grid(True)
plt.show()

#%% Stability continued

# 5 possibly good storm (02/14/2025 and 04/01/25 PM are unrealistic)
rho_slabs = np.zeros(6)
Hs = np.zeros(6)
complexs = np.zeros(6)
SDIs = np.zeros(6)
# calculated using the complexity curve
Wfs_SI = np.zeros(6)
Wgs_SI = np.zeros(6)
SI = np.zeros(6)

# using complexity again
Wfs_SI2 = np.zeros(6)
Wgs_SI2 = np.zeros(6)
SI2 = np.zeros(6)

# 6pt fit
Wfs_SI3 = np.zeros(6)
Wgs_SI3 = np.zeros(6)
SI3 = np.zeros(6)

# gather final heights from Mode I tiltboards in cm
final_heights = np.array([0.5, 11.5, 13, 4, 2, 7.5])

# convert to mm
final_heights = 10*final_heights

storm_list = ['DEID_Particle_2025-03-31_15-05-01.csv',
              'DEID_Particle_2025-03-13_09-08-55.csv',
              'DEID_Particle_2025-03-03_18-47-33.csv',
              'DEID_Particle_feb19_total.csv',
              'DEID_Particle_2025-02-13_08-44-19.csv',
              'DEID_Particle_2024-02-07_12-39-54.csv']

# verify the cutoff time for 04-01-evening
cutoff_times = ['2025-04-01 06:47',
                '2025-03-13 17:41',
                '2025-03-04 20:42',
                '2025-02-20 18:30',
                '2025-02-14 12:23',
                '2025-02-07 16:34']

storm_dates = ['2025-04-01 AM',
                '2025-03-13',
                '2025-03-04',
                '2025-02-20',
                '2025-02-14',
                '2025-02-07']


# Feb 19 has a different column name for FBF Snow Accum
snow_accum_col_name = ['FBF Snow Accum (mm)',
                       'FBF Snow Accum (mm)',
                       'FBF Snow Accum (mm)',
                       'Snow_Accum_mm',
                       'FBF Snow Accum (mm)',
                       'FBF Snow Accum (mm)']

indexVals = [0, 1, 2, 3, 5]
indexVals = [0, 1, 2, 3, 4, 5]
for u in indexVals:
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
        startTime = pd.to_datetime('2025-03-31 19:23')
    if u ==1:
        startTime = pd.to_datetime('2025-03-12 15:56')
    if u == 2:
        snowAccum = particleData[snow_accum_col_name[u]]
        snowAccum = snowAccum.to_numpy()
        lowerSnowAccum = max(snowAccum) - 24*10
        particleData = particleData[(particleData[snow_accum_col_name[u]] >= lowerSnowAccum)]
        #startTime = pd.to_datetime('2025-03-04 13:10')
    
    # Filter the DataFrame
    particleData = particleData[(particleData["Time"] < cutoff_time) & (particleData["Time"] >= startTime)]
    
    # get snow accum as column
    snowAccum = np.array(particleData[snow_accum_col_name[u]])
    snowAccum = snowAccum - min(snowAccum)
                             
    # calculate the absolute difference with respect to snow height
    abs_diff = abs(snowAccum - final_heights[u])
    idx = np.argmin(abs_diff)
    
    # using min to find H
    Hs[u] = max(snowAccum) - snowAccum[idx]
    
    # get rho_slab
    mass = particleData["Mass"]
    volume = particleData["Volume HFD"]
    rho_slabs[u] = sum(mass[idx:-1])/sum(volume[idx:-1])
    
    # gather complexities
    complexity = particleData['Complexity']
    comlexity = complexity.to_numpy()
    SDI = particleData['SDI']
    SDI = SDI.to_numpy()
    # filter complexity by setting less than/= 1 equal to nan these 
    # particles hit the resolution limit of the thermal camera
    complexity = np.array([float('nan') if x <= 1 else x for x in complexity])
    complexs[u] = np.nanmean(complexity[idx:-1]) 
    SDIs[u] = np.nanmean(SDI[idx:-1])
    Wfs_SI[u] = complexs[u]*(rho_slabs[u]/917)**(2*complexs[u])        #kPa m 
    Wgs_SI[u] = (rho_slabs[u]*9.81*Hs[u]*(5/1000))*pow(10, -6)         #kPa m
    SI[u] = Wfs_SI[u]/Wgs_SI[u]
    
    Wfs_SI2[u] = (10**-complexs[u])*(rho_slabs[u]/917)**(complexs[u])        #kPa m 
    Wgs_SI2[u] = (rho_slabs[u]*9.81*Hs[u]*(5/1000))*pow(10, -6)         #kPa m
    SI2[u] = Wfs_SI[u]/Wgs_SI[u]
    
    Wfs_SI3[u] = 0.07061183868583236*(rho_slabs[u]/917)**1.49608036286816
    Wgs_SI3[u] = (rho_slabs[u]*9.81*Hs[u]*(5/1000))*pow(10, -6)         #kPa m
    SI3[u] = Wfs_SI[u]/Wgs_SI[u]
    
    
    
    
#%% Plotting
# convert Hs to meters 
Hs_mask = Hs[indexVals]
rho_slabs_mask = rho_slabs[indexVals]

# get the pressure associated with the slab and convert from Pa to kPa and H from mm to meters
Wgs = rho_slabs_mask*9.81*Hs_mask*pow(10, -6)

# convert to fracture energy with little h (m)
hs = np.linspace(1/1000, 1/100, 10)

# create a range of possible densities
simulated_density = np.linspace(10, 150, 50)

alpha = 2.5826446280991737
beta = 1.86
complexity = particleData['Complexity']
comlexity = complexity.to_numpy()
# filter complexity by setting less than/= 1 equal to nan these 
# particles hit the resolution limit of the thermal camera
complexity = np.array([float('nan') if x <= 1 else x for x in complexity])

SDI = particleData['SDI']
SDI = SDI.to_numpy()

Wf = alpha*(simulated_density/917)**beta

colors = [
    'blue',
    'orange',
    'green',
    'red',
    'purple',
    'brown',
    'pink',
    'gray',
    'olive',
    'cyan'
]

plt.figure(dpi = 400)
plt.plot(simulated_density, Wf)
for i in range(len(hs)):
    current = "h = " + str(round(hs[i], 3))
    plt.scatter(rho_slabs_mask, Wgs*hs[i], color = colors[i], label = current)
plt.xlabel(r"Density ($kg\,m^{-3}$)")
plt.ylabel(r"Critical Fracture Energy ($kPa\,m$)")
plt.title(r"Critical Fracture Energy and $W_gh$ vs. Density")
plt.text(np.mean(simulated_density), max(Wf), f"Alpha = {alpha:.3f}, beta = {beta}")
plt.legend()

plt.figure(dpi=400)
plt.scatter(rho_slabs_mask, Wgs*0.01)
plt.xlabel(r"Density ($kg\,m^{-3}$)")
plt.ylabel(r"Critical Fracture Energy ($kPa\,m$)")
plt.title(r"$W_gh$ vs Density")
# rho_clean = np.array([rho_slabs_mask[4], rho_slabs_mask[2], rho_slabs_mask[3]])
# Wgs_clean = np.array([Wgs[4], Wgs[2], Wgs[3]])
# get fitting values
b, a = np.polyfit(np.log10(rho_slabs_mask/917), np.log10(Wgs*0.01), 1)

alpha = np.nanmean(complexity[idx:-1])      # complexity? average slab complexity
beta = 2*alpha                              # 2*complexity?
# initial Wf

alpha2 = (10**(-np.nanmean(complexity[idx:-1])))
beta2 = np.nanmean(complexity[idx:-1])

plt.figure(dpi = 400)
plt.loglog(rho_slabs_mask, Wgs*0.01, '.', label = r"$W_gh$")
plt.loglog(rho_slabs_mask, (10**a)*pow(rho_slabs_mask/917, b), label = r"$0.071(\rho_s/\rho_{ice})^{1.50}$")
plt.loglog(rho_slabs_mask, (alpha)*pow(rho_slabs_mask/917, beta), label = r"$Complexity(\rho_s/\rho_{ice})^{2*Complexity}$")
plt.loglog(rho_slabs_mask, alpha2*pow(rho_slabs_mask/917, beta2), label = r"$(10^{-Complexity})(\rho_s/\rho_{ice})^{Complexity}$")
plt.xlabel(r"Density ($kg\,m^{-3}$)")
plt.ylabel(r"Critical Fracture Energy ($kPa\,m$)")
plt.title("Critical Fracture Energy vs Density")
plt.legend()

Wf = alpha*(simulated_density/917)**beta
Wf2 = alpha2*(simulated_density/917)**beta2
Wf3 = (10**a)*pow(simulated_density/917, b)

plt.figure(dpi = 400)
plt.plot(simulated_density, Wf, label = r"$Complexity(\rho_s/\rho_{ice})^{2*Complexity}$")
plt.plot(simulated_density, Wf2, label = r"$(10^{-Complexity})(\rho_s/\rho_{ice})^{Complexity}$")
plt.scatter(rho_slabs_mask, Wgs*0.01, color = "green", label = r"$W_gh$")
plt.plot(simulated_density, (10**a)*pow(simulated_density/917, b), label = r"$0.071(\rho_s/\rho_{ice})^{1.50}$")
plt.legend()
plt.xlabel(r"Density ($kg\,m^{-3}$)")
plt.ylabel(r"Critical Fracture Energy ($kPa\,m$)")
plt.title(r"Critical Fracture energy derived from $W_gh$ vs Density")
#%%
alpha = np.mean(complexs[indexVals])      # complexity? average slab complexity
beta = 2*alpha  
# test all the complexs 
Wf_modeled = np.zeros(6)
Wf_modeled2 = np.zeros(6)
Wf_modeled3 = np.zeros(6)
for u in indexVals:
    Wf_modeled[u] = complexs[u]*(rho_slabs[u]/917)**(2*complexs[u])
    Wf_modeled2[u] = (10**-complexs[u])*(rho_slabs[u]/917)**(complexs[u])
    Wf_modeled3[u] = (10**a)*pow(rho_slabs_mask[u]/917, b)

plt.figure(dpi = 400)
plt.plot(simulated_density, Wf, color = "purple", label = r"$Complexity(\rho_s/\rho_{ice})^{2*Complexity}$")
plt.plot(simulated_density, Wf2, color = "orange", label = r"$(10^{-Complexity})(\rho_s/\rho_{ice})^{Complexity}$")
plt.plot(simulated_density, Wf3, color = "blue", label = r"$0.071(\rho_s/\rho_{ice})^{1.50}$")
plt.scatter(rho_slabs_mask, Wgs*0.01, color = "green", label = r"$W_gh$")
plt.scatter(rho_slabs_mask, Wf_modeled[indexVals], color = "purple", label = r"Modeled $G_{I_c}$")
plt.scatter(rho_slabs_mask, Wf_modeled2, color = "orange", label = r"Modeled $G_{I_c}$")
plt.scatter(rho_slabs_mask, Wf_modeled3, color = "blue", label = r"Modeled $G_{I_c}$")
plt.legend()
plt.xlabel(r"Density ($kg\,m^{-3}$)")
plt.ylabel(r"Critical Fracture Energy ($kPa\,m$)")
plt.title("Modeled Critical Fracture Energy vs Density")


#%% PLotting SI vs complexity 
xts = np.linspace(1, 6, 6)
plt.figure(dpi = 400)
plt.scatter(xts, SI)
plt.scatter(xts, SI2)
plt.gca().invert_xaxis()
plt.xticks(xts, storm_dates, rotation = 45, ha = "right")
plt.title("Tiltboard Stability Index Across Storms")
plt.xlabel("Storm Date")
plt.ylabel("Stability Index")

plt.figure(dpi = 400)
plt.scatter(xts, Wfs_SI, label = r"$W_f$")
plt.scatter(xts, Wgs_SI, label = r"$W_gh$")
plt.gca().invert_xaxis()
plt.xticks(xts, storm_dates, rotation = 45, ha = "right")
plt.xlabel("Storm Date")
plt.ylabel("Energy (kPa*m)")
plt.title("Energy Across Storm Dates")
plt.legend()

plt.figure(dpi = 400)
plt.scatter(xts, Wfs_SI2, label = r"$W_f$")
plt.scatter(xts, Wgs_SI2, label = r"$W_gh$")
plt.gca().invert_xaxis()
plt.xticks(xts, storm_dates, rotation = 45, ha = "right")
plt.xlabel("Storm Date")
plt.ylabel("Energy (kPa*m)")
plt.title("Energy Across Storm Dates")
plt.legend()