# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 11:57:53 2021

@author: Lauren
"""
#ESCI 895 Final Project
#Fall 2021
#Lauren Kaehler
#%% Import libraries
import numpy as np
from matplotlib import pyplot as plt
import math
from scipy.special import gamma, factorial
from scipy.signal import argrelextrema
import pandas as pd
import datetime as datetime
#%% Specify inputs
filenames = ['LampreyInstant.txt', 'SugarInstant.txt', 'LampreyAnnual.txt', 'SugarAnnual.txt']
rivers = ['Lamprey River', 'Sugar River'] 
#%% Function to handle annual data using GEV and GUMBEL
def GEVandGUMBEL(file):
    #read file
    dfannual = pd.read_csv(file, delimiter="\t", comment='#', header=1, parse_dates=['10d'])
    #rename columns
    dfannual = dfannual.rename(columns={"10d": "DATE","8s": "annual_discharge_cfs"})
    #set date as index
    dfannual = dfannual.set_index('DATE')
    #Remove unneccesary columns
    dfannual = dfannual[['annual_discharge_cfs']]
    #change data to float64 type
    #dfannual['annual_discharge_cfs']= pd.to_numeric(dfannual['annual_discharge_cfs'])
    
    #Create a dataframe with return periods
    interp = np.array([2,5,10,25,50,100,200,500,1000])
    dfinterp = pd.DataFrame(interp, columns=['Return Period (yrs)'])
    
    #Calculate exceedence probablity from return periods
    dfinterp['EP'] = 1/dfinterp['Return Period (yrs)']
    #Need a 1-EP column for Gumbel and GEV using L-moments
    dfinterp['1 - EP'] = 1 - dfinterp['EP']
    #Set index to return period
    dfinterp = dfinterp.set_index('Return Period (yrs)')

    #Create a rank column, where the largest value is ranked 1
    dfannual['rank'] = dfannual['annual_discharge_cfs'].rank(ascending= False)
    #Now sort by rank, so largest values are at the top
    dfannual = dfannual.sort_values(by = 'rank', ascending = True)
    
    #Calculate L moments
    #calculate record length
    num = len(dfannual)
    #calculate b1 for each discharge value
    dfannual['b1'] = ((num-dfannual['rank'])/(num*(num-1)))*dfannual['annual_discharge_cfs']
    #calculate b2 for each discharge value
    dfannual['b2'] = (((num-dfannual['rank'])*(num-dfannual['rank']-1))/(num*(num-1)*(num-2)))*dfannual['annual_discharge_cfs']
    #Calulate B1 and B2 by summing b1 and b2 columns
    B1 = sum(dfannual['b1'])
    B2 = sum(dfannual['b2'])
    
    #calulate l-moments
    lamda1 = np.mean(dfannual['annual_discharge_cfs'])
    lamda2 = 2*B1-lamda1
    lamda3 = 6*B2-6*B1+lamda1
    skew = lamda3/lamda2
    
    #Calulate constants for GEV
    c = 2*lamda2/(lamda3+3*lamda2)-np.log(2)/np.log(3)
    k = 7.859*c+2.9554*c**2
    alpha = (k*lamda2)/(gamma(1+k)*(1-2**(-k)))
    squiggle = lamda1 + (alpha/k)*(math.gamma(1+k)-1)
    print(c , k , alpha, squiggle)
    
    #Use equation for GEV to calculate discharge predictions for each return period
    dfinterp['GEV'] = squiggle+(alpha/k)*(1-(-np.log(dfinterp['1 - EP']))**k)
    
    #GUMBEL - calculate constants
    alphagum = lamda2/np.log(2)
    squigglegum = lamda1-0.5772*alphagum
    #Use equation for Gumbel to calculate discharge estimates for each return period.
    dfinterp['Gumbel'] = (squigglegum-alphagum*np.log(-np.log(dfinterp['1 - EP'])))
    return dfinterp
    #Print dataframe
    print(dfinterp)
    #return dfinterp
#%%
dfinterpL = GEVandGUMBEL(filenames[2])
#print(dfinterpL)
dfinterpS = GEVandGUMBEL(filenames[3])
#print(dfinterpS)
#%% POT and GPD for instaneous data
def GPD(file, threshold, rangelower, rangeupper, inc, river):
    #Load discharge file into dfpeak and format
    dfpeak = pd.read_csv(file, delimiter="\t", comment='#', header=1, parse_dates=['20d'], na_values = (9999, 999, 997,"Ice", "Eqp"))
    #Rename columns
    dfpeak = dfpeak.rename(columns={"20d": "DATE","14n": "discharge_cfs"})
    #Set date column as index
    dfpeak =dfpeak.set_index('DATE')
    #Remove not needed columns
    dfpeak = dfpeak[['discharge_cfs']]
    dfpeak['discharge_cfs'] = pd.to_numeric(dfpeak['discharge_cfs'])
    #Fill nan values through linear interpolation
    dfpeak.interpolate(method = 'linear',inplace = True)
    
    #POT method - select the max value per each storm event and remove extra values above the threshold within a single event
    #Create a new df peaks with all the peak values
    peaks = dfpeak.iloc[argrelextrema(dfpeak['discharge_cfs'].values, np.greater_equal, order = 100 )]
    #Drop all values below the threshold which are not needed for the extreme events analysis.
    peaks.drop(peaks[peaks['discharge_cfs'] < threshold].index, inplace = True)
    
    #Plot peaks selected in the POT method
    fig0, ax1 = plt.subplots()
    ax1.plot(dfpeak['discharge_cfs'])
    ax1.set_ylabel('Discharge (cfs)')
    plt.scatter(peaks.index, peaks['discharge_cfs'], color = 'r', marker = '*', label = 'Peaks')
    plt.title('Selected Peaks from Instaneous Discharge for the ' + river)
    plt.legend()
    plt.show()
    
    #Find EP for peaks df
    count = peaks['annual_discharge_cfs'].count()
    peaks = peaks.sort_values('annual_discharge_cfs', ascending = True)
    peaks['rank'] = peaks['annual_discharge_cfs'].rank(ascending= False)
    peaks['EP'] = peaks['rank']/(count+1)
    peaks['Return Period'] = 1/peaks['EP']
    
    interp = np.array([2,5,10,25,50,100,200,500,1000])
    dfinterp = pd.DataFrame(interp, columns=['Return Period (yrs)'])
    dfinterp['EP'] = 1/dfinterp['Return Period (yrs)']
    dfinterp['1 - EP'] = 1 - dfinterp['EP']
    dfinterp = dfinterp.set_index('Return Period (yrs)')
    
    #Rank and order data
    peaks = peaks.sort_values('discharge_cfs', ascending = True)
    peaks['rank'] = peaks['discharge_cfs'].rank(ascending= False)
    peaks = peaks.sort_values(by = 'rank', ascending = True)
    #Calculate L moments
    num = len(peaks)
    peaks['b1'] = ((num-peaks['rank'])/(num*(num-1)))*peaks['discharge_cfs']
    peaks['b2'] = (((num-peaks['rank'])*(num-peaks['rank']-1))/(num*(num-1)*(num-2)))*peaks['discharge_cfs']
    B1 = sum(peaks['b1'])
    B2 = sum(peaks['b2'])
    
    lamda1 = np.mean(peaks['discharge_cfs'])
    lamda2 = 2*B1-lamda1

    #GPD
    k = (lamda1 - threshold)/lamda2 - 2
    alpha = (lamda1 - threshold)/(1 + k)
    dfinterp['GPD'] = threshold +(alpha/k)*(1-(-np.log(dfinterp['1 - EP']))**k)

    #Calculating PDF/CDF
    rand_qs = np.arange(rangelower,rangeupper, inc).tolist()
    dfq = pd.DataFrame(rand_qs, columns = ['Discharge (cfs)'])
    dfq['PDF'] = (1/alpha)*(1-k*((dfq['Discharge (cfs)'] - threshold)/alpha))**((1/k)-1)
    dfq['CDF'] = 1 - (1- k*((dfq['Discharge (cfs)'] - threshold)/alpha))**(1/k)
    dfq['EP'] = 1 - dfq['CDF']
    dfq['Return Period (yrs)'] = 1/dfq['EP']
    
    #Plot CDF and PDF
    fig1, ax1 = plt.subplots()
    ax1.plot(dfq['Discharge (cfs)'], dfq['PDF'], label = 'PDF')
    #Create a secondary y axis to plot the CDF curve
    ax2 = ax1.twinx()
    ax2.plot( dfq['Discharge (cfs)'], dfq['CDF'], color = 'orange' , label = 'CDF')
    plt.title('PDF and CDF curves created using the GPD POT Method for the ' + river)
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    lines = lines_1 + lines_2
    #labels = labels1 + labels2
    labels = [l.get_label() for l in lines]
    ax1.set_xlabel('Discharge_cfs')
    ax1.set_ylabel('PDF(x)')
    ax2.set_ylabel('CDF(x)')
    ax1.legend(lines, labels, loc = 7)
    plt.show()
    return dfinterp, dfq, fig0, fig1, peaks
#%%Run GEV function for Lamprey and Sugar
GPDoutputL = GPD(filenames[0], 1000, 1500, 15000, 500, rivers[0])
print(GPDoutputL[1])
GPDoutputS = GPD(filenames[1], 1000, 1500, 10000, 250, rivers[1])
print(GPDoutputS[1])
#%%Create 2 dataframes, one for lamprey and other for sugar but combine dfinterps
dfL = dfinterpL.merge(GPDoutputL[0], how = 'outer', on = ['Return Period (yrs)', 'EP', '1 - EP'])
dfL = dfL.drop(columns = ['1 - EP'])
print(dfL)
#Sugar
dfS = dfinterpS.merge(GPDoutputS[0], how = 'outer', on = ['Return Period (yrs)', 'EP', '1 - EP'])
dfS = dfS.drop(columns = ['1 - EP'])
print(dfS)
#%%Create a figure like the one in lab 5 (one for Lamprey and one for Sugar)
fig1, (ax1, ax2) = plt.subplots(1,2)
fig1.suptitle('Distributions for ' + rivers)
ax1.scatter(GPDoutputL[], dfpeak['annual_discharge_cfs'],color = 'c', label = 'data')
ax2.scatter(dfpeak['Return Period'], dfpeak['annual_discharge_cfs'], color = 'c')
ax1.set_xlabel('Exceedence Probablity')
ax1.set_yscale('log')
ax1.set_ylabel('Annual Peak Discharge (cfs)')
ax1.set_xscale('log')
ax2.set_xlabel('Return Period')
ax2.set_yscale('log')
ax2.set_xscale('log')

#linear
m1, b1 = np.polyfit(dfpeak['EP'], dfpeak['annual_discharge_cfs'], 1)
dfinterp['linear'] = dfinterp['EP']*m1 +b1 #cfs
ax2.plot(dfinterp['Return Period (yrs)'],dfinterp['linear'], color = 'r', label = 'linear')
ax1.plot(dfinterp['EP'], dfinterp['linear'], color = 'r')

#log-log
m2, b2 = np.polyfit(np.log10(dfpeak['Return Period']), np.log10(dfpeak['annual_discharge_cfs']), 1)
dfinterp['power-law return period'] = 10**(np.log10(dfinterp['Return Period (yrs)'])*m2 +b2) #cfs
ax2.plot(dfinterp['Return Period (yrs)'],dfinterp['power-law return period'], color = 'g', label = 'log-log')
ax1.plot(dfinterp['EP'], dfinterp['power-law return period'], color = 'g')

#log normal
ax2.plot(dfinterp['Return Period (yrs)'],10**dfinterp['log Qp'], color = 'y', label = 'log normal')
ax1.plot(dfinterp['EP'], 10**dfinterp['log Qp'], color = 'y')

#log pearson 3
ax2.plot(dfinterp['Return Period (yrs)'],10**dfinterp['LP3 Q'], color = 'b', label = 'LP3')
ax1.plot(dfinterp['EP'], 10**dfinterp['LP3 Q'], color = 'b')

plt.legend()
plt.show()



