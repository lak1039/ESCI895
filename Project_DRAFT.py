# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 20:27:45 2021

@author: Lauren
"""
## Hi! 

#This is a bit of a mess right now, but I am trying to find the 2, 10, 25, 50, 100, 500, 1000 year floods for the lamprey and sugar rivers.
#This is similar to lab 5 except I am going to do different distributions (gumbel and GEV)
#Another part of this will be using a peaks over threshold technique so I can use instaneuos discharge rather than
#annual max. But I am really stuggling with that so I did not include that in this code... I am still trying to come up with what figures
#and charts to show to include, so adivce on that would be lovely. But I figure I will end up displaying a table like dfinterp
#I need to add in more detail to my code to explain what I am doing more aswell (I will be busy this weekend :)

#Sorry there is not more to critque!

#Best,
#Lauren

#I need to add more details on what I am doing and break up the code more...

#%% Import libraries
import numpy as np
from matplotlib import pyplot as plt
from scipy.special import gamma, factorial
import pandas as pd
import datetime
#%% Specify inputs
filenames = ['hourlyqlamprey.txt', 'annualn.txt'] 
location = "Newmarket, NH" 
#%% Part 1
#Load discharge file into dfdaily and format
dfdaily = pd.read_csv(filenames[0], delimiter="\t", comment='#', header=1, parse_dates=['20d'], na_values = (9999, 999, 997, "Eqp"))
dfdaily = dfdaily.rename(columns={"20d": "DATE"})
dfdaily =dfdaily.set_index('DATE')
dfdaily = dfdaily.rename(columns={"14n": "discharge_cfs"})
dfdaily = dfdaily[['discharge_cfs']]
dfdaily.interpolate(method = 'linear',inplace = True)
#%%
dfsorted = dfdaily.sort_values('discharge_cfs', ascending = False)
print("largest daily discharge", dfsorted.index[0], dfsorted.iat[0,0])
print("second largest daily discharge:", dfsorted.index[1], dfsorted.iat[1,0])
print("third largest daily discharge:", dfsorted.index[2], dfsorted.iat[2,0])
#%%Streamflow data
dfpeak = pd.read_csv(filenames[1], delimiter="\t", comment='#', header=1, parse_dates=['4s'])
dfpeak = dfpeak.rename(columns={"4s": "DATE"})
dfpeak =dfpeak.set_index('DATE')
dfpeak = dfpeak.rename(columns={"12n": "annual_discharge_cfs"})
dfpeak = dfpeak[['annual_discharge_cfs']]
dfpeak['annual_discharge_cfs']=pd.to_numeric(dfpeak['annual_discharge_cfs'])
#%%
#make a dfinterp
interp = np.array([2,5,10,25,50,100,200,500,1000])
dfinterp = pd.DataFrame(interp, columns=['Return Period (yrs)'])
dfinterp['EP'] = 1/dfinterp['Return Period (yrs)']
#Take mean and std of peak annual discharge data
peak_mean = dfpeak['annual_discharge_cfs'].mean()
peak_std = dfpeak['annual_discharge_cfs'].std()
print("peak mean: ", peak_mean)
print("peak std: ", peak_std)
#%% Print three highest peaks for yearly data
dfpeaksorted = dfpeak.sort_values('annual_discharge_cfs', ascending = False)
print("largest yearly discharge", dfpeaksorted.index[0], dfpeaksorted.iat[0,0])
print("second largest yearly discharge:", dfpeaksorted.index[1], dfpeaksorted.iat[1,0])
print("third largest yearly discharge:", dfpeaksorted.index[2], dfpeaksorted.iat[2,0])
#%% Create a time series plot
fig, ax0 = plt.subplots()
ax0.plot(dfdaily['discharge_cfs'], color = 'b' , linestyle = '-', label = 'daily discharge')
ax0.plot(dfpeak['annual_discharge_cfs'], color = 'r' , linestyle = '-', label = 'yearly discharge')     
# Create y-axis label 
ax0.set_ylabel('Discharge (cfs)')  
# Set the y-axis to start at 0 (default is to offset 0 from the axis)
ax0.set_ylim(bottom = 0)
# Make tick labels diagonal to avoid overlap
fig.autofmt_xdate()                
# Title the plot
ax0.set_title(location)
#Annotate with arrows
# ax0.annotate(s = dfsorted.index[0].strftime("%m-%d-%Y") + str(dfsorted.iat[0,0]), xy=(dfsorted.index[0], dfsorted.iat[0,0]), xytext=(dfsorted.index[0], dfsorted.iat[0,0]+100), horizontalalignment='right', verticalalignment='top',
#              arrowprops={'arrowstyle': '->'}, va='center')
# ax0.annotate(s = dfsorted.index[1].strftime("%m-%d-%Y") + str(dfsorted.iat[1,0]), xy=(dfsorted.index[1], dfsorted.iat[1,0]), xytext=(dfsorted.index[1], dfsorted.iat[1,0]+800),
#              arrowprops={'arrowstyle': '->'}, va='center')
# ax0.annotate(s = dfsorted.index[2].strftime("%m-%d-%Y") + str(dfsorted.iat[2,0]), xy=(dfsorted.index[2], dfsorted.iat[2,0]), xytext=(dfsorted.index[2], dfsorted.iat[2,0]+200),
#              arrowprops={'arrowstyle': '->'}, va='center')
# ax0.annotate(s = dfpeaksorted.index[0].strftime("%m-%d-%Y") + str(dfpeaksorted.iat[0,0]), xy=(dfpeaksorted.index[0], dfpeaksorted.iat[0,0]), xytext=(dfpeaksorted.index[0], dfpeaksorted.iat[0,0]-500), horizontalalignment='left',
#              arrowprops={'arrowstyle': '->'}, va='center')
# ax0.annotate(s = dfpeaksorted.index[1].strftime("%m-%d-%Y") + str(dfpeaksorted.iat[1,0]), xy=(dfpeaksorted.index[1], dfpeaksorted.iat[1,0]), xytext=(dfpeaksorted.index[1], dfpeaksorted.iat[1,0]+800),
#              arrowprops={'arrowstyle': '->'}, va='center')
# ax0.annotate(s = dfpeaksorted.index[2].strftime("%m-%d-%Y") + str(dfpeaksorted.iat[2,0]), xy=(dfpeaksorted.index[2], dfpeaksorted.iat[2,0]), xytext=(dfpeaksorted.index[2], dfpeaksorted.iat[2,0]+700),
#              arrowprops={'arrowstyle': '->'}, va='center')
ax0.legend()
plt.show()
#%% Use annual peak data to calculate flow exceednec probablity using Weibull
#Add columns to dfpeak for exceedence prob and return period
count = dfpeak['annual_discharge_cfs'].count()
dfpeak = dfpeak.sort_values('annual_discharge_cfs', ascending = True)
dfpeak['rank'] = dfpeak['annual_discharge_cfs'].rank(ascending= False)
dfpeak['EP'] = dfpeak['rank']/(count+1)
dfpeak['Return Period'] = 1/dfpeak['EP']
#dfpeak = dfpeak[['annual_discharge_cfs','EP','Return Period']]
#%%Interpolate to find discharge values for dfinterp
dfpeak = dfpeak.sort_values(by = 'rank', ascending = False)
dfinterp['interpolated discharge (cfs)']= np.interp(dfinterp['Return Period (yrs)'],dfpeak['Return Period'],dfpeak['annual_discharge_cfs'])
dfinterp['interpolated discharge (cfs)']= dfinterp['interpolated discharge (cfs)'].where(dfinterp['interpolated discharge (cfs)'] < dfpeak['annual_discharge_cfs'].max(), np.nan)

#%% 2.10
#%% Part 3
#Add a new column for log(Q) in dfpeak
dfpeak['log Q']= np.log10(dfpeak['annual_discharge_cfs']) #cfs
#take mean and std of log Q column
peaklog_mean = dfpeak['log Q'].mean()
peaklog_std = dfpeak['log Q'].std()
print("peak log mean: ", peaklog_mean)
print("peak log std: ", peaklog_std)
#%%
fig2, (ax3, ax4) = plt.subplots(1,2)
fig2.suptitle(location)
ax3.hist(dfpeak['annual_discharge_cfs'])
ax3.axvline(x= peak_mean, color = 'k')
#ax3.annotate(s = 'Average: ' + str(peak_mean)+' cfs', xy = (6000,11))
ax3.set_xlabel('Annual Peak Discharge (cfs)')
ax3.set_ylabel('Number of Occurences')
ax4.hist(dfpeak['log Q'])
ax4.axvline(x= peaklog_mean, color = 'k', label = 'Average: ' + str(peak_mean))
#ax4.annotate(s = 'Average: ' + str(round(peak_mean,3)) + ' cfs', xy = (3,10))
ax4.set_xlabel('Log10 Annual Peak Discharge (cfs)')
plt.legend()
plt.show()
#%% Part 3.5 Log normal
#Frequency factor
dfinterp['k'] = ((1-dfinterp['EP'])**0.135-dfinterp['EP']**0.135)/0.1975
dfinterp['log Qp'] = peaklog_mean + dfinterp['k']*peaklog_std 
dfinterp['10**log Qp'] = 10**dfinterp['log Qp'] #cfs
#%% Part 4 LP3
n = len(dfpeak['log Q'])
gsx =(n*sum((dfpeak['log Q']-peaklog_mean)**3))/((n-1)*(n-2)*peaklog_std**3)
mse = (10**((-0.33+0.08*gsx)+0.94-0.26*gsx))/(n**(0.94-0.26*gsx))
msegrx = 0.302
grx = 0.05
gx = ((gsx/mse)+(grx/msegrx))/((1/mse)+(1/msegrx))
dfinterp['kgep'] = (2/gx)*((1+gx*(((1-dfinterp['EP'])**0.135-dfinterp['EP']**0.135)/1.185))-(gx**2/36))**3-(2/gx)
dfinterp['LP3 Q'] = peaklog_mean + dfinterp['kgep']*peaklog_std 
dfinterp['10**LP3 Q'] = 10**dfinterp['LP3 Q'] #cfs
#%% NEW ADDING IN GUMBEL AND GEV
dfpeak = dfpeak.sort_values(by = 'rank', ascending = True)
#Calculate L moments
num = len(dfpeak)
dfpeak['b1'] = ((num-dfpeak['rank'])/(num*(num-1)))*dfpeak['annual_discharge_cfs']
dfpeak['b2'] = (((num-dfpeak['rank'])*(num-dfpeak['rank']-1))/(num*(num-1)*(num-2)))*dfpeak['annual_discharge_cfs']
B1 = sum(dfpeak['b1'])
B2 = sum(dfpeak['b2'])

lamda1 = np.mean(dfpeak['annual_discharge_cfs'])
lamda2 = 2*B1-lamda1
lamda3 = 6*B2-6*B1+lamda1
skew = lamda3/lamda2

#%% GEV
c = 2*lamda2/(lamda3+3*lamda2)-np.log(2)/np.log(3)
k = 7.859*c+2.9554*c**2
alpha = (k*lamda2)/(gamma(1+k)*(1-2**(-k)))
squiggle = lamda1 + (alpha/k)*(gamma(1+k)-1)
dfinterp['1 - EP'] = 1 - dfinterp['EP']
dfinterp['GEV'] = (squiggle+(alpha/k)*(1-(-np.log(dfinterp['1 - EP']))**k)).astype(int)

#%%GUMBEL
alphagum = lamda2/np.log(2)
squigglegum = lamda1-0.5772*alphagum
dfinterp['Gumbel'] = squigglegum-alphagum*np.log(-np.log(dfinterp['1 - EP']))

#%% Plot 2a
fig1, (ax1, ax2) = plt.subplots(1,2)
fig1.suptitle(location)
ax1.scatter(dfpeak['EP'], dfpeak['annual_discharge_cfs'],color = 'c', label = 'data')
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

#GEV
ax2.plot(dfinterp['Return Period (yrs)'],10**dfinterp['GEV'], color = 'k', label = 'GEV')
ax1.plot(dfinterp['EP'], 10**dfinterp['GEV'], color = 'k')

plt.legend()
plt.show()