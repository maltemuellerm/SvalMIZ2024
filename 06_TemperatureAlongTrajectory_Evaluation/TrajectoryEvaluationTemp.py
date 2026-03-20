#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from scipy.interpolate import griddata
import pyproj
from pyproj import Transformer
from datetime import datetime, timedelta
import os, fnmatch, glob
import cmocean 
import rasterio
from rasterio.plot import show
import sys
import geopy.distance
import pandas as pd


# In[2]:


class StatisticalEvaluation:
    
    def __init__(self,buoys,model,starttime,endtime,flag,filedrift,filetemp):
        self.buoys     = buoys
        self.model     = model
        self.starttime = starttime
        self.endtime   = endtime
        self.flag      = flag      
        self.OMBdrift  = xr.open_mfdataset(filedrift)
        self.OMBtemp   = xr.open_mfdataset(filetemp)
               
        
        
    def info(self):
        print ('Statistical evaluation of '+str(len(self.buoys))+' buoys')
        print ('Use of model: ',self.model)
        
        
    def flag_values(self,rad_threshold):
        
        #OMBtemp.temp_flag=np.zeros(OMBtemp.tr_nr,OMBtemp.time_ds)
        self.OMBtemp['temp_flag'] = (['tr_nr', 'time_ds'],self.OMBtemp['temp_flag_1m'].values)
        for buoyno in self.buoys:
            flag1 = self.OMBtemp.temp_flag_1m[buoyno,:].values
            flag2 = self.OMBtemp.temp_flag_cons[buoyno,:].values
            flag3 = self.OMBtemp.ssdr[0,0,buoyno,:].values
            
            # Radiation needs to be filled between the hours:
            arr_series = pd.Series(flag3)
            # Interpolate missing values
            flag3 = arr_series.interpolate(method='linear').to_numpy()
            
            condition = (flag1 == 1.) | (flag1 == 1) |(flag3 > rad_threshold)
            
            self.OMBtemp.temp_flag[buoyno,:]=condition
            
        
    def teval_cond_err(self,smin,smax,sinc,leadno,cond,doplot):
           
        model=np.where(self.OMBtemp.model == self.model)[0][0]
        
        print (' -- choose model number ',model)
        # Define bias according to leadtime intervals       
        bias = np.zeros(leadno); rmse = np.zeros(leadno); stde = np.zeros(leadno); 
        # Bins For Conditional analysis
        bins = np.arange(smin+sinc, smax, sinc)
        print(bins)
        cond_bias = np.zeros([len(bins)-1,leadno])
        cond_mae  = np.zeros([len(bins)-1,leadno])
        cond_rmse = np.zeros([len(bins)-1,leadno])
        cond_nobs = np.zeros([len(bins)-1,leadno])
           
        for lt in range(leadno):     
            temp_obs = [];temp_mod=[]
            sic_obs = []; sic_mod = []
        
            # Read all observations from specified buoys
            for by in range(len(self.buoys)):
                buoyno=self.buoys[by]
                
                # Find the appropriate times and their indexes within the specified range and the flagging
                time_values = self.OMBtemp.time_ds.values
                flag_values = self.OMBtemp.temp_flag[buoyno,:].values
                temp_1m_flagged = self.OMBtemp.temp_1m_calibrated[buoyno,:].values
                #temp_1m_flagged[self.OMBtemp.temp_flag[buoyno,:].values==1]=np.nan
                                
                # Select specific time range:       
                boolean_mask = (time_values >= np.datetime64(self.starttime)) & (time_values <= np.datetime64(self.endtime)) & (flag_values == 0.)
                indexes = np.where(boolean_mask)[0]
                
                temp_obs.extend(temp_1m_flagged[indexes])
                if model==0:
                    temp_mod.extend(self.OMBtemp.T2M[model,lt,buoyno,indexes].values)
                else:
                    temp_mod.extend(self.OMBtemp.T2M[model,lt,buoyno,indexes].values)
                
                sic_obs.extend(self.OMBdrift.AMSR2_SIC[buoyno,indexes].values)
                sic_mod.extend(self.OMBtemp.SIC[model,lt,buoyno,indexes].values)
                
            temp_obs = np.array(temp_obs)-273.15; temp_mod = np.array(temp_mod)-273.15
            sic_obs = np.array(sic_obs);   sic_mod = np.array(sic_mod)

            # Non-conditional bias and RMSE
            
            bias[lt] = np.nanmean(temp_mod - temp_obs)
            rmse[lt] = np.sqrt(np.nanmean((temp_obs - temp_mod) ** 2))
            stde[lt] = np.nanstd(temp_mod - temp_obs)
            #print('Bias',rmse[lt],' Leadtime',lt,' Array size',np.shape(temp_obs))

            # -----------------------------------
            # Calculate conditional bias and RMSE
            if (cond=='sicerr'):
                sic_diff = sic_mod - sic_obs
                digitized = np.digitize(sic_diff, bins)
                xlabel = "Sea Ice Concentration Error"
            elif (cond=='sic'):
                digitized = np.digitize(sic_obs, bins)
                xlabel = "Sea Ice Concentration"
            elif (cond=='temp'):
                digitized = np.digitize(temp_obs, bins)
                xlabel = "Observed temperature"
                
            
            conditional_bias = []; conditional_rmse = []; conditional_mae = [];

            for i in range(1, len(bins)):
                
                bin_mask = digitized == i
                if np.any(bin_mask):
                    bin_temp_obs = temp_obs[bin_mask]
                    bin_temp_mod = temp_mod[bin_mask]
                    cond_nobs[i-1,lt]=len(bin_temp_obs)
                    if len(bin_temp_obs) > 0:
                        bin_bias = np.nanmean(bin_temp_mod - bin_temp_obs)
                        bin_mae  = np.nanmean(np.abs(bin_temp_mod - bin_temp_obs))
                        bin_rmse = np.sqrt(np.nanmean((bin_temp_obs - bin_temp_mod) ** 2))
                        conditional_bias.append(bin_bias)
                        conditional_mae.append(bin_mae)
                        conditional_rmse.append(bin_rmse)
                    else:
                        conditional_bias.append(np.nan)
                        conditional_mae.append(np.nan)
                        conditional_rmse.append(np.nan)
                else:
                    conditional_bias.append(np.nan)
                    conditional_mae.append(np.nan)
                    conditional_rmse.append(np.nan)

            cond_bias[:,lt] =  conditional_bias
            cond_mae[:,lt]  =  conditional_mae
            cond_rmse[:,lt] =  conditional_rmse
             
        self.bias=bias
        self.rmse=rmse
        self.stde=stde
        self.cond_bias=cond_bias
        self.cond_mae =cond_mae
        self.cond_rmse=cond_rmse
        self.bins     =bins[1:len(bins)]
        bins     =bins[1:len(bins)]-sinc/2.
        
        if (doplot):
            
            fig, ax = plt.subplots(figsize=(8, 12),nrows=3,ncols=1)
            cmap = plt.get_cmap("Blues")
            norm = plt.Normalize(vmin=0, vmax=leadno)  

            for lt in range(leadno):
                color=cmap(norm(lt+1))
                ax[0].plot(bins[:],cond_bias[:,lt],marker='.',linewidth=1.5,color=color,label=self.OMBtemp.lt_int[lt].values)
                ax[0].plot(bins[:],cond_mae[:,lt],marker='*',linestyle='--', linewidth=1.5,color=color)

           
            ax[0].set_title("Bias (solid) and MAE (dashed)",fontsize=16)
            ax[0].set_xlabel(xlabel,fontsize=14)
            ax[0].set_ylabel("Temperature Bias and MAE",fontsize=14)
            #ax[0].legend()

            for lt in range(leadno):
                color=cmap(norm(lt+1))
                ax[1].plot(bins[:],cond_rmse[:,lt],marker='.',linewidth=1.5,color=color,label=self.OMBtemp.lt_int[lt].values)

           
            ax[1].set_title("RMSE",fontsize=16)
            ax[1].set_xlabel(xlabel,fontsize=14)
            ax[1].set_ylabel("Temperature RMSE",fontsize=14)
            ax[1].legend()
            plt.tight_layout()
        
            for lt in range(leadno):
                color=cmap(norm(lt+1))
                ax[2].plot(bins[:],cond_nobs[:,lt],marker='.',linewidth=1.5,color=color,label=self.OMBtemp.lt_int[lt].values)
            
            
            
            ax[2].legend(title='Leadtimes',fontsize=12)
            ax[2].set_title("Number of observations",fontsize=16)
            ax[2].set_xlabel(xlabel,fontsize=14)
            ax[2].set_ylabel("Number of Observations",fontsize=14)
            
            
            # Add vertical lines for bins
            for bin_position in bins[:]:
                ax[0].axvline(bin_position-sinc/2, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
                ax[1].axvline(bin_position-sinc/2, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
                ax[2].axvline(bin_position-sinc/2, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
            ax[0].axvline(bins[-1]-sinc/2, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
            ax[1].axvline(bins[-1]-sinc/2, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
            ax[2].axvline(bins[-1]-sinc/2, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
            # Add the text box to the overall figure
            fig.text(
                    0.2, 0.95, self.model,  # Position (x, y)
                    fontsize=12, ha='center', va='top',
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='black')
                    )
            plt.tight_layout()
            


# In[ ]:




