###############################################################################

import os
import numpy as np
import matplotlib.pyplot as plt
import requests
import pygrib
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cartopy
import cartopy.mpl.geoaxes
import sys

###############################################################################

"""
Input variables:
    
    GEFS forecast info:
        Year (int): Year of GEFS run
        Month (int): Month of GEFS run
        Day (int): Day of GEFS run
        
    Forecast time range info:
        StormBegin (int): GEFS forecast hour corresponding to beginning of
                          snowfall period
        StormEnd (int): GEFS forecast hour corresponding to end of snowfall
                        period
                        
    Forecast location info:
        lat (float): Latitude of forecast location (in degrees N)
        lon (float): Longitude of forecast location (in degrees E or W)
"""

Year=2025
Month=1
Day=17 
InitHour=18 

StormBegin=48
StormEnd=66

lat=42.3
lon=-71 

###############################################################################

def download_gefs_member(member,fhr,date,run_time,lat,lon,local_dir="data"):
    """
    Download GEFS ensemble member data for a specific forecast hour

    Parameters:
        member (int): Ensemble member number (1-30 for GEFS)
        fhr (int): Forecast hour (e.g., 0, 6, 12, 18, etc.)
        date (str): Date in YYYYMMDD format
        run_time (int): Model run time (0, 6, 12, 18)
        lat (float): Latitude of forecast location (in degrees N)
        lon (float): Longitude of forecast location (in degrees E or W)
        local_dir (str): Temporary directory to save downloaded data

    Returns:
        Path to the downloaded file (str)
    """
    
    nomads_url="https://nomads.ncep.noaa.gov/pub/data/nccf/com/gens/prod/"
    gefs_url=f"gefs.{date}/{run_time:02d}/atmos/pgrb2ap5/"
    base_url=nomads_url+gefs_url
    filename=f"gep{member:02d}.t{run_time:02d}z.pgrb2a.0p50.f{fhr:03d}"

    lon_min=lon
    lon_max=lon
    lat_min=lat
    lat_max=lat

    params = {
        "file": filename,
        "var_SNOD": "on",
        "var_WEASD": "on",
        "subregion": "",
        "leftlon": lon_min,
        "rightlon": lon_max,
        "toplat": lat_max,
        "bottomlat": lat_min,
    }

    # Filter out None values
    params = {k: v for k, v in params.items() if v is not None}

    file_url = os.path.join(base_url, filename)
    local_path = os.path.join(local_dir, filename+'.grb2')

    # Ensure local directory exists
    os.makedirs(local_dir, exist_ok=True)

    # Download the file
    if not os.path.exists(local_path):
        print(f"Downloading {filename}...")
        response = requests.get(file_url,params=params)
        if response.status_code == 200:
            with open(local_path, "wb") as f:
                f.write(response.content)
        else:
            exception=f"Failed to download {file_url}. HTTP status code: {response.status_code}"
            raise Exception(exception)

    return local_path

def extract_variable(file_path, variable, latitude, longitude):
    """
    Extract a variable from a GRIB file at a specific location.

    Parameters:
        file_path (str): Path to the data file.
        variable (str): Variable to extract (e.g., "TMP").
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.

    Returns:
        float: Variable value at the specified location.
    """
    # Open the GRIB file using pygrib
    grbs = pygrib.open(file_path)
    grb = grbs.select(shortName=variable)[0]

    # Get the data and find the nearest grid point
    data, lats, lons = grb.data()
    lat_idx = np.abs(lats[:,0] - latitude).argmin()
    lon_idx = np.abs(lons[0,:] - longitude).argmin()

    # Extract the value and close the file
    value = data[lat_idx, lon_idx]
    grbs.close()

    return value

def plume_variables(lat,lon):
    snow_depth=np.zeros((30,2))
    snow_water=np.zeros((30,2))
    for mem in range(1,31):
        snow_depth[mem-1,0]=extract_variable(download_gefs_member(mem,StormBegin,date,InitHour,lat,lon),'sde',lat,lon)
        snow_water[mem-1,0]=extract_variable(download_gefs_member(mem,StormBegin,date,InitHour,lat,lon),'sdwe',lat,lon)
        snow_depth[mem-1,1]=extract_variable(download_gefs_member(mem,StormEnd,date,InitHour,lat,lon),'sde',lat,lon)
        snow_water[mem-1,1]=extract_variable(download_gefs_member(mem,StormEnd,date,InitHour,lat,lon),'sdwe',lat,lon)
    return snow_depth,snow_water

###############################################################################

# Format date for grib file retrieval
date=str(Year)+str(Month).zfill(2)+str(Day).zfill(2)

# Convert longitude to degrees E if necessary to match file convention
if lon<0:
    lon=360+lon

# Define conversion factors between inches and meters/cm
cf=39.3701
inc=cf/100

# Define month names for figure title
months=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov',
        'Dec']

###############################################################################

snow_depth_data,snow_water_data=plume_variables(lat,lon)

###############################################################################

snow_depth=np.array(snow_depth_data)
snow_water=np.array(snow_water_data)/1000

snow_depth_accum=snow_depth[:,1]-snow_depth[:,0]
snow_water_accum=snow_water[:,1]-snow_water[:,0]

snow_water_accum[snow_water_accum<=0]=-1e-5

if np.max(snow_water_accum)==-1e-5:
    print('ERROR: No GEFS members found with accumulating snow.')
    sys.exit()

SLR=snow_depth_accum/snow_water_accum

SLR[SLR<=0]=np.mean(SLR[snow_water_accum>0])

snow_water_accum[snow_water_accum<0]=0

qsd=[]
for r in SLR:
    for p in snow_water_accum:
        qsd.append(r*p)
qsd=np.array(qsd)

###############################################################################

counts,bin_edges=np.histogram(qsd*cf,bins=np.arange(-inc,250,inc),density=True)
cumulative_counts=np.cumsum(counts)*(bin_edges[1]-bin_edges[0])*100

left=np.where((100-cumulative_counts)==100)[0]
right=np.where((100-cumulative_counts)<=1)[0]

fig,ax=plt.subplots(dpi=500)
fig.patch.set_facecolor('steelblue')
ax.set_facecolor('steelblue')
ax1=ax.twiny()
ax.set_xlim(left[-1]*inc,right[0]*inc)
ax1.set_xlim(left[-1],right[0])
ax.set_ylim(0,100)

ax.scatter(bin_edges[1::],(100-cumulative_counts),marker="$*$",c='white',s=50)
ax.set_xlabel('Snow Accumulation (in)')
ax1.set_xlabel('Snow Accumulation (cm)')
ax.set_ylabel('Probability of Exceedance (%)')
ax.set_yticks(np.arange(0,110,10))
ax.set_title('Snowfall Exceedance Function\n'+str(InitHour).zfill(2)+
             'z GEFS init '+months[Month-1]+' '+str(Day)+', '+str(Year)+
             ', fcst hrs '+str(StormBegin)+'-'+str(StormEnd)+'\n')

axin=inset_axes(ax, width="40%", height="40%", loc="upper right",
                   axes_class=cartopy.mpl.geoaxes.GeoAxes, 
                   axes_kwargs=dict(projection=cartopy.crs.PlateCarree()))

axin.add_feature(cartopy.feature.COASTLINE,edgecolor='black',lw=0.25,zorder=1)
axin.add_feature(cartopy.feature.STATES,edgecolor='black',lw=0.25,zorder=1)
axin.add_feature(cartopy.feature.BORDERS,edgecolor='black',lw=0.25,zorder=1)

axin.set_facecolor('steelblue')
axin.patch.set_facecolor('steelblue')

axin.set_ylim(lat-15,lat+15)
if lon>180:
    lon=lon-360
axin.set_xlim(lon-15,lon+15)

axin.scatter(lon,lat,c='white',s=20,zorder=2)

plt.show()

###############################################################################
