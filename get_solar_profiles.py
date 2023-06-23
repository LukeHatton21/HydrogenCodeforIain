import xarray as xr
import pandas as pd
import numpy as np
import pvlib
import time
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


class Get_Solar:
    def __init__(self, t2m, ssrd, bathymetry):
        self.t2m_ds = xr.open_dataset(t2m)
        self.ssrd_ds = xr.open_dataset(ssrd)
        self.bathymetry_data = xr.open_dataset(bathymetry)
        self.altitudes = self.get_altitudes()
        self.t2m_ds = self.update_time(self.t2m_ds)
        self.ssrd_ds = self.update_time(self.ssrd_ds)
        self.hourly_data = pd.to_datetime(self.t2m_ds.time.values)
        
            
    def get_altitudes(self):

        # Read Bathymetry file
        bathymetry = self.bathymetry_data

        # Read renewables file and identify resolution
        target_data = self.t2m_ds

        
        # Reindex using the xarray reindex function
        new_coords = {'lat': target_data.lat, 'lon': target_data.lon}
        bathymetry_resampled = bathymetry.reindex(new_coords, method='nearest')


        # Create new dataset with altitude values
        data_vars = {'z': bathymetry_resampled['z']}
        coords = {'lat': bathymetry_resampled.lat,
                  'lon': bathymetry_resampled.lon}
        altitudes = xr.Dataset(data_vars=data_vars, coords=coords)
        altitudes = xr.where(altitudes > 0, altitudes, 0)

        return altitudes
    
    def update_time(self, existing_dataset):
        
        # Define the start time and number of data points for the new datetime index
        start_time = self.t2m_ds.time.values[0]
        num_data_points = len(self.t2m_ds.time.values)

        # Create the new datetime index using pd.date_range
        new_time = pd.date_range(start=start_time, periods=num_data_points, freq="1H")

        # Update the time coordinate of the existing dataset with the new datetime index
        existing_dataset["time"] = new_time
                               
        return existing_dataset

    def get_data(self, latitude, longitude):
        """Imports the data from the nc files and interprets them into wind and solar profiles"""
        ssrd_ds = self.ssrd_ds
        t2m_ds = self.t2m_ds
        altitude = self.altitudes.z.loc[latitude, longitude].values
        t2m = t2m_ds.T2M.loc[:, latitude, longitude].values
        ssrd = ssrd_ds.SWGDN.loc[:, latitude, longitude].values

        return self.get_solar_power(ssrd, t2m, altitude, latitude, longitude)
    
    
    
    def get_coords_subset(self, values, start, end):

        # Calculate the absolute differences between each element and the desired values
        start_diff = np.abs(values - start)
        end_diff = np.abs(values - end)

        # Find the index of the minimum absolute difference for start and end values
        start_index = np.argmin(start_diff)
        end_index = np.argmin(end_diff)

        # Retrieve the subset of values based on the nearest neighbors
        selected_values = values[start_index:end_index+1]
    
        return selected_values
    
    

    def get_solar_power(self, ssrd, t2m, altitude, latitude, longitude):
        """Uses PV_Lib to estimate solar power based on provided weather data"""
        """Note t2m to the function in Kelvin - function converts to degrees C!"""
        # Manipulate input data
        times = self.hourly_data.tz_localize('ETC/GMT')
        ssrd = pd.DataFrame(ssrd, index=times, columns=['ghi']) 
        t2m = pd.DataFrame(t2m - 273.15, index=times, columns=['temp_air'])

        
        # Set up solar pv system characteristics
        if latitude > 0:
            azimuth_angle = 180
        else:
            azimuth_angle = 0
        tilt_angle = latitude
        temperature_model_parameters = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
        pvwatts_system = pvlib.pvsystem.PVSystem(surface_tilt = tilt_angle, surface_azimuth = azimuth_angle, module_parameters={'pdc0': 240, 'gamma_pdc': -0.004},inverter_parameters={'pdc0': 240},temperature_model_parameters=temperature_model_parameters)
        
        # Set up solar farm design
        mc_location = pvlib.location.Location(latitude=latitude, longitude=longitude, altitude=altitude,
                                              name='NA')
        solpos = pvlib.solarposition.pyephem(times, latitude=latitude, longitude=longitude, altitude=altitude,
                                             pressure=101325, temperature=t2m.mean(), horizon='+0:00')
        mc = pvlib.modelchain.ModelChain(pvwatts_system, mc_location, aoi_model='physical',
                                         spectral_model='no_loss')

        # Get the diffuse normal irradiance (dni) and diffuse horizontal irradiance (dhi) from the data; hence create a weather dataframe
        df_res = pd.concat([ssrd, t2m, solpos['zenith']], axis=1)
        #df_res['dni'] = pd.Series([pvlib.irradiance.disc(ghi, zen, i)['dni'] for ghi, zen, i in
                                   #zip(df_res['ghi'], df_res['zenith'], df_res.index)], index=times).astype(float)
        #df_res['dhi'] = df_res['ghi'] - df_res['dni'] * np.cos(np.radians(df_res['zenith']))
        
        df_res['dhi'] = pd.Series([pvlib.irradiance.boland(ghi, zen, i)['dhi'] for ghi, zen, i in zip(df_res['ghi'], df_res['zenith'], df_res.index)], index=times).astype(float)
        df_res['dni'] = pd.Series([pvlib.irradiance.boland(ghi, zen, i)['dni'] for ghi, zen, i in zip(df_res['ghi'], df_res['zenith'], df_res.index)], index=times).astype(float)
        weather = df_res.drop('zenith', axis=1)
        dc_power = mc.run_model(weather).results.dc/240
        return np.array(dc_power)  
    
    
    
    def parallelize_computation(self, latitudes, longitudes):

        
        grid_point_args = []
        for count_lat, lat in enumerate(latitudes):
            for count_lon, lon in enumerate(longitudes):
                grid_point_args.append((lat, lon))
        
        
        results = Parallel(n_jobs=-1,verbose=50, prefer="threads")(delayed(self.compute_solar)(lat, lon) for lat, lon in grid_point_args)

        solar_storage_array = np.empty((len(results[0]), len(latitudes), len(longitudes)))
        
        for i, result in enumerate(results):
            lat, lon = grid_point_args[i]
            pv_capacity_factor = result
            lat_idx = np.where(latitudes == lat)[0][0]
            lon_idx = np.where(longitudes == lon)[0][0]
            solar_storage_array[:, lat_idx, lon_idx] = pv_capacity_factor 
            
        
        solar_pv_results = xr.Dataset(data_vars={'Solar': (['time', 'latitude', 'longitude'], solar_storage_array)}, coords={'time': self.hourly_data,'latitude': latitudes,'longitude': longitudes})
        print(solar_pv_results)
                

        
        return solar_pv_results
        

    def compute_solar(self, lat, lon):
        
        return self.get_data(lat, lon)
    
    
    def store_dataset(self, data, filename):
        
        data.to_netcdf(filename, mode='w')







# Define the years you want to process
years = [2022]  # Add more years as needed
path = "/Users/lukehatton/Sync/MERRA2_SOLAR_VARS/"
output_path = "/Users/lukehatton/Documents/Imperial/Code/"
data_path = "/Users/lukehatton/Documents/Imperial/Code/Data"
lat_range = [49, 59]
lon_range = [-10, 5]

# Loop over the years
for year in years:
    # Construct the file names
    start_time = time.time()
    t2m_file = path + f"T2M.{year}.nc"
    swgdn_file = path + f"SWGDN.{year}.nc"
    output_file = output_path + f"NewEUSolarCF_{year}.nc" 

    # Create an instance of Get_Solar class for the current year
    get_solar_class = Get_Solar(t2m_file, swgdn_file, (data_path + "ETOPO_bathymetry.nc"))

    # Read the latitude and longitudes for the data
    ds = xr.open_dataset(t2m_file)
    latitudes = ds.lat.values
    longitudes = ds.lon.values

    # State start and end values for the latitude and longitude
    start_lat = lat_range[0]
    end_lat = lat_range[1]
    start_lon = lon_range[0]
    end_lon = lon_range[1]

    # Select the subset of latitude and longitude coordinates
    latitudes_sel = get_solar_class.get_coords_subset(latitudes, start_lat, end_lat)
    longitudes_sel = get_solar_class.get_coords_subset(longitudes, start_lon, end_lon)

    # Perform the computation in parallel for the selected coordinates
    solar_pv_results = get_solar_class.parallelize_computation(latitudes_sel, longitudes_sel)
    
    # Save the dataset
    get_solar_class.store_dataset(solar_pv_results, output_file)

    # Close the dataset
    solar_pv_results.close()
    
    end_time = time.time()
    running_time = end_time - start_time
    print(f"Running the parallelised code across the UK took {running_time} seconds")