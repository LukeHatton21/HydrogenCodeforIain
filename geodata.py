import geopy.distance
import xarray as xr
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import numpy as np
# import cartopy.crs as ccrs
from scipy.spatial.distance import cdist
import regionmask


class Global_Data:
    # All locations with depth, category and distance to shore computed
    def __init__(self, bathymetry_data, dist2shore, countries, datafile):
        self.bathymetry_data = xr.open_dataset(bathymetry_data)
        self.ds = datafile
        self.dist2shore = xr.open_dataset(dist2shore)
        self.countries_data = xr.open_dataset(countries)


    def get_depth_in_required_resolution(self):

        # Read Bathymetry file
        bathymetry = self.bathymetry_data
        #print(bathymetry)

        # Read renewables file and identify resolution
        target_data = self.ds

        
        # Reindex using the xarray reindex function
        bathymetry_renamed = bathymetry.rename({'lat': 'latitude','lon': 'longitude'})
        new_coords = {'latitude': target_data.latitude, 'longitude': target_data.longitude}
        bathymetry_resampled = bathymetry_renamed.reindex(new_coords, method='nearest')

        

        
        # Create new dataset with depth values
        data_vars = {'depth': bathymetry_resampled['z']}
        coords = {'latitude': bathymetry_resampled.latitude,
                  'longitude': bathymetry_resampled.longitude}
        bathymetry_output = xr.Dataset(data_vars=data_vars, coords=coords)

        # Convert from bathymetry to depth
        bathymetry_new = xr.where(bathymetry_output > 0, 0, bathymetry_output)

        # Plot data
        #self.plot_data(bathymetry_new) 
        
        return bathymetry_new
    
    
    def get_countries_in_required_resolution(self):

        # Read Countries file
        countries = self.countries_data


        # Read renewables file and identify resolution
        target_data = self.ds

        
        # Reindex using the xarray reindex function
        new_coords = {'latitude': target_data.latitude, 'longitude': target_data.longitude}
        countries_resampled = countries.reindex(new_coords, method='nearest')

        
        # Create new dataset with countries
        data_vars = {'land': countries_resampled['land'], 'sea': countries_resampled['sea']}
        coords = {'latitude': target_data.latitude,
                  'longitude': target_data.longitude}
        countries_output = xr.Dataset(data_vars=data_vars, coords=coords)

        # Plot data
        # self.plot_data(countries_output) 
        
        return countries_output

    def get_offshore_onshore_mask(self, data):

        # Read depth file
        bathymetry = data
        # Initialise an array like the depth file
        offshore_mask = xr.zeros_like(bathymetry)
        
        # Where bathymetry > 0, set equal to False, Where bathymetry < 0, set equal to True
        offshore_mask = xr.where(bathymetry < 0, 1, 0)
        offshore_mask = offshore_mask.rename_vars({'depth': 'offshore'})
        #self.plot_data(offshore_mask)
        
        # Return the output
        return offshore_mask
    
    def get_offshore_mask(self):
    
        longitude = np.arange(-179.5, 180, 0.1)
        latitude = np.arange(-89.5, 90, 0.1)
        mask = regionmask.defined_regions.natural_earth_v5_0_0.land_10.mask(longitude, latitude)
        mask = mask.rename({'lat': 'latitude','lon': 'longitude'})
        
        # Read renewables file and identify resolution
        target_data = self.ds
        
        # Reindex using the xarray reindex function
        new_coords = {'latitude': target_data.latitude, 'longitude': target_data.longitude}
        onshore_mask = mask.reindex(new_coords, method='nearest')
        offshore_mask = xr.where(onshore_mask == 0, 0, 1)
        
        data_vars = {'offshore': offshore_mask}
        coords = {'latitude': offshore_mask.latitude,
                  'longitude': offshore_mask.longitude}
        offshore_dataset = xr.Dataset(data_vars=data_vars, coords=coords)
        #print(offshore_dataset)
        #self.plot_data(offshore_dataset)

        return offshore_dataset

    def get_distance_to_shore(self):
        
        # Read Distance file
        dist2shore = self.dist2shore
        #print(dist2shore)
        
        # Read renewables file and identify resolution
        target_data = self.ds

        # Resample the bathymetry file along the latitude dimension
        resampled_dist2shore_lat = dist2shore.interp(Latitude=target_data.latitude, method='nearest')

        # Resample the bathymetry file along the longitude dimension
        resampled_dist2shore = resampled_dist2shore_lat.interp(Longitude=target_data.longitude, method='nearest')
        
        
        data_vars = {'distance': resampled_dist2shore['Distance']}
        coords = {'latitude': resampled_dist2shore.latitude,
                  'longitude': resampled_dist2shore.longitude}
        dist2shore_output = xr.Dataset(data_vars=data_vars, coords=coords)
        #print(dist2shore_output)

        # drop unused coordinates
        dist2shore_output = dist2shore_output.drop(['Latitude', 'Longitude'])

        # Plot the data
        #self.plot_data(dist2shore_output)
        
        return dist2shore_output
        
        
    def get_all_data_variables(self):

        # Call each function in turn
        depth = self.get_depth_in_required_resolution()
        offshore_mask = self.get_offshore_mask()
        distance2shore_raw = self.get_distance_to_shore()
        distance_to_shore = self.correct_distance_shore(offshore_mask, distance2shore_raw)
        countries = self.get_countries_in_required_resolution()
        
        # Combine the files , distance_to_shore]
        merged_dataset = xr.merge([depth, offshore_mask, distance_to_shore, countries])
        print(merged_dataset)
        return merged_dataset
    
    def correct_distance_shore(self, offshore_mask, dist2shore_data):
    
        offshore_ds = offshore_mask
        dist2shore_ds = dist2shore_data
        
        distance_to_shore = xr.where(offshore_ds['offshore'] == True, dist2shore_ds, 0)
        #print(distance_to_shore)
        #self.plot_data(distance_to_shore)
        return distance_to_shore

    def plot_data(self, data):
    
        # Get the name of the data variable
        varname = list(data.data_vars.keys())[0]

        # Set up data
        latitudes = data.latitude.values
        longitudes = data.longitude.values
        values = data[varname].values

        # create the heatmap using pcolormesh
        fig = plt.figure(figsize=(30, 15))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
        heatmap = ax.pcolormesh(longitudes, latitudes, values, transform=ccrs.PlateCarree(), cmap='plasma')
        fig.colorbar(heatmap, ax=ax, shrink=0.5)


        # set the extent and aspect ratio of the plot
        ax.set_extent([longitudes.min(), longitudes.max(), latitudes.min(), latitudes.max()], crs=ccrs.PlateCarree())
        aspect_ratio = (latitudes.max() - latitudes.min()) / (longitudes.max() - longitudes.min())
        ax.set_aspect(aspect_ratio)

        # add axis labels and a title
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Values heatmap')
        ax.set_title('{} heatmap'.format(varname))
        ax.coastlines()
        ax.stock_img()
        
        plt.show()

        


#datafile = xr.open_dataset("UKWindCF2001_2005.nc")
#print(datafile)
#data = Global_Data("./Data/ETOPO_bathymetry.nc","./Data/distance2shore.nc", datafile).get_all_data_variables()
#print(data)







