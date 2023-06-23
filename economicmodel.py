import numpy as np
import xarray as xr
import glob
import pandas as pd
import pvlib
import matplotlib.pyplot as plt
# import cartopy.crs as ccrs



class Economic_Profile:
    " Class to generate/store the renewables profile from a weather profile (if supplied) and calculate the discounted costings relating to the renewables and electrolyser components "
    def __init__(self, renewables_capacity, percentage_wind, renew_op_cost, wind_capex, solar_capex=None, renewables_data = None, renew_discount_rate=None, 
                 lifetime=None, land_foundations = None, geodata = None, electrolyser_capacity = None, turbine_rating=None, turbine_diameter=None, elec_capex=None, elec_op_cost=None):
        """ Initialises the renewables_profile class
       
        Required:
        Renewables_capacity - installed capacity of renewables (solar and wind), in kW
        Percentage_wind - proportion of the renewables_capacity that is made up of wind
        renew_op_cost - assumption for the renewables OPEX, as a % of CAPEX
        wind_capex - capital expenditure for wind energy, in USD/kW
        
        Optional:
        solar_capex - capital expenditure figures for solar energy, in USD/kW
        renewables_data - capacity factor data for renewables, which is required in an hourly resolution
        renewable_discount_rate - individual discount rate for the renewables portion of the cost/output
        lifetime - number of years of operation
        land_foundation - capital cost of foundations for land-based wind, in USD/kW, taken from NREL 2021 Cost of Wind Energy Review
        turbine_diameter - diameter of modelled turbine, in m
        turbine_rating - power rating of modelled turbine, in kW
        
        """ 
        self.renew_op_cost = renew_op_cost
        self.wind_capex = wind_capex
        self.renewables_capacity = renewables_capacity
        self.percentage_wind = percentage_wind
        self.land_foundations = land_foundations
        self.turbine_diameter = turbine_diameter
        self.turbine_rating = turbine_rating

        if solar_capex is None:
            self.solar_capex = 0
        else:
            self.solar_capex = solar_capex

        if renew_discount_rate is not None:
            self.renew_discount_rate = renew_discount_rate
            
        if lifetime is not None:
            self.lifetime = lifetime
        
        if geodata is not None:
            self.geodata = geodata
            
        if electrolyser_capacity is not None:
            self.electrolyser_capacity = electrolyser_capacity
            
        if elec_capex is not None:
            self.elec_capex = elec_capex
        
        if elec_op_cost is not None:
            self.elec_op_cost = elec_op_cost
        
            

    def get_foundation_cost(self, data):
        """ Method to calculate the foundation cost for the wind turbine based on the depth of water, 
        using relationships from Bosch et al 2019 """
        
        depth_data = data
        
        # Set up relationships with depth
        a_parameter = [201, 114.24, 0]
        b_parameter = [612.93, -2270, 773.85]
        c_parameter = [411464, 531738, 680651]
        cutoff_data = [0, 25, 55, 1000]

        # initialise an empty array
        foundation_costs = xr.zeros_like(depth_data)

        # Use relationships with depth to estimate the foundation costs
        for i in range(len(cutoff_data) - 1):
            a = a_parameter[i]
            b = b_parameter[i]
            c = c_parameter[i]
            cutoff_start = cutoff_data[i]
            cutoff_end = cutoff_data[i + 1]

            # Apply cost relationship where depth is greater than the cutoff depth and not NaN
            foundation_costs = xr.where((depth_data > cutoff_start) & (depth_data <= cutoff_end), a * depth_data ** 2 + b * depth_data + c, foundation_costs)
        
        # Apply relationships for onshore (set foundation cost to zero) and offshore above the cutoff depth (>1000, N/A)
        foundation_costs = foundation_costs / 1000  # convert all into USD/kW
        foundation_costs = xr.where(depth_data < 0, self.land_foundations, foundation_costs)
        foundation_costs = xr.where(depth_data > 1000, np.nan, foundation_costs)
        

        return foundation_costs
    
    def get_transmission_cost(self, dist_data):
        """ Method to calculate the cost of electricity transmission (either through HVAC or HVDC) to shore, using
        relationships from the International Energy Agency's Wind Energy Outlook 2019 """
        dist = dist_data
        
        # Initialise empty arrays
        hvac = xr.zeros_like(dist)
        hvdc = xr.zeros_like(dist)      
        
        # Apply IEA relationships
        hvac = xr.where(dist > 0, (0.0085 * dist + 0.0568), 0) * 1000 # Conversion to USD/kW
        hvdc = xr.where(dist > 0, (0.0022 * dist + 0.3878), 0) * 1000 # Conversion to USD/kW
        transmission_costs = np.minimum(hvac, hvdc)
        #self.plot_data(transmission_costs, "Transmission Costs")
        
        return transmission_costs
    
    
    def get_interarray_costs(self, technology):
        """ Method to calculate the inter-array distance between wind turbines at each location and calculate the 
        cost of either AC cables or hydrogen pipelines between all of the wind turbines """
        
        # Get installed wind capacity
        wind_capacity = self.renewables_capacity
        turbine_rating = self.turbine_rating
        turbine_diameter = self.turbine_diameter
        
        # Calculate number of turbines
        n_turbines = wind_capacity / turbine_rating
        
        # Calculate interarray distance
        spacing = 7.5 * turbine_diameter / 1000 
        interarray_dist = n_turbines * spacing
        
        if technology == 'AC':
            interarray_cost = (0.0085 * interarray_dist + 0.0568) * 1000 * turbine_rating
        elif technology == 'Pipeline':
            interarray_dist = xr.DataArray(data = np.array(interarray_dist))
            interarray_cost = self.get_pipeline_cost(interarray_dist)
        
        return interarray_cost
    
    def get_pipeline_cost(self, dist_data):
        """ Method to calculate the cost of a hydrogen pipeline, based on the IEA's Future of Hydrogen Report"""
        
        # Set up constants relating to the electrolyser
        dist = dist_data
        electrolyser_size = self.electrolyser_capacity
        hydrogen_LHV = 33.3
        electrolyser_efficiency = 0.7
        
        # Calculate hydrogen capacity in tH2/day
        hydrogen_capacity = electrolyser_size * electrolyser_efficiency * 8760 / hydrogen_LHV / 1000 / 365.25 
        
        # Calculate the cost using a Linear relationship  from the IEA's Future of Hydrogen modelling assumptions
        IEA_cost = 807.38 * hydrogen_capacity + 426066  
        
        # Initialise empty arrays for storage
        h_pipeline = xr.zeros_like(dist_data)     
        
        # Apply cost relationship for hydrogen pipeline
        h_pipeline = xr.where(dist > 0, dist * IEA_cost, 0)
        pipeline_costs = h_pipeline
        
        
        return pipeline_costs
    
    
    def onshore_electrolysis(self, depth, dist_data):
        
        # Create a storage vector for the costs
        onshore_electrolysis_costs = xr.zeros_like(dist_data)
        
        # Calculate the offshore substation costs, taken from https://guidetoanoffshorewindfarm.com/wind-farm-costs 
        # and including the offshore substation total cost + installation cost
        offshore_substation_cost = 155 * 1.25 * self.renewables_capacity 
        
        
        # Calculate the onshore substation costs taken from https://guidetoanoffshorewindfarm.com/wind-farm-costs 
        # and including the offshore substation total cost + installation cost
        onshore_substation_cost = 55 * 1.25 * self.renewables_capacity
        
        # Calculate the interarray cable costs
        interarray_cable_costs = self.get_interarray_costs('AC')
        
        # Calculate the transmission costs to land
        transmission_costs = self.get_transmission_cost(dist_data) * self.renewables_capacity
        
        # Sum all costs relating to the configuration
        total_cost = offshore_substation_cost + onshore_substation_cost + interarray_cable_costs + transmission_costs
        onshore_electrolysis_costs = xr.where(depth < 1000,  total_cost, np.nan)
        
        return onshore_electrolysis_costs
    
    def offshore_electrolysis(self, depth, dist_data):
        
        # Create a storage vector                           
        offshore_electrolysis_costs = xr.zeros_like(dist_data)
        
        # Calculate the interarray cable costs
        interarray_cable_costs = self.get_interarray_costs('AC')
        
        # Calculate the offshore substation costs taken from https://guidetoanoffshorewindfarm.com/wind-farm-costs 
        # and including the offshore substation total cost + installation cost
        offshore_substation_costs = 155 * 1.25 * self.renewables_capacity + offshore_electrolysis_costs
        
        # Calculate the cost of an offshore platform for electrolysis, which are taken from 
        # https://guidetoanoffshorewindfarm.com/wind-farm-costs as the cost for facilities and structure
        # of an offshore substation and the installation cost
        offshore_platform_costs = 115 * 1.25 * self.electrolyser_capacity + offshore_electrolysis_costs
        
        # Calculate the pipeline costs
        pipeline_costs = self.get_pipeline_cost(dist_data)
        
        # Sum all costs
        total_costs = interarray_cable_costs + offshore_substation_costs + offshore_platform_costs + pipeline_costs
        offshore_electrolysis_costs = xr.where(depth < 1000, total_costs, np.nan)
                                   
        return offshore_electrolysis_costs
                                   
        
    def distributed_electrolysis(self, depth, dist_data):
        
        # Calculate the pipeline costs to shore
        pipeline_costs = self.get_pipeline_cost(dist_data)
        
        # Calculate the interarray pipeline costs
        interarray_pipeline_costs = self.get_interarray_costs('Pipeline')
        
        # Calculate the cost of a platform for central hydrogen collection, which are taken from 
        # https://guidetoanoffshorewindfarm.com/wind-farm-costs as the cost for facilities and structure
        # of an offshore substation and the installation cost
        offshore_platform_costs = 115 * 1.25 * self.electrolyser_capacity
        
        # Sum all costs
        total_costs = pipeline_costs + interarray_pipeline_costs + offshore_platform_costs
        distributed_costs = xr.where(depth < 1000, total_costs, np.nan)
        return distributed_costs
        
        
        
    def configuration_analysis(self, geodata):
    
        # Get depth and distance data
        dist_data = geodata['distance']
        depth = geodata['depth']
        offshore = geodata['offshore']
        
        # Use cost relationship with foundations and transmission
        foundation_costs_unit = self.get_foundation_cost(depth)
        
        # Sum the costs of turbine, transmission and foundation
        turbine_foundation_costs = foundation_costs_unit * self.renewables_capacity * self.percentage_wind
        wind_turbine_costs = self.wind_capex * self.renewables_capacity * self.percentage_wind
        nonconfig_costs = turbine_foundation_costs + wind_turbine_costs
        
        
        # Calculate the cost for each of the configurations
        
        onshore_electrolysis_cost = xr.where(offshore == True, self.onshore_electrolysis(depth, dist_data), 0)
        offshore_electrolysis_cost = xr.where(offshore == True, self.offshore_electrolysis(depth, dist_data), 0)
        distributed_electrolysis_cost = xr.where(offshore == True, self.distributed_electrolysis(depth, dist_data), 0)
        
            
        # Calculate total capital costs for each of the configurations
        
        onshore_electrolysis_tc = nonconfig_costs + onshore_electrolysis_cost
        offshore_electrolysis_tc = nonconfig_costs + offshore_electrolysis_cost
        distributed_electrolysis_tc = nonconfig_costs + distributed_electrolysis_cost
        
        # Calculate the minimium cost for each grid point
        storage_array = xr.zeros_like(dist_data)
        min_costs_initial = np.minimum(onshore_electrolysis_tc, offshore_electrolysis_tc)
        min_costs = np.minimum(min_costs_initial, distributed_electrolysis_tc)
        storage_array = xr.where(min_costs == onshore_electrolysis_tc, 'On.',
                            xr.where(min_costs == offshore_electrolysis_tc, 'Off.', 'Distr.'))
        #self.plot_data(min_costs, "Minimum Costs")
        df = storage_array.to_dataframe(name='values')
        #print(storage_array)
        #print(df['values'].value_counts())
        
        # Create a dataset with the three possible capital expenditures 
        data_vars = {'minimum capital costs': min_costs, 'minimum cost configuration' : storage_array, 'onshore electrolysis': onshore_electrolysis_tc, 
                     'offshore electrolysis': offshore_electrolysis_tc, 'distributed electrolysis': distributed_electrolysis_tc}
        coords = {'latitude': geodata.latitude,
                  'longitude': geodata.longitude}
        configuration_capital_costs = xr.Dataset(data_vars=data_vars, coords=coords)
        
        
        # Return the dataset
        return configuration_capital_costs
        

    
    def calculate_capital_depth_distance(self, geodata):
        "Updates the cost of the wind farm for each location depending on the depth and distance to shore"

        # Read geodata
        #geodata = xr.open_dataset('Europe_geodata.nc')

        # Use cost relationship with foundations and transmission
        foundation_costs_unit = self.get_foundation_cost(geodata['depth'])
        transmission_costs_unit = self.get_transmission_cost(geodata['distance'])
        
        # Sum the costs of turbine, transmission and foundation
        foundation_costs = foundation_costs_unit * self.renewables_capacity * self.percentage_wind
        wind_turbine_costs = self.wind_capex * self.renewables_capacity * self.percentage_wind
        transmission_costs = transmission_costs_unit * self.renewables_capacity * self.percentage_wind
        total_costs = foundation_costs + transmission_costs + wind_turbine_costs
        # self.plot_data(total_costs, "Total Capital Costs")

        # Save a capital cost for each location (lat/lon) and return this for use in the calculations
        data_vars = {'total capital costs': total_costs, 'foundation costs': foundation_costs, 
                     'wind turbine costs': wind_turbine_costs, 'transmission costs': transmission_costs}
        coords = {'latitude': geodata.latitude,
                  'longitude': geodata.longitude}
        capital_costs = xr.Dataset(data_vars=data_vars, coords=coords)
        return capital_costs

    
    def locational_operating_costs(self, capital_costs, renewables_data_yearly, cost_category):
        """Calculates the operating cost associated with the renewable or electrolyser capacity as a % of the CAPEX at
        each location"""
        
        # Calculate operating cost as a proportion of CAPEX
        if cost_category == 'Renew':
            operating_cost = capital_costs * self.renew_op_cost
        else:
            operating_cost = capital_costs * self.elec_op_cost
        
        # Assess size of renewables_profile_yearly and reproduce operating cost each year
        operating_cost_np = operating_cost.to_numpy()
        operating_costs_extended = operating_cost_np * np.ones_like(renewables_data_yearly)
        return operating_costs_extended
        
        
        
      

    def calculate_yearly_costs_using_depths(self, renewables_profile, geodata):
        "Calculates the yearly cost using cost relationships with water depth and distance to shore"
        
        # Read in the renewables_profile
        renewables_data_total = renewables_profile
        renewables_data_yearly = renewables_data_total.groupby('time.year').sum(dim='time')
        
        # Extract dimensions from the renewables_profile
        years = renewables_data_yearly.year
        latitudes = renewables_data_yearly.latitude
        longitudes = renewables_data_yearly.longitude

        # Need to account for CAPEX only in the 0th year
        lat_len = len(latitudes)
        lon_len = len(longitudes)
        years_len = len(years)
        new_year = [years[0]-1]
        years_appended = np.concatenate((new_year, years))
        zero_array = np.zeros((1, int(lat_len), int(lon_len)))

        
        # Create new arrays for storage
        renewables_array = xr.DataArray(renewables_data_yearly, dims=('year', 'latitude', 'longitude'),
                                        coords={'year': years,
                                                'latitude': latitudes,
                                                'longitude': longitudes})
        capital_costs_array = xr.DataArray(zero_array, dims=('year', 'latitude', 'longitude'),
                                        coords={'year': new_year,
                                                'latitude': latitudes,
                                                 'longitude': longitudes})
        # Calculate capital costs using depth
        costs_with_depth = self.calculate_capital_depth_distance(geodata)
        costs_with_config = self.configuration_analysis(geodata)
        
        
        # Transfer capital costs across to the capital costs array
        capital_costs = costs_with_config["minimum capital costs"]
        capital_costs_selected = capital_costs.sel(latitude=capital_costs_array.coords['latitude'],
                                                                  longitude=capital_costs_array.coords['longitude'])
        capital_costs_array[0, :, :] = capital_costs_selected #capital_costs_depth_selected  
        operating_costs_array = xr.DataArray(self.locational_operating_costs(capital_costs_selected, renewables_data_yearly, 'Renew'), dims=('year', 'latitude', 'longitude'), coords={'year': years,'latitude': latitudes,'longitude': longitudes})
        
        
        # Combine arrays to include zeroth year
        total_costs_array = xr.concat([capital_costs_array, operating_costs_array], dim = 'year')
        renewables_array = xr.concat([capital_costs_array * 0, renewables_array], dim = 'year')
        
        # Create a dataset with all the arrays
        data_vars = {'renewable_electricity': renewables_array,
                     'costs': total_costs_array, }
        coords = {'year': years_appended,
                  'latitude': latitudes,
                  'longitude': longitudes}
        ds = xr.Dataset(data_vars=data_vars, coords=coords)
        return ds
    
    
    def plot_data(self, data, name):
    
        # Set up data
        latitudes = data.latitude.values
        longitudes = data.longitude.values
        values = data.values

        # create the heatmap using pcolormesh
        fig = plt.figure(figsize=(20, 10))
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
        ax.set_title(name + ' heatmap')
        ax.coastlines()
        ax.stock_img()
        
        plt.show()



        

    def calculate_electrolyser_capex(self, geodata, capacity=None):
        "Calculates the capital cost associated with the electrolyser, doubling the capital cost for offshore locations"
        # Get offshore mask
        offshore_mask = geodata['offshore']
        
        # Check if capacity is specified, otherwise use default
        if capacity is not None:
            electrolyser_capacity = capacity
        else:
            electrolyser_capacity = self.electrolyser_capacity
        
        # Adjust capital expenditure for electrolyser when offshore
        capex_adjustment = xr.where(offshore_mask == True, 2, 1)
        
        # Calculate locational capital expenditure relating to the electrolyser
        capital_cost = capex_adjustment * self.elec_capex * electrolyser_capacity
        
        return capital_cost
    
    
    
    def calculate_combined_capital_costs(self, renewables_profile, geodata, capacity=None):
        "Calculates the yearly cost for both renewables and the electrolyser components using cost relationships with water depth and distance to shore"
        
        # Read in the renewables_profile
        renewables_data_total = renewables_profile
        renewables_data_yearly = renewables_data_total.groupby('time.year').sum(dim='time')
        
        # Extract dimensions from the renewables_profile
        years = renewables_data_yearly.year
        latitudes = renewables_data_yearly.latitude
        longitudes = renewables_data_yearly.longitude

        # Need to account for CAPEX only in the 0th year
        lat_len = len(latitudes)
        lon_len = len(longitudes)
        years_len = len(years)
        new_year = [years[0]-1]
        years_appended = np.concatenate((new_year, years))
        zero_array_renew = np.zeros((1, int(lat_len), int(lon_len)))
        zero_array_elec = np.zeros((1, int(lat_len), int(lon_len)))
        
        # Create new arrays for storage
        renewables_array = xr.DataArray(renewables_data_yearly, dims=('year', 'latitude', 'longitude'),
                                        coords={'year': years,
                                                'latitude': latitudes,
                                                'longitude': longitudes})
        renew_costs_array = xr.DataArray(zero_array_renew, dims=('year', 'latitude', 'longitude'),
                                        coords={'year': new_year,
                                                'latitude': latitudes,
                                                 'longitude': longitudes})
        elec_costs_array = xr.DataArray(zero_array_elec, dims=('year', 'latitude', 'longitude'),
                                        coords={'year': new_year,
                                                'latitude': latitudes,
                                                 'longitude': longitudes})
        
        
        # Calculate renewable capital costs using depth
        renew_config_costs = self.configuration_analysis(geodata)
        renew_capital_costs = renew_config_costs["minimum capital costs"]
        
        # Calculate electrolyser capital costs
        elec_capital_costs = self.calculate_electrolyser_capex(geodata, capacity)
        
        # Transfer capital costs across to the relevant cost arrays
        elec_costs_array[0, :, :] = elec_capital_costs
        renew_costs_array[0, :, :] = renew_capital_costs
        
        # Calculate the operating costs 
        renew_op_costs_array = xr.DataArray(self.locational_operating_costs(renew_capital_costs, renewables_data_yearly, 'Renew'), dims=('year', 'latitude', 'longitude'), coords={'year': years,'latitude': latitudes,'longitude': longitudes})
        elec_op_costs_array = xr.DataArray(self.locational_operating_costs(elec_capital_costs, renewables_data_yearly, 'Elec'), dims=('year', 'latitude', 'longitude'), coords={'year': years,'latitude': latitudes,'longitude': longitudes})

        
        # Combine capital and operating cost arrays
        renew_costs_combined = xr.concat([renew_costs_array, renew_op_costs_array], dim='year')
        elec_costs_combined = xr.concat([elec_costs_array, elec_op_costs_array], dim = 'year')
        total_costs_array = elec_costs_combined + renew_costs_combined
        renewables_array = xr.concat([renew_costs_array * 0, renewables_array], dim = 'year')

        
        # Create a dataset with all the arrays
        data_vars = {'renewable_electricity': renewables_array,
                     'renewable costs': renew_costs_array,
                     'electrolyser costs': elec_costs_array,
                     'total costs': total_costs_array}
        coords = {'year': years_appended,
                  'latitude': latitudes,
                  'longitude': longitudes}
        yearly_costs = xr.Dataset(data_vars=data_vars, coords=coords)
        return yearly_costs