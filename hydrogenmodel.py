import numpy as np
import xarray as xr
import time
import csv
# import cartopy.crs as ccrs
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math
import pandas as pd
from scipy.optimize import minimize
from scipy.optimize import basinhopping
from joblib import Parallel, delayed
from electrolyser import Electrolyser
from economicmodel import Economic_Profile
from geodata import Global_Data
from filepreprocessor import All_Files
from datetime import datetime  



class HydrogenModel:
    def __init__(self, dataset,  discount_rate=None, renewables_capacity=None, wind_capex=None, percentage_wind=None, renewable_op_cost=None,
                 params_file_elec=None, params_file_renew=None, data_path=None, output_folder=None, efficiency=None, electrolyser_type=None, electrolyser_capacity=None, elec_capex=None,
                 elec_op_cost=None, elec_discount_rate=None, renew_discount_rate=None, lifetime=None, years=None):
        if params_file_elec is not None:
            self.electrolyser_class = self.parameters_from_csv(params_file_elec, 'electrolyser')
        else:
            self.electrolyser = Electrolyser(electrolyser_capacity, efficiency, electrolyser_type, elec_capex, elec_op_cost)

        if params_file_renew is not None:
            self.economic_profile_class = self.parameters_from_csv(params_file_renew, 'renewables')
        else:
            self.percentage_wind = percentage_wind
            self.renewable_op_cost = renewable_op_cost
            self.wind_capex = wind_capex
            self.renewable_profile_class = Economic_Profile(renewables_capacity, percentage_wind,
                                                              renewable_op_cost, wind_capex, renewables_data=dataset,lifetime=None)
        self.geodata_class = Global_Data((data_path + "ETOPO_bathymetry.nc"),(data_path+"distance2shore.nc"), (data_path+"country_grids.nc"), dataset)
        self.geodata = self.geodata_class.get_all_data_variables()
        self.renewables_data = dataset
        self.economic_profile_class.electrolyser_capacity = self.electrolyser_capacity
        self.electrolyser_class.elec_capacity_array = xr.zeros_like(dataset) + self.electrolyser_capacity
        self.discount_rate = discount_rate
        self.lifetime = lifetime
        self.years = years
        self.output_folder = output_folder
        self.country_wacc_mapping = pd.read_csv((data_path + "country_wacc.csv"))
        self.country_data = xr.open_dataset((data_path + "country_grids.nc"))
        print("Setting up the Hydrogen Model Class")

            
    def parameters_from_csv(self, file_path, class_name):
        try:
            with open(file_path, 'r') as file:
                reader = csv.reader(file)
                next(reader)  # skip the header row
                params = {}
                for row in reader:
                    if len(row) == 2:
                        param_name = row[0].strip()
                        param_value = row[1].strip()
                        try:
                            param_value = float(param_value)
                            if param_value.is_integer():
                                param_value = int(param_value)
                        except ValueError:
                            pass
                        params[param_name] = param_value
            if class_name == 'renewables':
                class_initiated = Economic_Profile(**params)
                print("Parameters required to set up the Renewable Class have been read in from the CSV file")
                self.renewables_capacity = params['renewables_capacity']
            elif class_name == 'electrolyser':
                class_initiated = Electrolyser(**params)
                self.electrolyser_capacity = params['electrolyser_capacity']
                self.electrolyser_capex = params['elec_capex']
                print("Parameters required to set up the Electrolyser Class have been read in from the CSV file")
            return class_initiated
        except ValueError as e:
            print("Error: {}".format(e))
        except TypeError as f:
            print("Error: {}".format(f))

    def get_levelised_cost(self, renewables_data=None):
        
        start_time = time.time()
        # Setup variables stored in the Hydrogen Model Class
        lifetime = self.lifetime
        geodata = self.geodata
        
        # Call the Renewables Profile and Electrolyser Classes
        if renewables_data is None:
            renewables_profile = self.renewables_data * self.renewables_capacity
        else: 
            renewables_profile = renewables_data * self.renewables_capacity
        print("Calculating the hydrogen output at an hourly resolution")
        electrolyser_yearly_output = self.electrolyser_class.calculate_yearly_output(renewables_profile)
        print("Conducting the yearly economic analysis")
        combined_yearly_output = self.economic_profile_class.calculate_combined_capital_costs(renewables_profile, geodata)

        
        # Access relevant yearly variables for the LCOH calculation
        hydrogen_produced_yearly = electrolyser_yearly_output['hydrogen_produced']
        electrolyser_costs_yearly = combined_yearly_output['electrolyser costs']
        renewables_costs_yearly = combined_yearly_output['renewable costs']
        
        # Read the dimensions of the yearly output
        years = electrolyser_yearly_output.sizes['year'] - 1
        lat_num = electrolyser_yearly_output.sizes['latitude']
        lon_num = electrolyser_yearly_output.sizes['longitude']
        
        # If the size of the data is less than the lifetime, duplicate the data
        if years < lifetime:
            n_duplicates = round(lifetime / years)
            hydrogen_produced_yearly = self.extend_to_lifetime(hydrogen_produced_yearly, lifetime)
            electrolyser_costs_yearly = self.extend_to_lifetime(electrolyser_costs_yearly, lifetime)
            renewables_costs_yearly = self.extend_to_lifetime(renewables_costs_yearly, lifetime)
            
        # Check whether individual discount rates are provided, if so then discount the costs separately    
        if self.economic_profile_class.renew_discount_rate is not None:
            print("Discounting Renewable and Electrolyser costs separately")
            #discounted_renew_costs = self.cashflow_discount(renewables_costs_yearly, self.economic_profile_class.renew_discount_rate)
            discounted_renew_costs = self.country_wacc_discounts(renewables_costs_yearly)
            print("Renewable Costs Discounted")
            discounted_elec_costs = self.country_wacc_discounts(electrolyser_costs_yearly, 1)
            #discounted_elec_costs = self.cashflow_discount(electrolyser_costs_yearly, self.electrolyser_class.elec_discount_rate)
            print("Electrolyser Costs Discounted")
            discounted_output = self.country_wacc_discounts(hydrogen_produced_yearly, 1)
            #discounted_output = self.cashflow_discount(hydrogen_produced_yearly, self.electrolyser_class.elec_discount_rate)
            print("Hydrogen Output Discounted")
            discounted_costs = discounted_renew_costs + discounted_elec_costs
        else:
            print("Discounting Renewable and Eectrolyser costs with the same discount rate")
            total_costs_yearly = electrolyser_costs_yearly + renewables_costs_yearly
            discounted_costs = self.cashflow_discount(total_costs_yearly, self.discount_rate)
            discounted_output = self.cashflow_discount(hydrogen_produced_yearly, self.discount_rate)
            
        # Sum the discounted costs and hydrogen produced
        discounted_costs_sum = np.sum(discounted_costs, axis=0)
        hydrogen_produced_sum = np.sum(discounted_output, axis=0)
        
        # Calculate the average annual hydrogen production in tonnes per annum
        annual_hydrogen = hydrogen_produced_yearly.mean(dim='year') / 1000
        
        
        # Calculate the levelised costs, filtering to account for the locations that are too far from the shoreline
        levelised_cost_raw = np.divide(discounted_costs_sum, hydrogen_produced_sum)
        levelised_cost = xr.where(levelised_cost_raw == 0, np.nan, levelised_cost_raw)
        
        # Print time taken to run
        end_time = time.time()
        method_time = end_time - start_time
        print(f"Code took {method_time:.2f} seconds to run")
        
        # Map the individual costs
        print("Calculating the Levelised Cost of Hydrogen")
        print("Discounted Costs Mapping")
        # self.plot_data(discounted_costs_sum.values, discounted_costs_sum.latitude.values, discounted_costs_sum.longitude.values, "Discounted Costs" )
        print("Discounted Hydrogen Production Mapping")
        # self.plot_data(hydrogen_produced_sum.values, discounted_costs_sum.latitude.values, discounted_costs_sum.longitude.values, "Discounted Hydrogen Production" )
        
        
        return levelised_cost, annual_hydrogen
    
    
    def cashflow_discount(self, data, rate):
        # Read number of years, latitudes and longitudes
        years = data.sizes['year']
        lat_num = data.sizes['latitude']
        lon_num = data.sizes['longitude']
    
        # Create array for storage
        discounted_data = xr.zeros_like(data)
        
        # Apply discounting using nested for loops
        for year in range(years):
                #print(f"{year:.0f} discounting complete")
                for lat in range(lat_num):
                    for lon in range(lon_num):
                        discounted_data[year, lat, lon] = data[year, lat, lon] / (
                                (1 + rate) ** year)
                        
        return discounted_data
    
    def country_wacc_discounts(self, data, electrolyser=None):
        # Read number of years, latitudes and longitudes
        years = data.sizes['year']
        lat_num = data.sizes['latitude']
        lon_num = data.sizes['longitude']
        latitudes = data.latitude.values
        longitudes = data.longitude.values
    
        # Create array for storage
        discounted_data = xr.zeros_like(data)
        
        for year in range(years):
            for count_lat, lat in enumerate(latitudes):
                for count_lon, lon in enumerate(longitudes):
                    rate = self.get_country_wacc(lat, lon)
                    if electrolyser is not None:
                        rate = rate + 0.05
                    if np.isnan(rate) or rate == 0:
                        discounted_data[year, count_lat, count_lon] = data[year, count_lat, count_lon] / (
                                (1 + self.economic_profile_class.renew_discount_rate) ** year)
                    else:
                        discounted_data[year, count_lat, count_lon] = data[year, count_lat, count_lon] / (
                                (1 + rate) ** year)
       
                        
        return discounted_data
    
    def get_country_wacc(self, lat, lon):
    
        # Retrieve CSV file with mapping of countries and waccs
        country_wacc_mappings = self.country_wacc_mapping
        geodata = self.geodata
        land_value = geodata['land'].sel({'latitude': lat, 'longitude': lon}).values 
        
        
        # Retrieve offshore mask
        if np.isnan(land_value):
            sea_value = geodata['sea'].sel({'latitude': lat, 'longitude': lon}).values 
            if np.isnan(sea_value):
                country_wacc = np.nan
            else:
                country_row = country_wacc_mappings.loc[country_wacc_mappings['index'] == sea_value]
                country_wacc = country_row.loc[country_row.index[0],'offshore wacc']
                    # Lookup country code in the mapping
        else: 
            country_row = country_wacc_mappings.loc[country_wacc_mappings['index'] == land_value]
            country_wacc = country_row.loc[country_row.index[0],'onshore wacc']
        
    
        return country_wacc
        
    def extend_to_lifetime(self, data, lifetime):
        
        years = data.sizes['year'] - 1
        n_duplications = round(lifetime/years)
        remainder = lifetime % years
        
        # Separate 0th year and operation years 
        zeroth_year = data[0:1,:,:]
        operational_years = data[1:, :, :]
        remainder_index = remainder+1
        remainder_years = data[1:remainder_index, :, :]
        
        # Duplicate a set amount of times and then concenate over operational years
        duplicated_data = xr.concat([operational_years] * n_duplications, dim ='year')
        first_year = duplicated_data['year'][0]
        final_year = duplicated_data['year'][-1]
        year_range = final_year - first_year + 1
        duplicated_with_r = xr.concat((duplicated_data, remainder_years), dim = 'year')
        new_year_range = np.arange(first_year, (first_year + n_duplications * year_range + remainder), step = 1)
        duplicated_data =  duplicated_with_r .assign_coords({'year': new_year_range})
        
        # Concenate with the 0th year
        combined_data = xr.concat((zeroth_year, duplicated_data), dim ='year')
        # Return data
        return combined_data

    
    

    def print_results_separately(self, levelised_cost):
    
        # Get geodata
        geodata = self.geodata
        
        # Get latitudes, longitudes and raw costs
        print("Mapping the onshore levelised cost of hydrogen")
        latitudes = levelised_cost.latitude.values
        longitudes = levelised_cost.longitude.values
        costs = levelised_cost.values
        
        # Separate onshore and offshore
        offshore_mask = geodata['offshore'].values
        offshore_costs = xr.where(offshore_mask == True, costs, np.nan)
        onshore_costs = xr.where(offshore_mask == False, costs, np.nan)
        valid_offshore = offshore_costs[np.isfinite(offshore_costs)]
        valid_onshore = onshore_costs[np.isfinite(onshore_costs)]
        
        # Print results for offshore
        try:
            print("Results for offshore locations:")
            self.print_info_from_results(valid_offshore)
            #self.plot_data(offshore_costs, latitudes, longitudes, "LCOH (USD/kg), offshore locations", "offshore_lcoh", 1)
        except: 
            print("No offshore locations")
        
        
        # Print results for onshore
        try: 
            print("Results for onshore locations:")
            self.print_info_from_results(valid_onshore)
            #self.plot_data(onshore_costs, latitudes, longitudes, "LCOH (USD/kg), onshore locations", "onshore_lcoh_", 1)
        except:
            print("No onshore locations")
        
        
        # Print combined results for onshore and offshore
        print("Results for all locations:")
        self.print_info_from_results(costs)
        # self.plot_data(costs, latitudes, longitudes, "LCOH (USD/kg), all locations", "all_locations_lcoh_", 1)
        
    
 

    def print_info_from_results(self, data):
        
        # Get valid values
        valid_values = data[np.isfinite(data)]
        vmin = np.min(valid_values)
        vmax = np.max(valid_values)
        min_value = vmin
        max_value = vmax
        mean_value = np.mean(valid_values)
        renewables_cap = self.renewables_capacity/1000
        electrolyser_cap = self.electrolyser_capacity/1000
        print(f"The installed renewables capacity was: {renewables_cap:.2f} MW")
        print(f"The installed electrolyser capacity was: {electrolyser_cap:.2f} MW")
        print(f"The average LCOH was: {mean_value:.2f} USD/kg")
        print(f"The range of achievable LCOHs were: {min_value:.2f} - {max_value:.2f} USD/kg")
        
        
    
    def plot_data(self, values, latitudes, longitudes, name, filename=None, increment=None):
        

        # create the heatmap using pcolormesh
        fig = plt.figure(figsize=(30, 15))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
        heatmap = ax.pcolormesh(longitudes, latitudes, values, norm=colors.LogNorm(), transform=ccrs.PlateCarree(), cmap='YlOrRd')
        end = values.max()+1
        if increment is not None:
            ranges = np.arange(0, 16, 1)
            cb = fig.colorbar(heatmap, ax=ax, shrink=0.5, ticks=ranges)
            cb.ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=False, useOffset=False))
            cb.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0f'))
        else:
            cb = fig.colorbar(heatmap, ax=ax, shrink=0.5)

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
        
        if filename is not None:
            time_stamp = time.time()
            date_time = datetime.fromtimestamp(time_stamp)
            str_date_time = date_time.strftime("%d-%m-%Y-%H:%M:%S")
            start_year = years[0]
            end_year = years[-1]
            years_str = str(start_year) + '_' + str(end_year)
            plt.savefig(self.output_folder + filename  + '_' + years_str + '_' + str_date_time + '.png')
        
        plt.show()
        
        
    
    def get_supply_curves(self, levelised_costs, annual_production):
    
        # Get geodata
        geodata = self.geodata
        
        # Calculate area of each grid point in kms 
        latitudes = annual_production.latitude.values
        longitudes = annual_production.longitude.values
        grid_areas = self.get_areas(annual_production)
        
        # Set out constants
        turbine_density = 6520 # kW/km2
        installed_capacity = self.economic_profile_class.renewables_capacity * self.economic_profile_class.percentage_wind
        
        # Scale annual hydrogen production by turbine density
        max_installed_capacity = turbine_density * grid_areas['area']
        ratios = max_installed_capacity / installed_capacity
        technical_hydrogen_potential = annual_production * ratios
        
        # Create new dataset with cost and production volume
        data_vars = {'hydrogen technical potential': technical_hydrogen_potential,
                     'levelised cost': levelised_costs}
        coords = {'latitude': latitudes,
                  'longitude': longitudes}
        supply_curve_ds = xr.Dataset(data_vars=data_vars, coords=coords)
        
        # Extract values for cost and annual production
        cost_values = supply_curve_ds['levelised cost'].values.ravel()
        production_values = supply_curve_ds['hydrogen technical potential'].values.ravel()

        sorted_indices = np.argsort(cost_values)
        sorted_cost = cost_values[sorted_indices]
        sorted_production = production_values[sorted_indices]
        cost_rounded = sorted_cost.round(decimals=1)

        unique_costs, unique_indices = np.unique(cost_rounded, return_index=True)
        unique_production = np.add.reduceat(sorted_production, unique_indices)
        cumulative_production = np.cumsum(np.pad(unique_production, (1,0), "constant"))
        cumulative_production = cumulative_production[0:-1]

        plt.figure(figsize=(20,8))
        plt.bar(cumulative_production/1e+06, unique_costs, align='edge', width=unique_production/1e+06)  # Plotting cumulative amounts against unique costs
        plt.ylabel('Levelised Cost (USD/kg H2)')
        plt.xlabel('Annual Hydrogen Production (million tonnes per annum)')
        plt.axis([0, math.ceil(cumulative_production.max()/1e+06), 0, math.ceil(unique_costs.max())])
        plt.xticks(np.arange(0, math.ceil(cumulative_production.max()/1e+06), step=50))
        plt.yticks(np.arange(0, math.ceil(unique_costs.max()), step=1))
        plt.axvline(x = 94, color = 'b', label = 'Annual Hydrogen Demand (IEA 2021)')
        plt.axhline(y = 1, color = 'r', linestyle='--', label = 'LB Cost of Hydrogen from natural gas (IEA 2021)')
        plt.axhline(y = 2.4, color = 'r', linestyle='--', label = 'UB Cost of Hydrogen from natural gas (IEA 2021)')
        plt.legend(bbox_to_anchor = (1.0, 1), loc = 'upper right')
 
        plt.title('Supply-cost curve of green hydrogen')
        plt.show()
        
        return supply_curve_ds
    
    
    def get_areas(self, annual_production):
        
        latitudes = annual_production.latitude.values
        longitudes = annual_production.longitude.values

        # Add an extra value to latitude and longitude coordinates
        latitudes_extended = np.append(latitudes, latitudes[-1] + np.diff(latitudes)[-1])
        longitudes_extended = np.append(longitudes, longitudes[-1] + np.diff(longitudes)[-1])

        # Calculate the differences between consecutive latitude and longitude points
        dlat_extended = np.diff(latitudes_extended)
        dlon_extended = np.diff(longitudes_extended)
        
        # Calculate the Earth's radius in kms
        radius = 6371

        # Compute the mean latitude value for each grid cell
        mean_latitudes_extended = (latitudes_extended[:-1] + latitudes_extended[1:]) / 2
        mean_latitudes_2d = mean_latitudes_extended[:, np.newaxis]

        # Convert the latitude differences and longitude differences from degrees to radians
        dlat_rad_extended = np.radians(dlat_extended)
        dlon_rad_extended = np.radians(dlon_extended)

        # Compute the area of each grid cell using the Haversine formula
        areas_extended = np.outer(dlat_rad_extended, dlon_rad_extended) * (radius ** 2) * np.cos(np.radians(mean_latitudes_2d))

        # Create a dataset with the three possible capital expenditures 
        area_dataset = xr.Dataset()
        area_dataset['latitude'] = latitudes
        area_dataset['longitude'] = longitudes
        area_dataset['area'] = (['latitude', 'longitude'], areas_extended, {'latitude': latitudes, 'longitude': longitudes})
        
        return area_dataset
    
    
    def new_cost_function(self, capacity, renewables_gridpoint):
        levelised_cost, hydrogen_produced = self.new_get_levelised_cost_optimisation(renewables_data=renewables_gridpoint, capacity=capacity)
        return levelised_cost.mean()
    
    
    def new_get_levelised_cost_optimisation(self, renewables_data, capacity):
        
        # Extract latitudes and longitudes
        latitude = renewables_data.latitude
        longitude = renewables_data.longitude
        
        # Setup variables stored in the Hydrogen Model Class
        lifetime = self.lifetime
        geodata = self.geodata
        geodata = geodata.sel(latitude=latitude, longitude=longitude)
       
        
        # Call the Renewables Profile and Electrolyser Classes
        renewables_profile = renewables_data * self.renewables_capacity
        electrolyser_yearly_output = self.electrolyser_class.calculate_yearly_output(renewables_profile, capacity)
        combined_yearly_output = self.economic_profile_class.calculate_combined_capital_costs(renewables_profile, geodata)

        
        # Access relevant yearly variables for the LCOH calculation
        hydrogen_produced_yearly = electrolyser_yearly_output['hydrogen_produced']
        electrolyser_costs_yearly = combined_yearly_output['electrolyser costs']
        renewables_costs_yearly = combined_yearly_output['renewable costs']
        
        
        # Read the dimensions of the yearly output
        years = electrolyser_yearly_output.sizes['year'] - 1
        lat_num = electrolyser_yearly_output.sizes['latitude']
        lon_num = electrolyser_yearly_output.sizes['longitude']
        
        # If the size of the data is less than the lifetime, duplicate the data
        if years < lifetime:
            n_duplicates = round(lifetime / years)
            hydrogen_produced_yearly = self.extend_to_lifetime(hydrogen_produced_yearly, lifetime)
            electrolyser_costs_yearly = self.extend_to_lifetime(electrolyser_costs_yearly, lifetime)
            renewables_costs_yearly = self.extend_to_lifetime(renewables_costs_yearly, lifetime)
        
        # Discount renewables and electrolyser costs separately
        discounted_renew_costs = self.country_wacc_discounts(renewables_costs_yearly)
        discounted_elec_costs = self.cashflow_discount(electrolyser_costs_yearly, self.electrolyser_class.elec_discount_rate)
        discounted_output = self.cashflow_discount(hydrogen_produced_yearly, self.electrolyser_class.elec_discount_rate)
        print(f"Electrolyser Capacity: {capacity} kW")
        discounted_costs = discounted_renew_costs + discounted_elec_costs
            
            
        # Sum the discounted costs and hydrogen produced
        discounted_costs_sum = np.sum(discounted_costs, axis=0)
        hydrogen_produced_sum = np.sum(discounted_output, axis=0)
        
        # Calculate the average annual hydrogen production in tonnes per annum
        annual_hydrogen = hydrogen_produced_yearly.mean(dim='year') / 1000
        
        # Calculate the levelised costs, filtering to account for the locations that are too far from the shoreline
        levelised_cost_raw = np.divide(discounted_costs_sum, hydrogen_produced_sum)
        levelised_cost = xr.where(levelised_cost_raw == 0, np.nan, levelised_cost_raw)
        return levelised_cost, annual_hydrogen 


    def process_grid_point(self, lat, lon):
        
        loop_start = time.time()
        # Get renewables data at each gridpoint
        renewables_gridpoint = self.renewables_data.sel(longitude = lon, latitude=lat)
        renewables_gridpoint = renewables_gridpoint.expand_dims(latitude=[lat], longitude=[lon])
        renewables_gridpoint = renewables_gridpoint.transpose("time", "latitude", "longitude")
                    
        # Evaluate nature of gridpoint
        offshore_status = self.geodata['offshore'].sel(longitude = lon, latitude=lat)      
                    
        # Set up optimisation problem
        if offshore_status == 1:
            initial_guess = [0.8 * self.renewables_capacity]
        else: 
            initial_guess = [0.6 * self.renewables_capacity]
        low_bound = 0.45 * self.renewables_capacity
        upp_bound = 1.0 * self.renewables_capacity
        bounds = [(low_bound, upp_bound)]
            
                    
        # Use SciPy's Minimisation Function
        result = basinhopping(self.new_cost_function, initial_guess, minimizer_kwargs={"args": (renewables_gridpoint,), "bounds": bounds}, stepsize=100)
                    
        # Store electrolyser capacity at that location
        optimal_electrolyser_capacity = result.x[0]
        optimal_lcoh = result.fun
        electrolyser_capacity = optimal_electrolyser_capacity 
                    
        # Store lowest achievable LCOH
        levelised_cost = optimal_lcoh
                    
        # Store hydrogen production
        optimal_levelised_cost, optimal_production = self.new_get_levelised_cost_optimisation(renewables_gridpoint, electrolyser_capacity)
        hydrogen_produced = optimal_production.mean()
            
        print(levelised_cost)
        print(electrolyser_capacity)
        print(hydrogen_produced)
        
        loop_end = time.time()
        loop_time = loop_end - loop_start
        return levelised_cost, electrolyser_capacity, hydrogen_produced, loop_time
    
    
    
    def global_optimisation_parallelised(self):
        start_time = time.time()
        latitudes = self.renewables_data.latitude.values
        longitudes = self.renewables_data.longitude.values
        
        # Create storage arrays
        levelised_cost_storage = xr.DataArray(data=np.empty_like(self.geodata['depth']), coords={'latitude': latitudes,'longitude': longitudes},dims=['latitude', 'longitude'])
        electrolyser_capacity_storage = xr.DataArray(data=np.empty_like(self.geodata['depth']), coords={'latitude': latitudes,'longitude': longitudes},dims=['latitude', 'longitude'])
        hydrogen_produced_storage = xr.DataArray(data=np.empty_like(self.geodata['depth']), coords={'latitude': latitudes,'longitude': longitudes},dims=['latitude', 'longitude'])
        time_processing = xr.DataArray(data=np.empty_like(self.geodata['depth']), coords={'latitude': latitudes,'longitude': longitudes},dims=['latitude', 'longitude'])
        
        # Create a list of arguments for each grid point
        grid_point_args = []
        for count_lat, lat in enumerate(latitudes):
            for count_lon, lon in enumerate(longitudes):
                grid_point_args.append((lat, lon))
    
        # Use joblib to parallelize the processing of grid points
        num_cores = 8  # Use all available CPU cores
        results = Parallel(n_jobs=num_cores, verbose=50, prefer="threads")(delayed(self.process_grid_point)(lat=lat, lon=lon) for lat, lon in grid_point_args)

        # Extract the results
        for i, result in enumerate(results):
            lat, lon = grid_point_args[i]
            levelised_cost, electrolyser_capacity, hydrogen_produced, loop_time = result
            levelised_cost_storage.loc[{'latitude': lat, 'longitude': lon}] = levelised_cost
            electrolyser_capacity_storage.loc[{'latitude': lat, 'longitude': lon}] = electrolyser_capacity
            hydrogen_produced_storage.loc[{'latitude': lat, 'longitude': lon}] = hydrogen_produced
            time_processing.loc[{'latitude': lat, 'longitude': lon}] = loop_time
        
        
        end_time = time.time()
        total_time = end_time - start_time
        loop_time = time_processing.mean()
        print(f"Optimisation took {loop_time:.2f} seconds to run for each grid point")
        print(f"Optimisation took {total_time:.2f} seconds to run for all grid points")
        return levelised_cost_storage, electrolyser_capacity_storage, hydrogen_produced_storage 
    
    
    
    
    def global_cost_parallelised(self):
        start_time = time.time()
        latitudes = self.renewables_data.latitude.values
        longitudes = self.renewables_data.longitude.values
        
        # Create storage arrays
        levelised_cost_array = xr.DataArray(data=np.empty_like(self.geodata['depth']), coords={'latitude': latitudes,'longitude': longitudes},dims=['latitude', 'longitude'])
        hydrogen_produced_array = xr.DataArray(data=np.empty_like(self.geodata['depth']), coords={'latitude': latitudes,'longitude': longitudes},dims=['latitude', 'longitude'])
        time_processing = xr.DataArray(data=np.empty_like(self.geodata['depth']), coords={'latitude': latitudes,'longitude': longitudes},dims=['latitude', 'longitude'])
        # Create a list of arguments for each grid point
        grid_point_args = []
        for count_lat, lat in enumerate(latitudes):
            for count_lon, lon in enumerate(longitudes):
                grid_point_args.append((lat, lon))
    
        # Use joblib to parallelize the processing of grid points
        num_cores = 8  # Use all available CPU cores
        results = Parallel(n_jobs=num_cores, verbose=50, prefer="threads")(delayed(self.levelised_cost_grid_point)(lat=lat, lon=lon) for lat, lon in grid_point_args)
        print(results)
        print(results[0])
        print(grid_point_args[0])
        print(levelised_cost_array)
        
        # Extract the results
        for i, result in enumerate(results):
            lat, lon = grid_point_args[i]
            print(lat)
            result_levelised_cost, result_hydrogen_produced, loop_time = result
            levelised_cost_array.loc[{'latitude': lat, 'longitude': lon}] = result_levelised_cost
            hydrogen_produced_array.loc[{'latitude': lat, 'longitude': lon}] = result_hydrogen_produced
            time_processing.loc[{'latitude': lat, 'longitude': lon}] = loop_time
        
        
        end_time = time.time()
        total_time = end_time - start_time
        loop_time = time_processing.mean()
        print(f"Running levelised cost calculation took {loop_time:.2f} seconds to run for each grid point")
        print(f"Running levelised cost calculation took {total_time:.2f} seconds to run for all grid points")
        return levelised_cost_array, hydrogen_produced_array

    
    
    def levelised_cost_grid_point(self, lat, lon):
        
        loop_start = time.time()
        
        # Get renewables data at each gridpoint
        renewables_gridpoint = self.renewables_data.sel(longitude = lon, latitude=lat)
        renewables_gridpoint = renewables_gridpoint.expand_dims(latitude=[lat], longitude=[lon])
        renewables_gridpoint = renewables_gridpoint.transpose("time", "latitude", "longitude")
                    
        # Store hydrogen production
        result_levelised_cost, result_hydrogen_production = self.new_get_levelised_cost_optimisation(renewables_gridpoint, self.electrolyser_capacity)
        
        
        loop_end = time.time()
        loop_time = loop_end - loop_start
        return result_levelised_cost, result_hydrogen_production, loop_time
    

    
    
    
    
    

### HYDROGEN MODEL PREAMBLE ###

## INPUTS ##


## CONSTRAINTS ##


## OUTPUTS ##

#### FOR LUKE
# Specify Paths to Input Data, Renewables Profiles and Location for the Output File
renewable_profiles_path = r"/Users/lukehatton/Sync/MERRA2_WIND_CF/"
input_data_path = r"/Users/lukehatton/Documents/Imperial/Code/Data/"
output_folder = r"/Users/lukehatton/Documents/Imperial/Code/Results/"

#### FOR IAIN
# Specify Paths to Input Data, Renewables Profiles and Location for the Output File
renewable_profiles_path = r"I:/NINJA_ERA5_GRIDDED_LUKE/MERRA2_WIND_CF/"
input_data_path = r"I:/NINJA_ERA5_GRIDDED_LUKE/"
output_folder = r"I:/NINJA_ERA5_GRIDDED_LUKE/OUTPUT_FOLDER/"
    
# Record start time
start_time = time.time()


# Read in the renewable profile dataset from files provided by Iain
#### UK
all_files_class = All_Files(lat_lon=[48, 62, -10, 2], filepath=renewable_profiles_path, name_format="WIND_CF.")

#### GLOBAL
all_files_class = All_Files(lat_lon=[-90, 90, -180, 180], filepath=renewable_profiles_path, name_format="WIND_CF.")
files_provided, years = all_files_class.preprocess_combine_yearly()
renewable_profile_array = files_provided['CF'] 
print("Files from Renewables Ninja read in, corrected and combined")

#renewable_profile = xr.open_dataset(renewable_profiles_path + "WIND_CF.2022.nc")
#lat_lon = [48, 62, -10, 2]
#years=[2022]
#renewable_profile = renewable_profile.sel(lat=slice(lat_lon[0], lat_lon[1]), lon=slice(lat_lon[2], lat_lon[3]))
#renewable_profile = renewable_profile.rename({'lat': 'latitude', 'lon': 'longitude'})
#renewable_profile_array = renewable_profile['CF'] 




# Initialise an HydrogenModel object
model = HydrogenModel(dataset=renewable_profile_array, lifetime = 20, years=years, params_file_elec=(input_data_path + "elec_parameters.csv"), params_file_renew=(input_data_path + "renew_parameters.csv"), data_path = input_data_path, output_folder=output_folder)


# Get timestamp
time_stamp = time.time()
date_time = datetime.fromtimestamp(time_stamp)
str_date_time = date_time.strftime("%d-%m-%Y-%H-%M-%S")
start_year = years[0]
end_year = years[-1]
years_str = str(start_year) + '_' + str(end_year)

# Calculate the levelised cost
levelised_costs, annual_production = model.get_levelised_cost()
try:
    levelised_costs.to_netcdf(output_folder + 'results_' + filename.rpartition('/')[-1])
except NameError:
    filename = 'results_' + years_str + '_' + str_date_time + '.nc'
    levelised_costs.to_netcdf(output_folder + filename)

# Record end time
end_time = time.time()

# Calculate elapsed time
elapsed_time = end_time - start_time

# Print elapsed time in seconds
print(f"Model took {elapsed_time:.2f} seconds to run")

# Calculate the levelised cost and create the supply cost curve
model.print_results_separately(levelised_costs)
supplycurvedata = model.get_supply_curves(levelised_costs, annual_production)
    

user_input = input("Complete Global Minimisation? (Y/N)")
if user_input == "Y":
    opt_levelised_costs, opt_electrolyser_capacity, opt_annual_production = model.global_optimisation_parallelised()
    print("SciPy BasinHopping finished running")
    model.print_results_separately(opt_levelised_cost)
    opt_supplycurvedata = model.get_supply_curves(opt_levelised_costs, opt_annual_production)
else:
    print("Model finished running")
    
    
    
    



