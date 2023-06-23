import numpy as np
import xarray as xr
import pandas as pd
import time
# Class to model the electrolyser and produce a hydrogen production profile from the renewable profile with details of CAPEX/OPEX



class Electrolyser:
    def __init__(self, electrolyser_capacity, efficiency, electrolyser_type, elec_capex, elec_op_cost, elec_discount_rate=None, elec_capacity_array=None, temperature=None, pressure=None):
        # Initialises the Electrolyser class
        self.electrolyser_capacity = electrolyser_capacity
        self.efficiency = efficiency
        self.electrolyser_type = electrolyser_type
        self.elec_capex = elec_capex
        self.elec_op_cost = elec_op_cost
        self.elec_discount_rate = elec_discount_rate
        self.temperature = temperature
        self.pressure = pressure
        if elec_capacity_array is not None:
            self.elec_capacity_array = elec_capacity_array

        # New attributes for the stack replacement
        self.stack_replacement_cost = 2000
        self.stack_lifespan = 10
        self.capital_costs = np.zeros(0)
        self.operating_costs = np.zeros(0)
        
        # Calculate the dynamic efficiency curve
        self.efficiency_curve = self.setup_efficiency_curve(temperature, pressure)

    def calculate_capital_cost(self):
        "Calculates the capital cost associated with the electrolyser "
        capex_adjustment = 1.0
        return self.elec_capex * self.electrolyser_capacity * capex_adjustment
    
    def update_capacity(self, new_capacity):
        self.electrolyser_capacity = new_capacity

    def calculate_capital_locational_cost(self, geodata, hydrogen_produced_yearly):
        "Calculates the capital cost associated with the electrolyser, doubling the capital cost for offshore locations"
        # Get offshore mask
        offshore_mask = geodata['offshore']
        sel_offshore_mask = offshore_mask.sel(latitude=hydrogen_produced_yearly.coords['latitude'],
                                                                  longitude=hydrogen_produced_yearly.coords['longitude'])
        
        # Adjust capital expenditure for electrolyser when offshore
        capex_adjustment = xr.where(sel_offshore_mask == True, 2, 1)
        
        # Calculate locational capital expenditure relating to the electrolyser
        capital_cost = capex_adjustment * self.elec_capex * self.electrolyser_capacity
        
        return capital_cost
        
    
    def calculate_operating_cost(self):
        "Calculates the operating cost associated with the electrolyser as a % of the CAPEX"
        return self.calculate_capital_cost() * self.elec_op_cost
    
    def locational_operating_costs(self, capital_costs, hydrogen_produced_yearly):
        """Calculates the operating cost associated with the renewable capacity as a % of the CAPEX at
        each location"""
        
        # Calculate operating cost as a proportion of CAPEX
        operating_cost = capital_costs * self.elec_op_cost
        
        # Assess size of hydrogen_produced_yearly and reproduce operating cost each year
        operating_cost_np = operating_cost.to_numpy()
        operating_costs_extended = operating_cost_np * np.ones_like(hydrogen_produced_yearly)
        return operating_costs_extended
        

    def calculate_max_yearly(self):
        max_H2_output = self.electrolyser_capacity * self.efficiency * 8760
        max_elec_input = self.electrolyser_capacity * 8760
        return max_elec_input, max_H2_output

   
    def setup_efficiency_curve(self, temperature, pressure):
    
        r1 = 4.45153e-5
        r2 = 6.88874e-9
        d1 = -3.12996e-6
        d2 = 4.47137e-7
        s = 0.33824
        t1 = -0.01539
        t2 = 2.00181
        t3 = 15.24178
        Urev = 1.23

        T = temperature
        p = pressure
        i = np.linspace(50, 6000, 200)
        A = 1
        Vcell = Urev + ((r1+d1) + r2*T + d2*p)*i + s*np.log10((t1+t2/T + t3/(T**2))*i + 1)
        I = A * i

        f11 = 478645.74
        f12 = -2953.15
        f21 = 1.03960
        f22 = -0.00104
        f1 = f11 + f12
        f2 = f21 + f22
        Nf = i**2 / (f1 + i**2) * f2
        F = 96500
        Nh = Nf * I / 2 / F
        LHV = 241000
        P_rate = I.max() * Vcell.max()
        P = I * Vcell
        P_perc = P/P_rate
        efficiency = Nh * LHV / P

        P_interp = np.arange(0.2, 1.01, 0.01)
        efficiency_interp = np.interp(P_interp, P_perc, efficiency)
        efficiency_interp = efficiency_interp.round(2)
        
        efficiency_df = pd.DataFrame({'P_Rated (%)': P_interp, 'Efficiency': efficiency_interp}, columns=['P_Rated (%)', 'Efficiency'])
        return efficiency_df
    
   
    def get_dynamic_efficiency(self, P_load):
    
        
        # Convert to numpy array
        power_values = P_load.values
        latitudes = P_load.latitude.values
        longitudes = P_load.longitude.values
        time_values = P_load.time.values
        
        # Create a storage array for the efficiency
        efficiency_array = np.zeros_like(power_values)
        
        
        # Get power curve
        efficiency_curve = self.efficiency_curve['Efficiency']
        
        # Round input power values
        power_load_rounded = power_values.round(decimals=2)
        
        # Use numpy where to apply 
        possible_values = np.arange(0.2, 1.01, 0.01)
        possible_values = possible_values.round(2)
        for index, value in enumerate(possible_values):
            efficiency_array = np.where(power_load_rounded == value, efficiency_curve[index], efficiency_array)
        
        dynamic_efficiency = xr.DataArray(efficiency_array, dims=('time', 'latitude', 'longitude'),
                                           coords={'time': time_values,
                                                   'latitude': latitudes,
                                                   'longitude': longitudes})
        
        
        return dynamic_efficiency


    def hydrogen_production(self, renewable_profile, capacity=None):
        "Calculates the hydrogen produced and renewable energy curtailed for the electrolyser"
        hydrogen_LHV = 33.3  # kWh/kg
        if capacity is not None:
            electrolyser_capacity = capacity
        else: 
            electrolyser_capacity = self.electrolyser_capacity
        
        # Calculate electricity available for hydrogen production, including any curtailed electricity and the shortfall
        electricity_H2 = xr.where(renewable_profile > electrolyser_capacity, electrolyser_capacity , renewable_profile)
        electricity_H2 = xr.where(electricity_H2 < 0.2 * electrolyser_capacity, 0, electricity_H2)
        curtailed_electricity = xr.where(renewable_profile > electrolyser_capacity, renewable_profile - electrolyser_capacity, 0)
        electrolyser_shortfall = xr.where(renewable_profile < electrolyser_capacity, electrolyser_capacity - renewable_profile, 0)
        
        # Calculate the partial load
        P_load = electricity_H2 / electrolyser_capacity 
        
        # Look up dynamic efficiency
        dynamic_eff = self.get_dynamic_efficiency(P_load)
        
        hydrogen_production = electricity_H2 * dynamic_eff / hydrogen_LHV  # In kg, assumes hourly resolution
        max_elec_input, max_H2_output = self.calculate_max_yearly()
        electrolyser_shortfall = electrolyser_shortfall / max_elec_input
        return hydrogen_production, curtailed_electricity, electrolyser_shortfall


    def calculate_yearly_output(self, renewable_profile, capacity=None):
        "Calculates the yearly hydrogen production at each location"
        hydrogen_produced, curtailed_electricity, electrolyser_shortfall = self.hydrogen_production(renewable_profile, capacity)
        latitudes = renewable_profile.latitude
        longitudes = renewable_profile.longitude
        
        
        hydrogen_produced_array = xr.DataArray(hydrogen_produced, dims=('time', 'latitude', 'longitude'),
                                           coords={'time': renewable_profile.time,
                                                   'latitude': latitudes,
                                                   'longitude': longitudes})
        curtailed_electricity_array = xr.DataArray(curtailed_electricity, dims=('time', 'latitude', 'longitude'),
                                           coords={'time': renewable_profile.time,
                                                   'latitude': latitudes,
                                                   'longitude': longitudes})
        electrolyser_shortfall_array = xr.DataArray(electrolyser_shortfall, dims=('time', 'latitude', 'longitude'),
                                           coords={'time': renewable_profile.time,
                                                   'latitude': latitudes,
                                                   'longitude': longitudes})
        hydrogen_produced_yearly = hydrogen_produced_array.groupby('time.year').sum(dim='time')
        curtailed_electricity_yearly = curtailed_electricity_array.groupby('time.year').sum(dim='time')
        electrolyser_shortfall_yearly = electrolyser_shortfall_array.groupby('time.year').sum(dim='time')
        
        # Accounting for CAPEX only in year 0
        years = hydrogen_produced_yearly.year
        lat_len = len(latitudes)
        lon_len = len(longitudes)
        new_year = [years[0]-1]
        years_appended = np.concatenate((new_year, years))
        zero_array = np.zeros((1, int(lat_len), int(lon_len)))
        
        zeroth_year_array = xr.DataArray(zero_array, dims=('year', 'latitude', 'longitude'),
                                           coords={'year': new_year,
                                                   'latitude': latitudes,
                                                   'longitude': longitudes})
        

        hydrogen_produced_yearly = xr.concat([zeroth_year_array * 0, hydrogen_produced_yearly], dim = 'year')
        curtailed_electricity_yearly = xr.concat([zeroth_year_array * 0, curtailed_electricity_yearly], dim = 'year')
        electrolyser_shortfall_yearly = xr.concat([zeroth_year_array * 0, electrolyser_shortfall_yearly], dim = 'year')
        
        # Create a dataset with all the arrays
        data_vars = {'hydrogen_produced': hydrogen_produced_yearly,
                     'curtailed_electricity': curtailed_electricity_yearly, 'electrolyser_shortfall': electrolyser_shortfall_yearly}
        coords = {'year': years_appended,
                  'latitude': latitudes,
                  'longitude': longitudes}
        electrolyser_output = xr.Dataset(data_vars=data_vars, coords=coords)
        
        return electrolyser_output
    
    
    
    def calculate_yearly_locational_output(self, renewable_profile, geodata):
        "Calculates the yearly cost and hydrogen produced by location"
        " REDUNDANT - TO BE REMOVED"
        hydrogen_produced, curtailed_electricity, electrolyser_shortfall = self.hydrogen_production(renewable_profile)
        latitudes = renewable_profile.latitude
        longitudes = renewable_profile.longitude
        
        
        hydrogen_produced_yearly = hydrogen_produced.groupby('time.year').sum(dim='time')
        curtailed_electricity_yearly = curtailed_electricity.groupby('time.year').sum(dim='time')
        electrolyser_shortfall_yearly = electrolyser_shortfall.groupby('time.year').sum(dim='time')
        
        # Accounting for CAPEX only in year 0
        years = hydrogen_produced_yearly.year
        lat_len = len(latitudes)
        lon_len = len(longitudes)
        new_year = [years[0]-1]
        years_appended = np.concatenate((new_year, years))
        zero_array = np.zeros((1, int(lat_len), int(lon_len)))
        
        capital_costs_array = xr.DataArray(zero_array, dims=('year', 'latitude', 'longitude'),
                                           coords={'year': new_year,
                                                   'latitude': latitudes,
                                                   'longitude': longitudes})
        locational_capital_costs = self.calculate_capital_locational_cost(geodata, hydrogen_produced_yearly)
        capital_costs_array[0, :, :] = locational_capital_costs
        operating_costs_array = xr.DataArray(self.locational_operating_costs(locational_capital_costs, hydrogen_produced_yearly), dims=('year', 'latitude', 'longitude'), coords={'year': years,'latitude': latitudes,'longitude': longitudes})
        total_costs_array = xr.concat([capital_costs_array, operating_costs_array], dim = 'year')
        hydrogen_produced_yearly = xr.concat([capital_costs_array * 0, hydrogen_produced_yearly], dim = 'year')
        curtailed_electricity_yearly = xr.concat([capital_costs_array * 0, curtailed_electricity_yearly], dim = 'year')
        electrolyser_shortfall_yearly = xr.concat([capital_costs_array * 0, electrolyser_shortfall_yearly], dim = 'year')
        print("Electrolyser Costs previous method: ")
        print(total_costs_array[0, 0, 0])
        
        # Create a dataset with all the arrays
        data_vars = {'hydrogen_produced': hydrogen_produced_yearly,
                     'costs': total_costs_array,
                     'curtailed_electricity': curtailed_electricity_yearly, 'electrolyser_shortfall': electrolyser_shortfall_yearly}
        coords = {'year': years_appended,
                  'latitude': latitudes,
                  'longitude': longitudes}
        ds = xr.Dataset(data_vars=data_vars, coords=coords)
        
        return ds
    
    
    
    
