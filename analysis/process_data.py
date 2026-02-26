'''
Script to process public NHTSA crash data from the following databases:
1. Fatality Analysis Reporting System (FARS)
2. General Estimates System (GES)
3. Crash Report Sampling System (CRSS) 

The outputs of this script are csv files of filtered data containing 
a subset of columns. These csv files serve as inputs to the cluster_data.py
script to extract representative scenarios.
'''

import os
import pandas as pd
import numpy as np

def split_by_size(data, casenum, ACC_TYPE_name, ACC_TYPE_miscellaneous_list):
    '''
    Splits data into two dataframes, one with 1 vehicle accidents and one with
    2+ vehicle accidents. For 2+ vehicle accidents, only two rows are saved,
    prioritizing rows with ACC_TYPEs not in the miscellaneous list.  
    Returns the two dataframes.
    '''

    df = data.copy()
    df['_i'] = np.arange(len(df)) # stable tie-breaker

    grp_size = df.groupby(casenum)[casenum].transform('size')

    # 1 vehicle cases
    one_veh_df = df[grp_size == 1].drop(columns=['_i']).reset_index(drop=True)

    # 2+ vehicles: prefer non-misc ACC_TYPE, then keep first two rows per case
    mask_two_plus = grp_size >= 2
    tmp = df.loc[mask_two_plus].copy()
    tmp['_pref'] = (~tmp[ACC_TYPE_name].isin(ACC_TYPE_miscellaneous_list)).astype('int8')

    # sort by case, then preference (non-misc first), then original order
    tmp = tmp.sort_values([casenum, '_pref', '_i'],
                           ascending=[True, False, True],
                           kind='mergesort')
    tmp = tmp[tmp.groupby(casenum).cumcount() < 2].drop(columns=['_pref', '_i'])

    two_veh_df = tmp.reset_index(drop=True)

    return one_veh_df, two_veh_df

def filter_by_vehicle_type(data, casenum, body_typ_list, body_typ_name, gvwr_list, gvwr_name):
    '''
    Filters data by vehicle type, only keeping case number if at least 
    one of the vehicles matches the desired body and gvwr values. Returns the 
    filtered data.
    '''
    cond = (data[body_typ_name].isin(body_typ_list) & data[gvwr_name].isin(gvwr_list))
    keep_case = cond.groupby(data[casenum]).transform('any')
    return data[keep_case].reset_index(drop=True)

def assign_vehicle_nums(data, casenum):
    '''
    Assigns each vehicle in 2-vehicle groups to 1 or 2, where 1 is
    -- generally speaking -- the vehicle causing the accident / colliding 
    into vehicle 2. Returns data with the VEH_NUM column added.
    '''

    assigned_df = data.copy()
    assigned_df['_i'] = np.arange(len(assigned_df))

    # keep only two-row cases and make sure rows are paired in order
    grp_size = assigned_df.groupby(casenum)[casenum].transform('size')
    assigned_df = assigned_df.loc[grp_size == 2].sort_values([casenum, '_i'], kind='mergesort')
    assigned_df = assigned_df.drop(columns=['_i'])

    p = assigned_df['P_CRASH2'].to_numpy().reshape(-1, 2)
    a = assigned_df['ACC_TYPE'].to_numpy().reshape(-1, 2)

    n = p.shape[0]
    decided = np.zeros(n, dtype=bool) # track which pairs have been assigned
    swap = np.zeros(n, dtype=bool)    # True => (2,1), False => (1,2)

    # reusable membership masks
    paired_acc = np.array([20, 24, 28, 34, 36, 38, 40, 50, 54, 56, 58, 60, 
                           64, 68, 70, 72, 76, 78, 80, 82, 86, 88, 92])
    unknown_acc = np.array([92, 93, 97, 98, 99, 0])
    rear_slow_pcrash2 = np.array([50, 51, 52])
    encroach_pcrash2 = np.array([60, 61, 62, 63, 64, 65, 66, 67, 68, 70, 
                                 71, 72, 73, 74, 78])
    lost_ctrl_pcrash2 = np.array([1, 2, 3, 4, 5, 6, 8, 9])
    in_lane_pcrash2 = np.array([50, 51, 52, 53, 54, 55, 56, 59])
    unknown_pcrash2 = np.array([98, 99])

    a1_in_paired = np.isin(a[:,0], paired_acc)
    a2_in_paired = np.isin(a[:,1], paired_acc)
    a1_in_unknown = np.isin(a[:,0], unknown_acc)
    a2_in_unknown = np.isin(a[:,1], unknown_acc)
    p1_in_rear_slow = np.isin(p[:,0], rear_slow_pcrash2)
    p2_in_rear_slow = np.isin(p[:,1], rear_slow_pcrash2)
    p1_in_enc = np.isin(p[:,0], encroach_pcrash2)
    p2_in_enc = np.isin(p[:,1], encroach_pcrash2)
    p1_lost = np.isin(p[:,0], lost_ctrl_pcrash2)
    p2_lost = np.isin(p[:,1], lost_ctrl_pcrash2)
    p1_inlane = np.isin(p[:,0], in_lane_pcrash2)
    p2_inlane = np.isin(p[:,1], in_lane_pcrash2)
    p1_unk = np.isin(p[:,0], unknown_pcrash2)
    p2_unk = np.isin(p[:,1], unknown_pcrash2)

    def decide(mask, make_swap):
        m = mask & (~decided)
        if (make_swap):
            swap[m] = True
        decided[m] = True

    # apply rules for determining vehicle number

    # using paired ACC_TYPEs
    decide((a[:,0] < a[:,1]) & a1_in_paired, make_swap=False)
    decide((a[:,1] < a[:,0]) & a2_in_paired, make_swap=True)

    # this vehicle in P_CRASH2
    decide((p[:,0] <= 21) & (p[:,1] > 21), make_swap=False)
    decide((p[:,1] <= 21) & (p[:,0] > 21), make_swap=True)

    # rear-end faster/slower dynamics
    decide(p[:,0] == 53, make_swap=True)
    decide(p[:,1] == 53, make_swap=False)
    decide(p1_in_rear_slow, make_swap=False)
    decide(p2_in_rear_slow, make_swap=True)

    # encroached into lane
    decide(p1_in_enc & (~p2_in_enc), make_swap=True)
    decide(p2_in_enc & (~p1_in_enc), make_swap=False)

    # lost control
    decide(p1_lost & (~p2_lost), make_swap=False)
    decide(p2_lost & (~p1_lost), make_swap=True)

    # other vehicle was in lane
    decide(p1_inlane & (~p2_inlane), make_swap=True)
    decide(p2_inlane & (~p1_inlane), make_swap=False)

    # unknown P_CRASH2
    decide((~p1_unk) & p2_unk, make_swap=False)
    decide((~p2_unk) & p1_unk, make_swap=True)

    # unknown ACC_TYPE
    decide((~a1_in_unknown) & a2_in_unknown, make_swap=False)
    decide((~a2_in_unknown) & a1_in_unknown, make_swap=True)

    # P_CRASH2 and ACC_TYPE same
    decide((p[:,0] == p[:,1]) & (a[:,0] == a[:,1]), make_swap=False)

    veh = np.tile(np.array([1, 2], dtype=int), (n, 1))
    veh[swap] = veh[swap][:, ::-1]
    assigned_df['VEH_NUM'] = veh.reshape(-1)

    return assigned_df

def process_data(script_dir, year, database, columns):
    '''
    Loads, merges, and filters data for a given year and database type.
    Returns data_one and data_two: pandas dataframes for 1 vehicle and 2+ vehicle
    accidents with rows for each accident containing data in columns of interest.
    '''

    # read accident.csv and vehicle.csv
    acc_path = os.path.join(script_dir, f'../data_in/{database}/{year}/accident.csv')
    veh_path = os.path.join(script_dir, f'../data_in/{database}/{year}/vehicle.csv')
    if (year <= 2014 and database == 'GES'): # GES 2011-2014 files use tab separation
        acc = pd.read_csv(acc_path, encoding='latin1', low_memory=False, sep='\t')
        veh = pd.read_csv(veh_path, encoding='latin1', low_memory=False, sep='\t')
    else:
        acc = pd.read_csv(acc_path, encoding='latin1', low_memory=False)
        veh = pd.read_csv(veh_path, encoding='latin1', low_memory=False)

    # set casenum variable and add it to saved columns
    if (database == 'FARS'):
        casenum = 'ST_CASE'
    else:
        casenum = 'CASENUM'
    columns.insert(0, casenum)

    # merge accident and vehicle data
    merged = pd.merge(acc, veh, on=casenum, how='left', suffixes=('_acc', '_veh'))  

    # append year to casenum and add casenum to saved columns
    if (database == 'FARS'):
        merged[casenum] = merged[casenum].apply(lambda x: f'{year}_{x}')
    else: # year already in casenum for GES/CRSS, just formatting it to match GES
        merged[casenum] = merged[casenum].astype(str).apply(lambda x: f'{x[:4]}_{x[4:]}')
    
    #------------------------------------------------------------------#
    #------------------------ START USER INPUT ------------------------#
    #------------------------------------------------------------------#
    
    # filter conditions for weather
    WEATHER_list = [3,  # Sleet or Hail
                    4,  # Snow
                    11, # Blowing Snow
                    12] # Freezing Rain or Drizzle
    if (database == 'FARS'):
        WEATHER_name = 'WEATHER'
    else:
        WEATHER_name = 'WEATHR_IM' # imputed weather for GES/CRSS

    # filter conditions for road surface
    VSURCOND_list = [3,  # Snow
                     4,  # Ice/Frost
                     10] # Slush
    VSURCOND_name = 'VSURCOND'

    # filter conditions for vehicle body type
    BODY_TYP_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 17, 18, 19, 20,
                     21, 22, 24, 25, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                     40, 41, 45, 46, 47, 48, 49, 79] # Passenger Vehicles (includes vans, light trucks)
                                                     # as defined in Appendix C of FARS User Manual.
                                                     # GES/CRSS use the same categorization 
    if (database == 'FARS' or year >= 2021):
        BODY_TYP_name = 'BODY_TYP'
    else:
        BODY_TYP_name = 'BDYTYP_IM' # imputed body type for GES/CRSS 2011-2020

    # filter conditions for gross vehicle weight class  
    if (year >= 2020): # starting in 2020, FARS/CRSS use GVWR_TO variable
        GVWR_list = [11, # Class 1 (< 6,000 lbs)
                     12] # Class 2 (6,000-10,000 lbs)
        GVWR_name = 'GVWR_TO'
    else:
        GVWR_list = [0, # Not Applicable (most light vehicles have this designation)
                     1] # 10,000 lbs or Less
        GVWR_name = 'GVWR'

    # filter for winter conditions
    merged_filtered_veh = merged[(merged[WEATHER_name].isin(WEATHER_list)) | 
                                 (merged[VSURCOND_name].isin(VSURCOND_list))]
    merged_filtered_veh_unique = merged_filtered_veh.drop_duplicates(subset=casenum, keep='first')
    merged_filtered = merged[merged[casenum].isin(merged_filtered_veh_unique[casenum])]

    # users can modify the winter conditions filter above and/or add additional filters here

    # here is an example of adding an additional filter for nighttime conditions:

    # LGT_COND_list = [2, # Dark – Not Lighted
    #                  3, # Dark – Lighted
    #                  6] # Dark – Unknown Lighting
    # LGT_COND_name = 'LGT_COND'

    # merged_filtered_veh = merged_filtered[(merged_filtered[LGT_COND_name].isin(LGT_COND_list))]
    # merged_filtered_veh_unique = merged_filtered_veh.drop_duplicates(subset=casenum, keep='first')
    # merged_filtered = merged_filtered[merged_filtered[casenum].isin(merged_filtered_veh_unique[casenum])]

    #------------------------------------------------------------------#
    #------------------------- END USER INPUT -------------------------#
    #------------------------------------------------------------------#

    # accident type name and list of miscellaneous types, 
    # this is used for determining most crash-relevant vehicles in multi-vehicle accidents
    ACC_TYPE_name = 'ACC_TYPE'
    ACC_TYPE_miscellaneous_list = [0,  # No Impact
                                   92, # Backing Vehicle 
                                   93, # Other Vehicle or Object
                                   97, # Untripped Rollover
                                   98, # Other Crash Type
                                   99] # Unknown Crash Type
    
    # split data into two dataframes: one for 1 vehicle accidents and one for 2+ vehicle accidents
    (merged_filtered_one, 
     merged_filtered_two) = split_by_size(merged_filtered, 
                                          casenum, 
                                          ACC_TYPE_name,
                                          ACC_TYPE_miscellaneous_list)
    
    # filter for accidents with at least one of first two vehicles in desired vehicle type
    merged_filtered_vehtype_one = filter_by_vehicle_type(merged_filtered_one, 
                                                         casenum, 
                                                         BODY_TYP_list, 
                                                         BODY_TYP_name, 
                                                         GVWR_list, 
                                                         GVWR_name)
    merged_filtered_vehtype_two = filter_by_vehicle_type(merged_filtered_two, 
                                                         casenum, 
                                                         BODY_TYP_list, 
                                                         BODY_TYP_name, 
                                                         GVWR_list, 
                                                         GVWR_name)

    # remove _acc from column names to make sure they match saved column names
    merged_filtered_vehtype_one.rename(columns={col: col.replace('_acc', '') 
                                                for col in merged_filtered_vehtype_one.columns 
                                                if col.endswith('_acc')}, inplace=True)
    merged_filtered_vehtype_two.rename(columns={col: col.replace('_acc', '') 
                                                for col in merged_filtered_vehtype_two.columns 
                                                if col.endswith('_acc')}, inplace=True)

    # create new dataframes containing just the saved columns
    data_one = merged_filtered_vehtype_one[columns].copy()
    data_two_unassigned = merged_filtered_vehtype_two[columns].copy()

    # assign vehicles to 1 or 2 for two vehicle data
    data_two = assign_vehicle_nums(data_two_unassigned, casenum)

    return data_one, data_two

if __name__ == '__main__':

    # directory of this script for data upload/saving
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # range of years for analysis, default is all available years (2011-2022)
    year_start = 2011
    year_end   = 2022
    years = range(year_start, year_end+1)
    
    # FARS column names to save for clustering script
    columns_FARS = ['VE_FORMS', # Number of Motor Vehicles In-Transport (MVIT)
                    'HARM_EV',  # First Harmful Event
                    'RELJCT2',  # Relation to Junction—Specific Location
                    'TYP_INT',  # Type of Intersection
                    'REL_ROAD', # Relation to Trafficway
                    'LGT_COND', # Light Condition
                    'WEATHER',  # Atmospheric Conditions
                    'VTRAFWAY', # Trafficway Description
                    'VNUM_LAN', # Total Lanes in Roadway
                    'VSPD_LIM', # Speed Limit
                    'VALIGN',   # Roadway Alignment
                    'VPROFILE', # Roadway Grade
                    'VSURCOND', # Roadway Surface Condition
                    'VTRAFCON', # Traffic Control Device
                    'P_CRASH1', # Pre-Event Movement (Prior to Recognition of Critical Event)
                    'P_CRASH2', # Critical Event-Precrash
                    'P_CRASH3', # Attempted Avoidance Maneuver
                    'PCRASH4',  # Pre-Impact Stability
                    'PCRASH5',  # Pre-Impact Location
                    'ACC_TYPE'] # Crash Type
    
    # GES/CRSS column names to save for clustering script
    columns_GESCRSS = ['VE_FORMS',   # Number of Motor Vehicles In-Transport (MVIT)
                       'EVENT1_IM',  # Imputed First Harmful Event
                       'RELJCT2_IM', # Imputed Relation to Junction—Specific Location
                       'TYP_INT',    # Type of Intersection
                       'REL_ROAD',   # Relation to Trafficway
                       'LGTCON_IM',  # Imputed Light Condition
                       'WEATHR_IM',  # Imputed Atmospheric Conditions
                       'VTRAFWAY',   # Trafficway Description
                       'VNUM_LAN',   # Total Lanes in Roadway
                       'VSPD_LIM',   # Speed Limit
                       'VALIGN',     # Roadway Alignment
                       'VPROFILE',   # Roadway Grade
                       'VSURCOND',   # Roadway Surface Condition
                       'VTRAFCON',   # Traffic Control Device
                       'PCRASH1_IM', # Imputed Pre-Event Movement (Prior to Recognition of Critical Event)
                       'P_CRASH2',   # Critical Event-Precrash
                       'P_CRASH3',   # Attempted Avoidance Maneuver
                       'PCRASH4',    # Pre-Impact Stability
                       'PCRASH5',    # Pre-Impact Location
                       'ACC_TYPE']   # Crash Type

    # empty dataframes needed for looping through data
    data_one_all_FARS = pd.DataFrame()
    data_two_all_FARS = pd.DataFrame()
    data_one_all_GESCRSS = pd.DataFrame()
    data_two_all_GESCRSS = pd.DataFrame()

    FARS_yearly_counts = {}
    GESCRSS_yearly_counts = {}

    # loop through each year and process+gather the data
    for year in years:
        
        print(f'Reading, merging, and filtering data for year: {year}')
        
        # FARS
        data_one_FARS, data_two_FARS = process_data(script_dir, 
                                                    year, 
                                                    'FARS', 
                                                    columns_FARS.copy())
        data_one_all_FARS = pd.concat([data_one_all_FARS, data_one_FARS], ignore_index=True)
        data_two_all_FARS = pd.concat([data_two_all_FARS, data_two_FARS], ignore_index=True)

        FARS_year_data = pd.concat([data_one_FARS, data_two_FARS], ignore_index=True)
        FARS_yearly_counts[year] = FARS_year_data['ACC_TYPE'].value_counts()

        # GES/CRSS
        if (year <= 2015):
            data_one_GESCRSS, data_two_GESCRSS = process_data(script_dir, 
                                                              year, 
                                                              'GES', 
                                                              columns_GESCRSS.copy())
        else:
            data_one_GESCRSS, data_two_GESCRSS = process_data(script_dir, 
                                                              year, 
                                                              'CRSS', 
                                                              columns_GESCRSS.copy())
        data_one_all_GESCRSS = pd.concat([data_one_all_GESCRSS, data_one_GESCRSS], ignore_index=True)
        data_two_all_GESCRSS = pd.concat([data_two_all_GESCRSS, data_two_GESCRSS], ignore_index=True)

        GESCRSS_year_data = pd.concat([data_one_GESCRSS, data_two_GESCRSS], ignore_index=True)
        GESCRSS_yearly_counts[year] = GESCRSS_year_data['ACC_TYPE'].value_counts()

    # convert two vehicle data to dataframes with 1 row and double the columns for each accident
    data_two_all_FARS_1 = data_two_all_FARS[data_two_all_FARS['VEH_NUM'] == 1].add_suffix('_1').rename(columns={'ST_CASE_1': 'ST_CASE'})
    data_two_all_FARS_2 = data_two_all_FARS[data_two_all_FARS['VEH_NUM'] == 2].add_suffix('_2').rename(columns={'ST_CASE_2': 'ST_CASE'})
    data_two_all_FARS_final = pd.merge(data_two_all_FARS_1, 
                                       data_two_all_FARS_2, 
                                       on='ST_CASE')
    
    data_two_all_GESCRSS_1 = data_two_all_GESCRSS[data_two_all_GESCRSS['VEH_NUM'] == 1].add_suffix('_1').rename(columns={'CASENUM_1': 'CASENUM'})
    data_two_all_GESCRSS_2 = data_two_all_GESCRSS[data_two_all_GESCRSS['VEH_NUM'] == 2].add_suffix('_2').rename(columns={'CASENUM_2': 'CASENUM'})
    data_two_all_GESCRSS_final = pd.merge(data_two_all_GESCRSS_1, 
                                          data_two_all_GESCRSS_2, 
                                          on='CASENUM')

    # set/create output path
    output_path = os.path.join(script_dir, '../data_out')
    os.makedirs(output_path, exist_ok=True)
    
    # lastly, save data to csv files
    data_one_all_FARS_name = f'FARS_{year_start}_{year_end}_one_filtered.csv'
    data_one_all_FARS.to_csv(os.path.join(output_path, data_one_all_FARS_name), index=False)
    print(f'Saved {data_one_all_FARS_name} with dimensions:', data_one_all_FARS.shape)
    
    data_two_all_FARS_name = f'FARS_{year_start}_{year_end}_two_filtered.csv'
    data_two_all_FARS_final.to_csv(os.path.join(output_path, data_two_all_FARS_name), index=False)
    print(f'Saved {data_two_all_FARS_name} with dimensions:', data_two_all_FARS_final.shape)

    data_one_all_GESCRSS_name = f'GESCRSS_{year_start}_{year_end}_one_filtered.csv'
    data_one_all_GESCRSS.to_csv(os.path.join(output_path, data_one_all_GESCRSS_name), index=False)
    print(f'Saved {data_one_all_GESCRSS_name} with dimensions:', data_one_all_GESCRSS.shape)

    data_two_all_GESCRSS_name = f'GESCRSS_{year_start}_{year_end}_two_filtered.csv'
    data_two_all_GESCRSS_final.to_csv(os.path.join(output_path, data_two_all_GESCRSS_name), index=False)
    print(f'Saved {data_two_all_GESCRSS_name} with dimensions:', data_two_all_GESCRSS_final.shape)