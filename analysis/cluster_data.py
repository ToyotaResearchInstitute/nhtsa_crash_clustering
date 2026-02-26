'''
Script to cluster NHTSA crashes with one or multiple vehicles.

This script uses as input the 'one' or 'two' filtered csv files created 
in the process_data.py script. After calling this script, the user selects 
which of the processed csvs to perform clustering analysis on via a prompt 
in the terminal. All text output in the terminal, including the final 
scenario clusters, are saved in a txt file.
'''

import pandas as pd
import numpy as np
import os
import sys
import io
import atexit
from sklearn.metrics import silhouette_score
from sklearn.cluster import SpectralClustering, AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import OrdinalEncoder
from collections import namedtuple, Counter
from numba import njit
from pathlib import Path
from types import SimpleNamespace

def setup_terminal_logger(selected_csv_path):
    '''
    Echoes all printed text in the terminal and writes a final log that is saved in the data_out folder.
    '''

    p = Path(selected_csv_path)
    stdout0, stderr0 = sys.stdout, sys.stderr
    buf = io.StringIO()

    sys.stdout = SimpleNamespace(
        write=lambda s, out=stdout0: (out.write(s), buf.write(s)) and len(s),
        flush=lambda out=stdout0: (out.flush(), buf.flush()),
        encoding=getattr(stdout0, 'encoding', 'utf-8'),
        isatty=lambda out=stdout0: getattr(out, 'isatty', lambda: False)(),
    )
    sys.stderr = SimpleNamespace(
        write=lambda s, out=stderr0: (out.write(s), buf.write(s)) and len(s),
        flush=lambda out=stderr0: (out.flush(), buf.flush()),
        encoding=getattr(stderr0, 'encoding', 'utf-8'),
        isatty=lambda out=stderr0: getattr(out, 'isatty', lambda: False)(),
    )

    # name txt file, eg FARS_2011_2022_one_filtered.csv -> FARS_2011_2022_one_cluster_results.txt
    name = p.name
    if (name.endswith('_filtered.csv')):
        logname = name[:-len('_filtered.csv')] + '_cluster_results.txt'
    else:
        logname = p.stem + '_cluster_results.txt'
    log_path = p.with_name(logname)

    # write once at script exit
    atexit.register(
        lambda lp=log_path, b=buf, out=stdout0: (
            open(lp, 'w', encoding='utf-8').write(b.getvalue()),
            out.flush()
        )
    )

def categorize_data(input_df, is_one, LGT_COND_name, HARM_EV_name):
    '''
    Groups a few key variables into categories for clustering. This enables the clustering algorithms
    to form more meaningfully distinct groups.
    '''
    
    Category = namedtuple('Category', ['name', 'values'])
    
    if (is_one):

        categories = {
            'ACC_TYPE': {
                '01': Category('01: Single Driver, Roadside Depart -- Drive Off Road', [1, 6]),
                '02': Category('02: Single Driver, Roadside Depart -- Ctl/Traction Loss', [2, 7]),
                '03': Category('03: Single Driver, Roadside Depart -- Avoid Collision', [3, 8]),
                '04': Category('04: Single Driver, Roadside Depart -- Other/Unknown', [4, 5, 9, 10]),
                '05': Category('05: Single Driver, Forward Impact -- Parked Vehicle', [11]),
                '06': Category('06: Single Driver, Forward Impact -- Static Object', [12]),
                '07': Category('07: Single Driver, Forward Impact -- Ped/Animal', [13]),
                '08': Category('08: Single Driver, Forward Impact -- End Depart', [14]),
                '09': Category('09: Single Driver, Forward Impact -- Other/Unknown', [15, 16]),
                '10': Category('10: Same-Same, Rear End -- Stopped', [20, 21, 22, 23]),
                '11': Category('11: Same-Same, Rear End -- Slower', [24, 25, 26, 27]),
                '12': Category('12: Same-Same, Rear End -- Decel', [28, 29, 30, 31]),
                '13': Category('13: Same-Same, Rear End -- Other/Unknown', [32, 33]),
                '14': Category('14: Same-Same, Forward Impact -- Ctl/Traction Loss Avoid Veh', [34, 35]),
                '15': Category('15: Same-Same, Forward Impact -- Ctl/Traction Loss Avoid Obj', [36, 37]),
                '16': Category('16: Same-Same, Forward Impact -- Avoid Veh', [38, 39]),
                '17': Category('17: Same-Same, Forward Impact -- Avoid Obj', [40, 41]),
                '18': Category('18: Same-Same, Forward Impact -- Other/Unknown', [42, 43]),
                '19': Category('19: Same-Same, Angle/Sideswipe -- Driving Straight', [44, 45]),
                '20': Category('20: Same-Same, Angle/Sideswipe -- Changing Lanes', [46, 47]),
                '21': Category('21: Same-Same, Angle/Sideswipe -- Other/Unknown', [48, 49]),
                '22': Category('22: Same-Oppo, Head-On -- Lateral Moves', [50, 51]),
                '23': Category('23: Same-Oppo, Head-On -- Other/Unknown', [52, 53]),
                '24': Category('24: Same-Oppo, Forward Impact -- Ctl/Traction Loss Avoid Veh', [54, 55]),
                '25': Category('25: Same-Oppo, Forward Impact -- Ctl/Traction Loss Avoid Obj', [56, 57]),
                '26': Category('26: Same-Oppo, Forward Impact -- Avoid Veh', [58, 59]),
                '27': Category('27: Same-Oppo, Forward Impact -- Avoid Obj', [60, 61]),
                '28': Category('28: Same-Oppo, Forward Impact -- Other/Unknown', [62, 63]),
                '29': Category('29: Same-Oppo, Angle/Sideswipe -- Lateral Moves', [64, 65]),
                '30': Category('30: Same-Oppo, Angle/Sideswipe -- Other/Unknown', [66, 67]),
                '31': Category('31: Veh Turning, Turn Across -- Init Oppo Directions', [68, 69]),
                '32': Category('32: Veh Turning, Turn Across -- Init Same Directions', [70, 71, 72, 73]),
                '33': Category('33: Veh Turning, Turn Across -- Other/Unknown', [74, 75]),
                '34': Category('34: Veh Turning, Turn Into -- Into Same Direction', [76, 77, 78, 79]),
                '35': Category('35: Veh Turning, Turn Into -- Into Oppo Direction', [80, 81, 82, 83]),
                '36': Category('36: Veh Turning, Turn Into -- Other/Unknown', [84, 85]),
                '37': Category('37: Intersect, Straight Paths -- T-Bone Known Direction', [86, 87, 88, 89]),
                '38': Category('38: Intersect, Straight Paths -- Other/Unknown', [90, 91]),
                '39': Category('39: Miscellaneous', [92, 93, 97, 98, 99, 0]),
                'Unclassified': Category('Unclassified', [])
            },
            HARM_EV_name: {
                '01': Category('01: Rollover', [1]),
                '02': Category('02: Other Non-Crash', [2, 3, 4, 5, 6, 7, 51, 72]),
                '03': Category('03: Hit Person/Animal', [8, 9, 11, 15, 47, 49]),
                '04': Category('04: Hit Motor Vehicle in Transport', [12, 13, 55]),
                '05': Category('05: Hit Other Vehicle', [10, 14, 45, 74]),
                '06': Category('06: Hit With Object', [16, 17, 18, 54, 73, 91]),
                '07': Category('07: Hit Building/Barrier', [19, 20, 21, 22, 23, 24, 25, 26, 38, 39, 52, 57]),
                '08': Category('08: Hit Sign/Post', [27, 28, 29, 30, 31, 46, 59]),
                '09': Category('09: Crashed into Road Structure', [32, 33, 34, 35, 36, 37, 50, 58]),
                '10': Category('10: Hit Natural Object', [41, 42, 48]),
                '11': Category('11: Hit Other Object', [40, 43, 53, 93]),
                '12': Category('12: Pavement Surface Irregularity', [44]),
                '13': Category('13: Miscellaneous', [98, 99]),
                'Unclassified': Category('Unclassified', [])
            },
            'P_CRASH2': {
                '01': Category('01: Lost Ctl Due To -- Poor Road Conditions', [5]),
                '02': Category('02: Lost Ctl Due To -- Traveling Too Fast for Conditions', [6]),
                '03': Category('03: Lost Ctl Due To -- Other', [1, 2, 3, 4, 8, 9]),
                '04': Category('04: This Veh Traveling -- Over Lane Line', [10, 11]),
                '05': Category('05: This Veh Traveling -- Off Road Edge', [12, 13]),
                '06': Category('06: This Veh Traveling -- Turning', [15, 16, 21]),
                '07': Category('07: This Veh Traveling -- Driving Through Road/Junction', [14, 17]),
                '08': Category('08: This Veh Traveling -- Decelerating', [18]),
                '09': Category('09: This Veh Traveling -- Other', [19, 20]),
                '10': Category('10: Other Veh in Lane -- Stopped/Slower', [50, 51, 52]),
                '11': Category('11: Other Veh in Lane -- Faster', [53]),
                '12': Category('12: Other Veh in Lane -- Oppo Direction', [54]),
                '13': Category('13: Other Veh in Lane -- Other', [55, 56, 59]),
                '14': Category('14: Other Veh into Lane -- From Adj Lane', [60, 61]),
                '15': Category('15: Other Veh into Lane -- From Oppo Direction', [62, 63]),
                '16': Category('16: Other Veh into Lane -- From Parking/Driveway', [64, 70, 71, 72, 73]),
                '17': Category('17: Other Veh into Lane -- From Crossing Street', [65, 66, 67, 68]),
                '18': Category('18: Other Veh into Lane -- From Highway Entrance', [74]),
                '19': Category('19: Other Veh into Lane -- Other', [78]),
                '20': Category('20: Person -- Pedestrian', [80, 81, 82]),
                '21': Category('21: Person -- Pedalcyclist/Non-Motorist', [83, 84, 85]),
                '22': Category('22: Animal/Object -- Animal', [87, 88, 89]),
                '23': Category('23: Animal/Object -- Object', [90, 91, 92]),
                '24': Category('24: Miscellaneous', [98, 99]),
                'Unclassified': Category('Unclassified', [])
            },
            LGT_COND_name: {
                '01': Category('01: Daylight', [1]),
                '02': Category('02: Dark', [2, 3, 6]),
                '03': Category('03: Dawn', [4]),
                '04': Category('04: Dusk', [5]),
                '05': Category('05: Miscellaneous', [7, 8, 9]),
                'Unclassified': Category('Unclassified', [])
            },
            'VALIGN': {
                '01': Category('01: Straight', [1]),
                '02': Category('02: Curve', [2, 3, 4]),
                '03': Category('03: Non-Trafficway', [0]),
                '04': Category('04: Miscellaneous', [8, 9]),
                'Unclassified': Category('Unclassified', [])
            },
        }

        for var, cat_map in categories.items():
            lookup = {v: k for k, cat in cat_map.items() for v in cat.values}
            new_col = var + '_category'
            if (var in input_df.columns):
                input_df[new_col] = input_df[var].apply(
                    lambda x: lookup[x] 
                    if x in lookup 
                    else (_ for _ in ()).throw(ValueError(f'Value {x} in column {var} is not captured by any category')))

    else:

        categories = {
            'ACC_TYPE': {
                '01': Category('01: Single Driver -- Right Roadside Departure', [1, 2, 3, 4, 5]),
                '02': Category('02: Single Driver -- Left Roadside Departure', [6, 7, 8, 9, 10]),
                '03': Category('03: Single Driver -- Forward Impact', [11, 12, 13, 14, 15, 16]),
                '04': Category('04: Same Trafficway, Same Direction -- Rear End', [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]),
                '05': Category('05: Same Trafficway, Same Direction -- Forward Impact', [34, 35, 36, 37, 38, 39, 40, 41, 42, 43]),
                '06': Category('06: Same Trafficway, Same Direction -- Sideswipe/Angle', [44, 45, 46, 47, 48, 49]),
                '07': Category('07: Same Trafficway, Opposite Direction -- Head-On', [50, 51, 52, 53]),
                '08': Category('08: Same Trafficway, Opposite Direction -- Forward Impact', [54, 55, 56, 57, 58, 59, 60, 61, 62, 63]),
                '09': Category('09: Same Trafficway, Opposite Direction -- Sideswipe/Angle', [64, 65, 66, 67]),
                '10': Category('10: Changing Trafficway, Vehicle Turning -- Turn Across Path', [68, 69, 70, 71, 72, 73, 74, 75]),
                '11': Category('11: Changing Trafficway, Vehicle Turning -- Turn Into Path', [76, 77, 78, 79, 80, 81, 82, 83, 84, 85]),
                '12': Category('12: Intersecting Paths -- Straight Paths', [86, 87, 88, 89, 90, 91]),
                '13': Category('13: Miscellaneous', [92, 93, 97, 98, 99, 0]),
                'Unclassified': Category('Unclassified', [])
            },
            HARM_EV_name: {
                '01': Category('01: Rollover', [1]),
                '02': Category('02: Other Non-Crash', [2, 3, 4, 5, 6, 7, 51, 72]),
                '03': Category('03: Hit Person/Animal', [8, 9, 11, 15, 47, 49]),
                '04': Category('04: Hit Motor Vehicle in Transport', [12, 13, 55]),
                '05': Category('05: Hit Other Vehicle', [10, 14, 45, 74]),
                '06': Category('06: Hit With Object', [16, 17, 18, 54, 73, 91]),
                '07': Category('07: Hit Building/Barrier', [19, 20, 21, 22, 23, 24, 25, 26, 38, 39, 52, 57]),
                '08': Category('08: Hit Sign/Post', [27, 28, 29, 30, 31, 46, 59]),
                '09': Category('09: Crashed into Road Structure', [32, 33, 34, 35, 36, 37, 50, 58]),
                '10': Category('10: Hit Natural Object', [41, 42, 48]),
                '11': Category('11: Hit Other Object', [40, 43, 53, 93]),
                '12': Category('12: Pavement Surface Irregularity', [44]),
                '13': Category('13: Miscellaneous', [98, 99]),
                'Unclassified': Category('Unclassified', [])
            },
            'P_CRASH2': {
                '01': Category('01: Lost Ctl Due To -- Poor Road Conditions', [5]),
                '02': Category('02: Lost Ctl Due To -- Traveling Too Fast for Conditions', [6]),
                '03': Category('03: Lost Ctl Due To -- Other', [1, 2, 3, 4, 8, 9]),
                '04': Category('04: This Veh Traveling -- Over Lane Line', [10, 11]),
                '05': Category('05: This Veh Traveling -- Off Road Edge', [12, 13]),
                '06': Category('06: This Veh Traveling -- Turning', [15, 16, 21]),
                '07': Category('07: This Veh Traveling -- Driving Through Road/Junction', [14, 17]),
                '08': Category('08: This Veh Traveling -- Decelerating', [18]),
                '09': Category('09: This Veh Traveling -- Other', [19, 20]),
                '10': Category('10: Other Veh in Lane -- Stopped/Slower', [50, 51, 52]),
                '11': Category('11: Other Veh in Lane -- Faster', [53]),
                '12': Category('12: Other Veh in Lane -- Oppo Direction', [54]),
                '13': Category('13: Other Veh in Lane -- Other', [55, 56, 59]),
                '14': Category('14: Other Veh into Lane -- From Adj Lane', [60, 61]),
                '15': Category('15: Other Veh into Lane -- From Oppo Direction', [62, 63]),
                '16': Category('16: Other Veh into Lane -- From Parking/Driveway', [64, 70, 71, 72, 73]),
                '17': Category('17: Other Veh into Lane -- From Crossing Street', [65, 66, 67, 68]),
                '18': Category('18: Other Veh into Lane -- From Highway Entrance', [74]),
                '19': Category('19: Other Veh into Lane -- Other', [78]),
                '20': Category('20: Person -- Pedestrian', [80, 81, 82]),
                '21': Category('21: Person -- Pedalcyclist/Non-Motorist', [83, 84, 85]),
                '22': Category('22: Animal/Object -- Animal', [87, 88, 89]),
                '23': Category('23: Animal/Object -- Object', [90, 91, 92]),
                '24': Category('24: Miscellaneous', [98, 99]),
                'Unclassified': Category('Unclassified', [])
            },
            LGT_COND_name: {
                '01': Category('01: Daylight', [1]),
                '02': Category('02: Dark', [2, 3, 6]),
                '03': Category('03: Dawn', [4]),
                '04': Category('04: Dusk', [5]),
                '05': Category('05: Miscellaneous', [7, 8, 9]),
                'Unclassified': Category('Unclassified', [])
            },
            'VALIGN': {
                '01': Category('01: Straight', [1]),
                '02': Category('02: Curve', [2, 3, 4]),
                '03': Category('03: Non-Trafficway', [0]),
                '04': Category('04: Miscellaneous', [8, 9]),
                'Unclassified': Category('Unclassified', [])
            },
        }

        for var, cat_map in categories.items():
            lookup = {v: k for k, cat in cat_map.items() for v in cat.values}
            for suffix in ['_1', '_2']:
                var_col = var + suffix
                new_col = var + '_category' + suffix
                if (var_col in input_df.columns):
                    input_df[new_col] = input_df[var_col].apply(
                        lambda x: lookup[x] if x in lookup else (_ for _ in ()).throw(
                            ValueError(f'Value {x} in column {var_col} is not captured by any category')
                        )
                    )

def get_two_vehicle_columns(df, base_columns):
    '''
    Returns column names in _1, _2 order for multi-vehicle data.
    '''
    vars_1 = [f'{col}_1' for col in base_columns if f'{col}_1' in df.columns and 'VEH_NUM' not in col]
    vars_2 = [f'{col}_2' for col in base_columns if f'{col}_2' in df.columns and 'VEH_NUM' not in col]
    return vars_1 + vars_2

def prepare_gower_inputs(input_df, is_one, cat_var_weights, num_var_weights):
    '''
    Function that builds a feature table, encodes categories, assembles weights, and indexes 
    helpers before running Gower's distance calculation.
    '''

    if (is_one):

        cat_vars = list(cat_var_weights.keys())
        num_vars = list(num_var_weights.keys())
        feature_cols = cat_vars + num_vars

        data_for_gower = pd.DataFrame()
        for col in feature_cols:
            if (col in num_vars):
                data_for_gower[col] = pd.to_numeric(input_df[col], errors='coerce').astype(float)
            else:
                data_for_gower[col] = input_df[col].astype('object')

        num_data = data_for_gower[num_vars].to_numpy()
        encoder = OrdinalEncoder(dtype=np.int32)
        cat_data = encoder.fit_transform(data_for_gower[cat_vars])

        weights_cat = np.array([cat_var_weights[col] for col in cat_vars], dtype=np.float64)
        weights_num = np.array([num_var_weights[col] for col in num_vars], dtype=np.float64)

        vnum_lan_idx = num_vars.index('VNUM_LAN')
        vspd_lim_idx = num_vars.index('VSPD_LIM')

    else:

        cat_vars = get_two_vehicle_columns(input_df, list(cat_var_weights.keys()))
        num_vars = get_two_vehicle_columns(input_df, list(num_var_weights.keys()))
        feature_cols = cat_vars + num_vars

        data_for_gower = pd.DataFrame()
        for col in feature_cols:
            if any(base in col for base in num_var_weights):
                data_for_gower[col] = pd.to_numeric(input_df[col], errors='coerce').astype(float)
            else:
                data_for_gower[col] = input_df[col].astype('object')

        num_data = data_for_gower[[col for col in data_for_gower.columns 
                                   if any(base in col for base in num_var_weights)]].to_numpy()
        encoder = OrdinalEncoder(dtype=np.int32)
        cat_data = encoder.fit_transform(data_for_gower[[col for col in data_for_gower.columns 
                                                         if col not in num_vars]])

        weights_cat = np.array([cat_var_weights[c[:-2]] for c in cat_vars], dtype=np.float64)
        weights_num = np.array([num_var_weights[c[:-2]] for c in num_vars], dtype=np.float64)

        vnum_lan_idx = num_vars.index('VNUM_LAN_1')
        vspd_lim_idx = num_vars.index('VSPD_LIM_1')

    return data_for_gower, num_data, cat_data, weights_num, weights_cat, vnum_lan_idx, vspd_lim_idx

@njit
def compute_min_max(data, vnum_lan_idx, vspd_lim_idx, is_one):
    '''
    Computes per-column min/max, accounting for 'unknown' values.
    This is used for Gower's distance calculation for numerical 
    (non-categorical) data.
    '''

    n_rows, n_cols = data.shape
    min_vals = np.empty(n_cols, dtype=data.dtype)
    max_vals = np.empty(n_cols, dtype=data.dtype)
    half = n_cols // 2 # used only if is_one == False

    for col in range(n_cols):

        min_val = np.inf
        max_val = -np.inf

        for row in range(1, n_rows):
            val = data[row, col]

            if (is_one): # make sure unknown values don't mess up min/max calc (single vehicle)
                if (not ((col == vnum_lan_idx and
                          val in (8, 9)) or
                         (col == vspd_lim_idx and
                          val in (98, 99)))):
                    if (val < min_val):
                        min_val = val
                    if (val > max_val):
                        max_val = val
            else: # make sure unknown values don't mess up min/max calc (multi-vehicle)
                if (not (((col == vnum_lan_idx or
                           col == vnum_lan_idx + half) and
                           val in (8, 9)) or
                         ((col == vspd_lim_idx or
                           col == vspd_lim_idx + half) and
                           val in (98, 99)))):
                    if (val < min_val):
                        min_val = val
                    if (val > max_val):
                        max_val = val

        min_vals[col] = min_val
        max_vals[col] = max_val

    return min_vals, max_vals

@njit
def gower_matrix(data_num, data_cat, weights_num, weights_cat, vnum_lan_idx, vspd_lim_idx, is_one):
    '''
    Calculates the distance matrix by determining the Gower's distance between each crash in the
    input csv. Gower's distance is used here to account for a combination of categorical and 
    numerical data.
    '''

    n = data_num.shape[0]
    n_num = data_num.shape[1]
    n_cat = data_cat.shape[1]
    total_weight = np.sum(weights_num) + np.sum(weights_cat)

    # normalize numerical data
    min_vals, max_vals = compute_min_max(data_num, vnum_lan_idx, vspd_lim_idx, is_one)
    ranges = max_vals - min_vals
    for i in range(len(ranges)):
        if (ranges[i] == 0):
            ranges[i] = 1.0 # avoid division by zero

    data_num_norm = (data_num - min_vals) / ranges
    half = n_num // 2 # used only if is_one == False

    dists = np.zeros((n, n))

    # loop over all pairs of crashes
    for i in range(n):
        for j in range(i, n):
            
            # compute distances for numerical variables, accounting for unknown values
            num_score = 0.0            
            for k in range(n_num):

                a = data_num[i, k]
                b = data_num[j, k]

                if (is_one):
                    if (k == vnum_lan_idx):
                        if ((a in (8, 9)) != (b in (8, 9))):
                            diff = 1.0
                        elif ((a in (8, 9)) and (b in (8, 9))):
                            diff = 0.0
                        else:
                            diff = abs(data_num_norm[i, k] - data_num_norm[j, k])
                    elif (k == vspd_lim_idx):
                        if ((a in (98, 99)) != (b in (98, 99))):
                            diff = 1.0
                        elif ((a in (98, 99)) and (b in (98, 99))):
                            diff = 0.0
                        else:
                            diff = abs(data_num_norm[i, k] - data_num_norm[j, k])
                    else:
                        diff = abs(data_num_norm[i, k] - data_num_norm[j, k])
                else:
                    if (k == vnum_lan_idx or k == vnum_lan_idx + half):
                        if ((a in (8, 9)) != (b in (8, 9))):
                            diff = 1.0
                        elif ((a in (8, 9)) and (b in (8, 9))):
                            diff = 0.0
                        else:
                            diff = abs(data_num_norm[i, k] - data_num_norm[j, k])
                    elif (k == vspd_lim_idx or k == vspd_lim_idx + half):
                        if ((a in (98, 99)) != (b in (98, 99))):
                            diff = 1.0
                        elif ((a in (98, 99)) and (b in (98, 99))):
                            diff = 0.0
                        else:
                            diff = abs(data_num_norm[i, k] - data_num_norm[j, k])
                    else:
                        diff = abs(data_num_norm[i, k] - data_num_norm[j, k])

                num_score += weights_num[k] * diff

            # compute distances for categorical variables
            cat_score = 0.0
            for k in range(n_cat):
                cat_score += weights_cat[k] * (data_cat[i, k] != data_cat[j, k])

            total_dist = (num_score + cat_score) / total_weight
            dists[i, j] = total_dist
            dists[j, i] = total_dist

    return dists

def run_clustering(dist_matrix, data_pca, n_clusters_range, algorithms_to_try, min_cluster_fraction):
    '''
    Loops over each algorithm + number of clusters combination and returns best result
    according to the highest silhouette score.
    '''

    # build algorithm map
    all_algorithms = {
        'Spectral Linear': SpectralClustering,
        'Spectral Gaussian': SpectralClustering,
        'Agglomerative (Average)': lambda n: AgglomerativeClustering(n_clusters=n, metric='precomputed', linkage='average'),
        'Agglomerative (Complete)': lambda n: AgglomerativeClustering(n_clusters=n, metric='precomputed', linkage='complete'),
        'Agglomerative (Single)': lambda n: AgglomerativeClustering(n_clusters=n, metric='precomputed', linkage='single'),
        'KMeans': lambda n: KMeans(n_clusters=n, random_state=42),
        'GMM': lambda n: GaussianMixture(n_components=n, random_state=42),
    }
    algorithms = {k: all_algorithms[k] for k in algorithms_to_try if k in all_algorithms}

    best_score = -1
    best_labels = None
    best_n = None
    best_name = None

    # loop over each clustering algorithm
    for name, algo_builder in algorithms.items():

        print(f'\n=== Running: {name} ===')
        best_score_algo = -1
        best_labels_algo = None
        best_n_algo = None

        # loop over each number of clusters and run clustering algorithm
        for n_clusters in n_clusters_range:

            try:
                if (name == 'Spectral Linear'):
                    algo = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=42)
                    labels = algo.fit_predict(1 - dist_matrix) + 1
                elif (name == 'Spectral Gaussian'):
                    algo = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=42)
                    delta = np.median(dist_matrix[dist_matrix > 0])
                    affinity = np.exp(-dist_matrix ** 2 / (2.0 * delta ** 2))
                    labels = algo.fit_predict(affinity) + 1
                elif ('Agglomerative' in name):
                    algo = algo_builder(n_clusters)
                    labels = algo.fit_predict(dist_matrix) + 1
                else:
                    algo = algo_builder(n_clusters)
                    if (name == 'GMM'):
                        labels = algo.fit(data_pca).predict(data_pca) + 1
                    else:
                        labels = algo.fit_predict(data_pca) + 1

                sil_score = silhouette_score(dist_matrix, labels, metric='precomputed')
                print(f'n_clusters = {n_clusters}, sil_score = {sil_score}')
                if (sil_score > best_score_algo):
                    best_score_algo = sil_score
                    best_labels_algo = labels
                    best_n_algo = n_clusters

            except Exception as e:
                print(f'n_clusters = {n_clusters} failed for {name}: {e}')

        # choose best algorithm+n_clusters combination that returns meaningful
        # clusters above a user-defined minimum size
        num_total_crashes = dist_matrix.shape[0]
        if (best_score_algo > best_score and 
            min(Counter(best_labels_algo).values()) > min_cluster_fraction * num_total_crashes):
            best_score = best_score_algo
            best_labels = best_labels_algo
            best_n = best_n_algo
            best_name = name

    return best_labels, best_name, best_n, best_score

def print_cluster_summary(cluster_df, cluster_name, cluster_id, cluster_indices, is_one):
    '''
    Outputs relevant information for each cluster in the best clustering result, which includes
    cluster size; intra-cluster Gower's distance; the central (medoid) crash; and the 
    mode, medoid, and distribution of variable values.
    '''

    global dist_matrix, input_df, id_column

    # identify medoid (central crash)
    cluster_distances = dist_matrix[np.ix_(cluster_indices, cluster_indices)]
    mean_dists = cluster_distances.mean(axis=1)
    medoid_index = cluster_indices[np.argmin(mean_dists)]
    medoid = input_df.loc[medoid_index]

    count = len(cluster_indices)
    proportion = count / len(input_df)
    mean_intra_dist = np.mean(cluster_distances)
    medoid_case = medoid[id_column]

    # print summary data for each cluster
    print('\n' + '-' * 80)
    print(f'\n{cluster_name} - Cluster {cluster_id}:')
    print(f'\n(A) Cluster size: {count} crashes ({proportion:.2%} of total)')
    print(f'(B) Mean Intra-Cluster Gowers Distance: {mean_intra_dist:.4f}')
    print(f'(C) Medoid {id_column}: {medoid_case}')

    if (is_one):

        cols_to_summarize = [col for col in cluster_df.columns if col != id_column]
        mode_series = cluster_df[cols_to_summarize].mode().iloc[0]
        medoid_series = medoid[cols_to_summarize]

        # mode and medoid variable values
        summary_df = pd.DataFrame({
            'Mode': mode_series,
            'Medoid': medoid_series
        })
        print('\nVariable Summary:')
        pd.set_option('display.max_colwidth', None) # prevent printing truncation
        pd.set_option('display.max_columns', None)  # prevent printing truncation
        print(summary_df.to_string(index_names=False))

        # top 5 variable value distributions
        print('\nTop 5 values per variable:\n')
        dist_rows = []
        for col in cols_to_summarize:
            dist = cluster_df[col].value_counts(normalize=True).head(5).round(2).to_dict()
            dist_rows.append([col, dist])

        dist_df = pd.DataFrame(dist_rows, columns=['Variable', 'Top 5'])
        print(dist_df.set_index('Variable').to_string(index_names=False, header=False))

    else: # slightly different data processing and printing for multi-vehicle data

        v1_cols = [col for col in cluster_df.columns if col.endswith('_1')]
        v2_cols = [col for col in cluster_df.columns if col.endswith('_2')]
        base_names = [col[:-2] for col in v1_cols if col[:-2] in [v[:-2] for v in v2_cols]]

        summary_rows = []
        for base in base_names:
            mode_v1 = cluster_df[f'{base}_1'].mode().iloc[0] if f'{base}_1' in cluster_df else None
            mode_v2 = cluster_df[f'{base}_2'].mode().iloc[0] if f'{base}_2' in cluster_df else None
            medoid_v1 = medoid.get(f'{base}_1', None)
            medoid_v2 = medoid.get(f'{base}_2', None)
            summary_rows.append([base, mode_v1, mode_v2, medoid_v1, medoid_v2])

        # mode and medoid variable values
        summary_df = pd.DataFrame(summary_rows, columns=['Variable', 'Mode (V1)', 'Mode (V2)', 'Medoid (V1)', 'Medoid (V2)'])
        print('\nVariable Summary:')
        print(summary_df.set_index('Variable').to_string(index_names=False))

        # top 5 variable value distributions
        print('\nTop 5 values per variable:\n')
        dist_rows = []
        for base in base_names:
            v1_dist = cluster_df[f'{base}_1'].value_counts(normalize=True).head(5).round(2).to_dict() if f'{base}_1' in cluster_df else {}
            v2_dist = cluster_df[f'{base}_2'].value_counts(normalize=True).head(5).round(2).to_dict() if f'{base}_2' in cluster_df else {}
            dist_rows.append([base, v1_dist, v2_dist])

        dist_df = pd.DataFrame(dist_rows, columns=['Variable', 'Top 5 (V1)', 'Top 5 (V2)'])
        print(dist_df.set_index('Variable').to_string(index_names=False, header=False))

if __name__ == '__main__':

    # directory of this script for data upload/logging
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # prompt user selection of csv in the terminal
    csvs = sorted((Path(script_dir)/'../data_out').glob('*.csv'))
    if (not csvs): 
        raise SystemExit('No CSVs found.')
    print('Select input file (enter=1):')
    print('\n'.join(f'  {i}) {p.name}' for i, p in enumerate(csvs,1)))
    i = input(f'1-{len(csvs)}: ').strip()
    if (i == ''):
        selected = csvs[0]
    else:
        selected = csvs[int(i)-1]
    print('')
    
    # log all text from the terminal into an output txt file
    setup_terminal_logger(selected)
    print(f'Running clustering analysis on: {selected}')

    # load data
    input_df = pd.read_csv(selected, encoding='latin1', low_memory=False)
    print('\nInput Data Dimensions:')
    print(input_df.shape)

    # determine FARS vs GESCRSS and one vs two
    is_fars = ('FARS' in selected.name.upper()) # True if filename contains 'FARS' (ALL CAPS)
    is_one  = ('one'  in selected.name.lower()) # True if filename contains 'one' (lowercase)

    # set variable names depending on FARS vs GESCRSS crash data
    if (is_fars):
        casenum = 'ST_CASE'
        LGT_COND_name = 'LGT_COND'
        WEATHER_name = 'WEATHER'
        P_CRASH1_name = 'P_CRASH1'
        HARM_EV_name = 'HARM_EV'
        RELJCT2_name = 'RELJCT2'
    else:
        casenum = 'CASENUM'
        LGT_COND_name = 'LGTCON_IM'
        WEATHER_name = 'WEATHR_IM'
        P_CRASH1_name = 'PCRASH1_IM'
        HARM_EV_name = 'EVENT1_IM'
        RELJCT2_name = 'RELJCT2_IM'
    id_column = casenum

    #------------------------------------------------------------------#
    #------------------------ START USER INPUT ------------------------#
    #------------------------------------------------------------------#

    # single vehicle crash weights for clustering
    cat_var_weights_one = {
        'RELJCT2': 1.0,
        'TYP_INT': 1.0,
        'REL_ROAD': 1.0,
        'LGT_COND_category': 10.0,
        'WEATHER': 10.0,
        'VTRAFWAY': 10.0,
        'VALIGN_category': 10.0,
        'VPROFILE': 10.0,
        'VSURCOND': 10.0,
        'VTRAFCON': 1.0,
        'P_CRASH1': 10.0,
        'ACC_TYPE_category': 5.0,
        'HARM_EV_category': 5.0,
        'P_CRASH2_category': 20.0
    }
    num_var_weights_one = {
        'VE_FORMS': 0.0,
        'VNUM_LAN': 1.0,
        'VSPD_LIM': 10.0
    }

    # multi-vehicle crash weights for clustering
    cat_var_weights_two = {
        'RELJCT2': 1.0,
        'TYP_INT': 1.0,
        'REL_ROAD': 1.0,
        'LGT_COND_category': 10.0,
        'WEATHER': 10.0,
        'VTRAFWAY': 10.0,
        'VALIGN_category': 10.0,
        'VPROFILE': 10.0,
        'VSURCOND': 10.0,
        'VTRAFCON': 1.0,
        'P_CRASH1': 10.0,
        'ACC_TYPE_category': 30.0,
        'HARM_EV_category': 1.0,
        'P_CRASH2_category': 30.0
    }
    num_var_weights_two = {
        'VE_FORMS': 0.0,
        'VNUM_LAN': 1.0,
        'VSPD_LIM': 30.0
    }

    # set of algorithms to try, default is all of them and user can comment out any
    # note: if you want to add more algorithms than are listed here, you also need
    # to add them as options in the run_clustering() function
    algorithms_to_try = [
        'Spectral Linear',
        'Spectral Gaussian',
        'Agglomerative (Average)',
        'Agglomerative (Complete)',
        'Agglomerative (Single)',
        'KMeans',
        'GMM',
    ]

    # number of clusters to loop over
    min_clusters = 3
    max_clusters = 7
    n_clusters_range = range(min_clusters, max_clusters+1)

    # minimum cluster size, as a fraction of the total number of crashes in the input csv
    min_cluster_fraction = 0.01

    #------------------------------------------------------------------#
    #------------------------- END USER INPUT -------------------------#
    #------------------------------------------------------------------#

    # make sure the weights reflect correct column names for the input csv
    name_map = {
        'RELJCT2': RELJCT2_name,
        'LGT_COND_category': f'{LGT_COND_name}_category',
        'WEATHER': WEATHER_name,
        'P_CRASH1': P_CRASH1_name,
        'HARM_EV_category': f'{HARM_EV_name}_category',
    }
    if (is_one):
        source_cat = cat_var_weights_one
        source_num = num_var_weights_one
    else:
        source_cat = cat_var_weights_two
        source_num = num_var_weights_two
    cat_var_weights = {name_map.get(k, k): v for k, v in source_cat.items()}
    num_var_weights = source_num.copy()

    # map some of the columns to categories to improve clustering
    categorize_data(input_df, is_one, LGT_COND_name, HARM_EV_name)

    # prepare categories and data for Gower's distance calculation
    [data_for_gower, 
     num_data, 
     cat_data,
     weights_num, 
     weights_cat, 
     vnum_lan_idx, 
     vspd_lim_idx] = prepare_gower_inputs(input_df, 
                                          is_one, 
                                          cat_var_weights, 
                                          num_var_weights)

    # calculate distance matrix based on Gower's distance
    dist_matrix = gower_matrix(num_data.astype(np.float64), 
                               cat_data.astype(np.float64), 
                               weights_num.astype(np.float64), 
                               weights_cat.astype(np.float64), 
                               vnum_lan_idx, 
                               vspd_lim_idx,
                               is_one)
    print('\nDistance Matrix:')
    print(dist_matrix)

    # clustering step: compare algorithms and number of clusters
    num_data_for_pca = data_for_gower.select_dtypes(include=[float])
    if (num_data_for_pca.shape[1] > 0):
        n_pca = min(3, num_data_for_pca.shape[1])
        data_pca = PCA(n_components=n_pca).fit_transform(num_data_for_pca)
    else:
        n_pca = 1
        data_pca = np.zeros((len(input_df), 1))
    best_labels, best_name, best_n, best_score = run_clustering(dist_matrix, 
                                                                data_pca, 
                                                                n_clusters_range, 
                                                                algorithms_to_try,
                                                                min_cluster_fraction)

    # lastly, print summary of cluster results
    if (best_labels is not None):

        print(f'\n======================= Best Clustering for {best_name} =======================')
        print(f'\nBest n_clusters: {best_n}')
        print(f'Silhouette Score: {best_score:.3f}')
        print(f'Cluster sizes: {dict(Counter(best_labels))}')
        input_df[f'Cluster_{best_name}'] = best_labels

        cluster_sizes = Counter(best_labels)
        total_cases = len(input_df)
        for c in sorted(set(best_labels)):
            cluster_df = input_df[input_df[f'Cluster_{best_name}'] == c]
            cluster_indices = np.where(input_df[f'Cluster_{best_name}'] == c)[0]
            print_cluster_summary(cluster_df.drop(columns=[col for col 
                                                  in cluster_df.columns 
                                                  if col.startswith('Cluster_')], 
                                                  errors='ignore'),
                                                  best_name,
                                                  c,
                                                  cluster_indices,
                                                  is_one)

    else:

        print(f'{best_name} failed for all n_clusters.')
