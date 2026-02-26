# NHTSA Crash Clustering

A repository for clustering publicly available National Highway Traffic Safety Administration (NHTSA) crash data from the following databases: 
1. Fatality Analysis Reporting System (FARS) -- all motor vehicle crashes in the US resulting in one or more fatalities.
2. General Estimates System (GES) -- a sample of police-reported motor vehicle crashes in the US, active until 2015.
3. Crash Report Sampling System (CRSS) -- a sample of police-reported motor vehicle crashes in the US, replaced GES in 2016.

This repository accompanies the paper [Mixed Methods Scenario Development for Human-Vehicle Interaction Research: A Case Study on Winter Driving](https://doi.org/10.1145/3744335.3758498), which appeared at the 2025 International Conference on Automotive User Interfaces and Interactive Vehicular Applications (AutoUI '25).

## Overview

The high level procedure for extracting representative scenario clusters from NHTSA crash data is as follows:
1. Create a python virtual environment and install all required packages.
2. Download data from NHTSA's online site with the `download_data.sh` bash script.
3. Filter data and separate into single vs multi-vehicle crashes with the `process_data.py` script.
4. Run clustering algorithms on each processed data file with the `cluster_data.py` script.

## Step 1: Setup Virtual Environment

To make sure you have the right environment and packages for running this code, run the following commands:

```bash
# first, git clone this repository
git clone git@github.com:ToyotaResearchInstitute/nhtsa_crash_clustering.git

# create a python virtual environment in the repo directory
cd nhtsa_crash_clustering
python -m venv clustering_env

# activate the virtual environment
source clustering_env/bin/activate

# install the requirements
python -m pip install -r requirements.txt
```

> Note that you will need to activate the virtual environment each time you run the code. Alternatively, you can install the required packages and run the code in your local machine environment.

## Step 2: Download Data

To automatically download the publicly available NHTSA crash data, run the following bash script:

```bash
sudo ./download_data.sh
```

This bash script will pull 2011-2022 data from the FARS, GES, and CRSS databases, which are hosted at the link here: [https://www.nhtsa.gov/file-downloads?p=nhtsa/downloads](https://www.nhtsa.gov/file-downloads?p=nhtsa/downloads/). 

> Note that you need to use sudo priviledges to download some of the data (specifically the 2020 and 2022 FARS datasets). You do not need sudo priviledges to run any of the analysis scripts once the data have been downloaded.

> If there are issues with any of the download links, or you do not have sudo priviledges to download the data, please reach out to Elliot Weiss (elliot.weiss@tri.global) for a local copy of the data.

## Step 3: Process Data

To filter the downloaded data according to user-input conditions and split into processed .csv files for clustering, run the following script:

```bash
python analysis/process_data.py
```

This script creates a `data_out` folder and saves four processed .csv files in this folder: FARS single vehicle crashes, FARS multi-vehicle crashes, GES/CRSS single vehicle crashes, and GES/CRSS multi-vehicle crashes. This enables unique clustering on each of these distinct sets of crashes.

Within the `process_data.py` script, users can revise and/or add their own filter conditions by modifying the code between the comments: 

```bash
#------------------------------------------------------------------#
#------------------------ START USER INPUT ------------------------#
#------------------------------------------------------------------#
```

and 

```bash
#------------------------------------------------------------------#
#------------------------- END USER INPUT -------------------------#
#------------------------------------------------------------------#
```

This script currently filters for winter weather + road surface conditions and light vehicles. A further example is provided to add a filter for nighttime lighting conditions. Any of these filters can be removed, and new filters can be added for many other variables including roadway/intersection type, driver impairment/distraction status, presence of pedestrians or bikers, high-level crash type, crash severity, etc. Please see the data definitions for FARS, GES, and CRSS in the [documentation](documentation) folder when creating custom filter conditions.

## Step 4: Cluster Data

To run clustering analysis on one of the processed .csv files in the `data_out` folder, run the following script:

```bash
python analysis/cluster_data.py
```

This script queries from the user which .csv file to cluster and then tries several clustering algorithm + number of clusters combinations, choosing the combination that results in the best clustering performance. The results for each candidate clustering algorithm and the final clusters are saved in a .txt file in the `data_out` folder.

Within the `cluster_data.py` script, users can change the weights on each variable category (emphasizing certain characteristics of the crash scenarios over others), the set of clustering algorithms to try, the number of clusters to try, and the minimum acceptable cluster size by modifying the code between the comments:  

```bash
#------------------------------------------------------------------#
#------------------------ START USER INPUT ------------------------#
#------------------------------------------------------------------#
```

and 

```bash
#------------------------------------------------------------------#
#------------------------- END USER INPUT -------------------------#
#------------------------------------------------------------------#
```

Given the high dimensional, highly nonlinear, and mostly categorical nature of the crash data, the performance of the clustering algorithms are somewhat sensitive to the choice of variable weights. For new input data, modifying the weights may help lead to more meaningful and distinct cluster categories. 

## Citation

If you find this repository useful, please cite the paper:

```
@inproceedings{weiss2025mixed,
  title={Mixed Methods Scenario Development for Human-Vehicle Interaction Research: A Case Study on Winter Driving},
  author={Weiss, Elliot and Srivatsa, Srijan and Yasuda, Hiroshi and Chen, Tiffany L},
  booktitle={Adjunct Proceedings of the 17th International Conference on Automotive User Interfaces and Interactive Vehicular Applications},
  pages={157--162},
  year={2025}
}
```
