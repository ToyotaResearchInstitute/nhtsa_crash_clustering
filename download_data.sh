#!/bin/bash

# make folders to hold input data
SCRIPT_DIR=$(dirname "$(realpath "$0")")
mkdir -p "$SCRIPT_DIR/data_in"
mkdir -p "$SCRIPT_DIR/data_in/FARS" "$SCRIPT_DIR/data_in/GES" "$SCRIPT_DIR/data_in/CRSS"
mkdir -p "$SCRIPT_DIR/data_in/FARS/"{2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022}
mkdir -p "$SCRIPT_DIR/data_in/GES/"{2011,2012,2013,2014,2015}
mkdir -p "$SCRIPT_DIR/data_in/CRSS/"{2016,2017,2018,2019,2020,2021,2022}

# function to download and extract NHTSA data for a given year and database
download_and_extract() {

  local year=$1
  local database=$2
  local destination
  local url
  local filename

  # handle downloading based on the database
  if [[ "$database" == "FARS" ]]; then
    destination="$SCRIPT_DIR/data_in/FARS/$year"
    url="https://static.nhtsa.gov/nhtsa/downloads/FARS/$year/National/FARS${year}NationalCSV.zip"
    filename="FARS${year}NationalCSV.zip"

  elif [[ "$database" == "GES" && $year -ge 2011 && $year -le 2015 ]]; then
    destination="$SCRIPT_DIR/data_in/GES/$year"

    # adjust the URL and filename based on the year for GES
    if [[ "$year" == "2011" ]]; then
      filename="GES11_Flatfile.zip"
    elif [[ "$year" == "2012" ]]; then
      filename="GES12_Flatfile.zip"
    elif [[ "$year" == "2013" ]]; then
      filename="GES13_Flatfile.zip"
    elif [[ "$year" == "2014" ]]; then
      filename="GES2014flat.zip"
    elif [[ "$year" == "2015" ]]; then
      filename="GES2015csv.zip"
    fi
    url="https://static.nhtsa.gov/nhtsa/downloads/GES/GES${year:2:2}/$filename"

  elif [[ "$database" == "CRSS" && $year -ge 2016 && $year -le 2022 ]]; then
    destination="$SCRIPT_DIR/data_in/CRSS/$year"
    url="https://static.nhtsa.gov/nhtsa/downloads/CRSS/$year/CRSS${year}CSV.zip"
    filename="CRSS${year}CSV.zip"

  else
    echo "Invalid combination of year and database for $year/$database"
    return

  fi

  # create target directory and download+extract zip, then remove zip after extraction
  mkdir -p "$destination"
  wget -q "$url" -O "$destination/$filename"
  unzip -qo "$destination/$filename" -d "$destination"

  # if there are any extra subdirectories, flatten them into the main directory
  find "$destination" -mindepth 2 -type f -exec mv {} "$destination" \; 2>/dev/null
  find "$destination" -mindepth 1 -type d -exec rm -r {} \; 2>/dev/null

  # remove zip after extraction
  rm "$destination/$filename"

  # rename files to lowercase and change .txt extensions to .csv
  find "$destination" -type f | while read -r file; do
    filename=$(basename "$file")
    lowercase_filename=$(echo "$filename" | tr '[:upper:]' '[:lower:]')
    if [ "$filename" != "$lowercase_filename" ]; then
        mv "$file" "$(dirname "$file")/$lowercase_filename"
    fi
  done
  find "$destination" -type f -iname "*.txt" -exec sh -c 'mv "$1" "${1%.*}.csv"' _ {} \;

  echo "$year $database data downloaded and extracted successfully"

}

# loop through the years and download NHTSA data files based on the year and database
for year in {2011..2022}
do
  if [[ $year -le 2015 ]]; then
    download_and_extract $year "GES"  # for years 2011-2015, download GES data
  fi
  if [[ $year -ge 2016 && $year -le 2022 ]]; then
    download_and_extract $year "CRSS" # for years 2016-2022, download CRSS data
  fi
  download_and_extract $year "FARS"
done
