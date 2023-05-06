import json
import pandas as pd
import requests
import urllib.request

class Buoy:
    """Parent class representing a buoy."""
    def __init__(self, buoy_id="none", temperature="none"):
        """Initialize the Buoy class with optional buoy_id and temperature attributes."""
        self.buoy_id = buoy_id
        self.temperature = temperature

    def check_buoy(self, buoy_id):
        """Check the buoy with the given buoy_id."""
        self.check_data = 'defaults' # put defaults into the code here

class Chosen_Buoy(Buoy):
    """Child class of Buoy representing a chosen buoy."""
    def __init__(self, buoy_type="none"):
        """Initialize the Chosen_Buoy class with buoy_lat, buoy_lng, buoy_depth, buoy_temp, and buoy_atmpressure attributes."""
        super().__init__("Chosen_Buoy")
        self.buoy_lat = 10.0
        self.buoy_lng = 10.0
        self.buoy_depth = 10.0
        self.buoy_temp = 10.0
        self.buoy_atmpressure = 10.0

    def report_out(self, chosen_metric):
        """Report the chosen metric for the Chosen_Buoy."""
        super().report_out(chosen_metric)
        # do things here

def main():
    # Get buoy data.
    product = 'air_temperature'
    station_id = str(8454000)
    start_year = str(2021)
    start_month = '11'
    start_day = str('11')
    end_year = str('2021')
    end_month = str('11')
    end_day = str(start_day)
    start_time_military = str('00:00')
    end_time_military = str('23:59')
    query_date = f'begin_date={start_year}{start_month}{start_day} {start_time_military}&end_date={end_year}{end_month}{end_day} {end_time_military}'
    unit_type = 'metric'
    query_date = 'range=24'
    urlData = f'https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?{query_date}&station={str(station_id)}&product={str(product)}&units={str(unit_type)}&time_zone=gmt&application=ports_screen&format=json'
    urlData = urlData.replace(" ", "%20")
    webUrl = requests.get(urlData,auth=('user','pass'))
    code = webUrl.status_code
    theJSON = webUrl.json()
    if code == 200:
        data = webUrl.text
    else:
        data = "na"
    if "title" in theJSON['metadata']:
        print(theJSON['metadata']['title'])
        # get air_temperature

    # Get buoy data from second method.
    urlData_obs = 'https://www.ndbc.noaa.gov/data/latest_obs/latest_obs.txt'
    webUrl_obs = requests.get(urlData_obs,auth=('user','pass'))
    fh = webUrl_obs.text
    file_string = str(fh)
    found = True
    while found:
        if file_string.find("  ")>-1:
            file_string = file_string.replace("  "," ")
        else:
            found = False
    file_string = file_string.replace(" ",",")
    file = open("tempfile.csv",'w+')
    file.write(file_string)

    # Convert data to a dictionary.
    dict1 = {}
    data2 = pd.read_csv("tempfile.csv", sep=",", header
