import pandas as pd
import numpy as np
import requests
import os
import time
from ratelimit import limits, sleep_and_retry
import re
from bs4 import BeautifulSoup
import csv
# from alive_progress import alive_bar
from tqdm import tqdm


# The main.py file is the main code body for the PyBuoy application.
#note: captain_seemore is available as a repo name on GitHub



# Helper functions
@sleep_and_retry
def get_drifting_buoy_data(buoy_id):
    # Get buoy data from NOAA # https://www.ndbc.noaa.gov/data/realtime2/{buoy_id}.drift
    url = f'https://www.ndbc.noaa.gov/data/realtime2/{buoy_id}.drift' # f-string url
    r = requests.get(url) # Get the data from the URL
    with open('data.csv', 'wb') as f: # Write the data to a file
        f.write(r.content) # Write the data to a file
    df = pd.read_csv('data.csv', header=1, parse_dates=True, delimiter = '\s+') # Read in the data
    return df # return the buoy data

@sleep_and_retry
def get_stationary_buoy_data(buoy_id):
    # https://www.ndbc.noaa.gov/data/realtime2/21415.dart
    # example buoy data pull for water column height (capture the latest one)
    url = f'https://www.ndbc.noaa.gov/data/realtime2/{buoy_id}.dart' # f-string url
    r = requests.get(url)
    with open('data.csv', 'wb') as f:
        f.write(r.content)
    df = pd.read_csv('data.csv', header=1, parse_dates=True, delimiter = '\s+')
    df.head()
    return df


# Buoy Cam Data
def get_buoy_cam(buoy_id):
    """
    NDBC operates BuoyCAMs at several stations. These BuoyCAMs typically take photos only during daylight hours.

    To view the most recent BuoyCAM image from an NDBC station, use this URL:

        https://www.ndbc.noaa.gov/buoycam.php?station=xxxxx

    where xxxxx is the desired station ID. To see which stations are currently reporting BuoyCAM images, check the BuoyCAMs map.

    If the server encounters any difficulties in processing your request, you will receive one of these error messages:

        No station specified

        Modify your URL to use the station parameter to specify a valid station with a BuoyCAM (station=xxxxx where xxxxx is the station ID). Review the BuoyCAMs map to see which stations have a BuoyCAM.
        Invalid station specified

        The station on the URL is not a valid station. Review the BuoyCAMs map to see which stations have a BuoyCAM.
        This station has no BuoyCAM

        The station on the URL is valid but has no BuoyCAM installed. Look at the BuoyCAMs map to see which stations have a BuoyCAM.
        BuoyCAM photo for this station is older than 16 hours

        The BuoyCAM on the specified station has not reported in the past 16 hours, hence there is no image to display.
        Unable to access BuoyCAMs at this time

        There is an issue preventing the BuoyCAM process from functioning properly. Recommend waiting at least 30 minutes and trying again, if the problem persist contact the NDBC webmaster with the URL used and the date/time the error was received."""

    # https://www.ndbc.noaa.gov/buoycam.php?station=21415
    url = f'https://www.ndbc.noaa.gov/buoycam.php?station={buoy_id}' # f-string url
    r = requests.get(url) # Get the data from the URL
    r_text = r.text # get the text from the request
    print(r_text)

    with open('data.csv', 'wb') as f:
        f.write(r.content) # this writes in latin-1 encoding
    df = pd.read_csv('data.csv', header=1, parse_dates=True, delimiter = '\s+') # Read in the data
    df.head() # this is a pandas dataframe
    return df # Main code body



def get_available_buoy_ids():

    # source for this data: https://www.ndbc.noaa.gov/to_station.shtml

    # task 1 -  use regex to extract all numbers like 44004 from stations.txt file and save to a list.
    # task 2 - use regex to also identify any identifiers like ALXN6 or ALXN7 and append them to the list as well.
    import re
    stations_text = open('data/stations.txt', 'r').read()
    task_one_stations_list = re.findall(r' \d+ ', stations_text) # find all numbers in the file
    task_two_stations_list = re.findall(r' \w+\d ', stations_text) # find all identifiers in the file
    # combine by adding the second list to the first with .extend()
    task_one_stations_list.extend(task_two_stations_list)
    # remove the spaces from the list
    stations_list = [x.strip() for x in task_one_stations_list] # remove whitespace
    # strip the whitespace from the list
    stations_list = [x.strip() for x in stations_list] # remove whitespace
    # remove duplicates
    stations_list = [x for x in task_one_stations_list if x != ''] # remove empty strings
    stations_list = list(dict.fromkeys(stations_list))
    # remove the first element

    # remove all nonalpha characters from the elements in the list
    stations_list = [re.sub(r'\W+', '', x) for x in stations_list]
    print(len(stations_list), ' stations were identified')
    # print(stations_list)


    return stations_list

# First we need to get the list of available buoy ids
# Then we need to ask the question, "Does this buoy transmit photos? (BuoyCam) Y/N"
# If it does then we want to get the latest photo and save it to the data folder with the buoy id as the file name (every time we run this script we want to overwrite the existing photo with the latest one.)

# buoy_ids = get_available_buoy_ids()
#* Now we have a list of buoy ids that we can use to get photos from (potentially).
#? Question: How do we know if a buoy has a camera? (BuoyCam)
# https://www.ndbc.noaa.gov/buoycam.php?station=XXXXX
# XXXXX is the buoy id
# an example of a buoy with a camera is 44013
# images/buoycam_example.jpg

# Pages that relate to the station 44013
# 1. https://www.ndbc.noaa.gov/station_page.php?station=44013
# This page shows data for the buoy, and includes the photos stitched together into a panorama.
# 2. https://www.ndbc.noaa.gov/buoycam.php?station=44013
# This page shows the latest photo from the buoy cam (stitched together from multiple photos).


# Process Flow:
# 1. First we want to get the list of buoy ids from the stations.txt file. This is done only once at the start of the script, and should be saved to a file so that we don't have to do it again.
# 2. Now that we have the list of buoy ids, we want to loop through them and check if they have a buoy cam.
# 2a. We know that the buoy has a camera if the following text appears on the buoy's page: "Buoy Camera Photos" AND "Click photo to enlarge." (Both of these phrases appear on the page when the buoy has a camera.)
# 2a1. Using requests and bs4 we can search the text of the page for these phrases and determine if the buoy has a camera.
# 2a2. The queries to the NOAA website should be controlled via the ratelimit module so that we don't get blocked. (using the sleep and retry decorators)
# 2b. If the buoy has a camera, then append the station id to a list of buoy ids that have cameras. (this will be saved to a csv file).
# 3. Now that we have a list of buoy ids that have cameras, we want to loop through them and get the latest photo.
# 3a. we want to get the latest photo from the buoy cam and save it to the data folder with the buoy id as the file name (every time we run this script we want to overwrite the existing photo with the latest one.)
#   3b1. We can get the latest photo from the buoy cam by going to the following URL: https://www.ndbc.noaa.gov/buoycam.php?station=XXXXX
#   3b2. XXXXX is the buoy id
#   3b3. an example of a buoy with a camera is 44013



# Code Body:

class ProcessFlow():
    def __init__(self):
        # read in the list of buoy ids from the buoy_ids.csv file
        with open('data/buoy_ids.csv', 'r') as f:
            buoy_ids = f.read()
        self.buoy_ids = buoy_ids.split(',')
        self.buoy_ids_with_camera = []
        self.verbose_output = True # set to False to disable verbose output

    def check_buoys(self):
        # 2. Now that we have the list of buoy ids, we want to loop through them and check if they have a buoy cam.
        for buoy_id in tqdm(self.buoy_ids):
            # check to make sure the buoy id is not in the blacklisted list file (blacklisted_buoy_ids.csv)
            with open('data/blacklisted_buoy_ids.csv', 'r') as f:
                blacklisted_buoy_ids = f.read()
            blacklisted_buoy_ids = blacklisted_buoy_ids.split(',')
            if buoy_id in blacklisted_buoy_ids:
                if self.verbose_output:
                    print(f'buoy id {buoy_id} is blacklisted')
                continue

            try:
                self.check_for_camera(buoy_id)
            except Exception as e:
                print(e)
                time.sleep(5)
                self.check_for_camera(buoy_id)

        print('all buoys have been checked, found', len(self.buoy_ids_with_camera), ' buoys with cameras out of ', len(self.buoy_ids), ' total buoys')

    @limits(calls=15, period=600)
    def check_for_camera(self, buoy_id): # Step 2 in the process flow above
        # 2. Now that we have the list of buoy ids, we want to loop through them and check if they have a buoy cam, we will use bs4, requests, and regex to do this.
        # 2a. We know that the buoy has a camera if the following text appears on the buoy's page: "Buoy Camera Photos" AND "Click photo to enlarge." (Both of these phrases appear on the page when the buoy has a camera.)
        # 2a1. Using requests and bs4 we can search the text of the page for these phrases and determine if the buoy has a camera.
        # 2a2. The queries to the NOAA website should be controlled via the ratelimit module so that we don't get blocked. (using the sleep and retry decorators)
        # 2b. If the buoy has a camera, then append the station id to a list of buoy ids that have cameras. (this will be saved to a csv file).


        #* Step 2a.
        # get the buoy page
        buoy_page = requests.get(f'https://www.ndbc.noaa.gov/station_page.php?station={buoy_id}')
        # parse the buoy page
        buoy_page_soup = BeautifulSoup(buoy_page.text, 'html.parser')
        # get the text from the buoy page
        buoy_page_text = buoy_page_soup.get_text()
        # check if the buoy has a camera
        buoy_has_camera = re.search(r'Buoy Camera Photos', buoy_page_text) and re.search(r'Click photo to enlarge.', buoy_page_text)
        # if the buoy has a camera, then append the buoy id to the list of buoy ids that have cameras
        if buoy_has_camera:
            self.buoy_ids_with_camera.append(buoy_id)
            print(f'buoy {buoy_id} has a camera')
        else:
            print(f'buoy {buoy_id} does not have a camera')
            # add this to the blacklisted buoy ids list
            with open('data/blacklisted_buoy_ids.csv', 'a+') as f:
                f.write(f'{buoy_id},')

    def query_buoy(self, buoy_id):
        # 3a. we want to get the latest photo from the buoy cam and save it to the data folder with the buoy id as the file name (every time we run this script we want to overwrite the existing photo with the latest one.)
        #   3b1. We can get the latest photo from the buoy cam by going to the following URL: https://www.ndbc.noaa.gov/buoycam.php?station=XXXXX
        #   3b2. XXXXX is the buoy id
        #   3b3. an example of a buoy with a camera is 44013
        # get the buoy page
        if self.verbose_output:
            print(f'querying buoy {buoy_id}')
        buoy_page = requests.get(f'https://www.ndbc.noaa.gov/buoycam.php?station={buoy_id}') # 3b1
        with open(f"images/buoys/{buoy_id}_latest.png", "wb+") as f:
            f.write(buoy_page.content) # write the image to the file
        if self.verbose_output:
            print(f'buoy {buoy_id} has been queried with response code {buoy_page.status_code}')


    def query_buoys(self):
        # 3. Now that we have the list of buoy ids with cameras, we want to loop through them and query them for the latest photo.
        for buoy_id in self.buoy_ids_with_camera:
            try:
                self.query_buoy(buoy_id)
            except Exception as e:
                print(e)
                time.sleep(5)
                self.query_buoy(buoy_id)

        print('all buoys have been queried')

    def run(self):
        # ids = self.get_buoy_ids()
        self.check_buoys()
        # self.save_buoyids_tocsv()
        self.query_buoys()


if __name__ == '__main__':
    buoy = ProcessFlow() # instantiate the class
    buoy.run() # run the process flow
