class BuoyData:
    """
        A download_data method that uses the NOAA API to download the data for a particular buoy. This method could take the buoy ID as an input and use it to query the API and retrieve the data for that buoy.

        A store_data method that saves the data for a particular buoy to a local database or file system. This method could take the buoy data as an input and use it to create a new entry in the database or save the data to a file.

        A get_data method that retrieves the data for a particular buoy from the local database or file system. This method could take the buoy ID as an input and use it to query the database or read the data from the file, and then return the data for that buoy.

    """
    def __init__(self, buoy_id, buoy_name, buoy_lat, buoy_lon, buoy_data):
        self.buoy_id = buoy_id
        self.buoy_name = buoy_name
        self.buoy_lat = buoy_lat
        self.buoy_lon = buoy_lon
        self.buoy_data = buoy_data

    def __str__(self):
        return "Buoy ID: " + self.buoy_id + " Buoy Name: " + self.buoy_name + " Buoy Latitude: " + self.buoy_lat + " Buoy Longitude: " + self.buoy_lon + " Buoy Data: " + self.buoy_data

    def download_data(self):
        pass

    def store_data(self):
        pass

    def get_data(self):
        pass
