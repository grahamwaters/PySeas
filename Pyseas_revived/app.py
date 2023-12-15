import os
import time
from flask import Flask, render_template, url_for
import requests
from bs4 import BeautifulSoup
# import app

app = Flask(__name__)

def get_buoy_list():
    # Add your list of buoy IDs here
    buoy_ids = ['41001', '41002', '41003', '41004']
    return buoy_ids

def get_buoy_images(buoy_ids):
    image_urls = []
    base_url = "http://www.ndbc.noaa.gov/buoycam.php?station="

    for buoy_id in buoy_ids:
        url = base_url + buoy_id
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        img_tag = soup.find('img', {"id": "latest_img"})

        if img_tag:
            image_url = img_tag['src']
            if image_url.startswith('/'):
                image_url = 'http://www.ndbc.noaa.gov' + image_url
            image_urls.append((buoy_id, image_url))

    return image_urls

@app.route('/')
def index():
    buoy_ids = get_buoy_list()
    image_urls = get_buoy_images(buoy_ids)
    return render_template('index.html', image_urls=image_urls)

if __name__ == '__main__':
    app.run(debug=True)
