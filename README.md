## PyBuoy Module Repository
========================

<div id="top"></div>
<!--
*** Thanks for checking out the PyBuoy. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][grahamwaters-linkedin-url]

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/grahamwaters/PyBuoy">
    <img src="images/PyBuoy_Main.png" alt="Logo" width="1640" height="200">
  </a>

  <h3 align="center">PyBuoy</h3>

  <p align="center">
    Generating a sense of randomness from the world's oceans.
    <br />
    <a href="https://github.com/grahamwaters/PyBuoy"><strong>Explore the docs »</strong></a>
    <br />
    <a href="https://github.com/grahamwaters/PyBuoy/issues">Report Bug</a>
    ·
    <a href="https://github.com/grahamwaters/PyBuoy/issues">Request Feature</a>
  </p>
</div>

Feel free to share this project on Twitter!

[![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/intent/tweet?text=Get%20over%20170%20free%20design%20blocks%20based%20on%20Bootstrap%204&url=https://froala.com/design-blocks&via=froala&hashtags=bootstrap,design,templates,blocks,developers)


### Basics

* Station ID - A 7 character station ID, or a currents station ID. Specify the station ID with the "station=" parameter.
> Example: station=9414290
Station listings for various products can be viewed at https://tidesandcurrents.noaa.gov or viewed on a map at Tides & Currents Station Map
* Date & Time -
The API understands several parameters related to date ranges.

* All dates can be formatted as follows:
yyyyMMdd, yyyyMMdd HH:mm, MM/dd/yyyy, or MM/dd/yyyy HH:mm

One the 4 following sets of parameters can be specified in a request:

* Parameter Name (s)
* Description
* begin_date
* end_date
Specify the date/time range of retrieval
date
Valid options for the date parameters are: latest (last data point available within the last 18 min), today, or recent (last 72 hours)
begin_date and a range

Specify a begin date and a number of hours to retrieve data starting from that date
end_date and a range

Specify an end date and a number of hours to retrieve data ending at that date
range

Specify a number of hours to go back from now and retrieve data for that date range

January 1st, 2012 through January 2nd, 2012
    begin_date=20120101&end_date=20120102
48 hours beginning on April 15, 2012
    begin_date=20120415&range=48
48 hours ending on March 17, 2012
    end_date=20120307&range=48
Today's data
    date=today
The last 3 days of data
    date=recent
The last data point available within the last 18 min
    date=latest
The last 24 hours from now
    range=24
The last 3 hours from now
    range=3

Data Products
Specify the type of data with the "product=" option parameter.

Option 	Description
water_level 	Preliminary or verified water levels, depending on availability.
air_temperature 	Air temperature as measured at the station.
water_temperature 	Water temperature as measured at the station.
wind 	Wind speed, direction, and gusts as measured at the station.
air_pressure 	Barometric pressure as measured at the station.
air_gap 	Air Gap (distance between a bridge and the water's surface) at the station.
conductivity 	The water's conductivity as measured at the station.
visibility 	Visibility from the station's visibility sensor. A measure of atmospheric clarity.
humidity 	Relative humidity as measured at the station.
salinity 	Salinity and specific gravity data for the station.
hourly_height 	Verified hourly height water level data for the station.
high_low 	Verified high/low water level data for the station.
daily_mean 	Verified daily mean water level data for the station.
monthly_mean 	Verified monthly mean water level data for the station.
one_minute_water_level 	One minute water level data for the station.
predictions 	6 minute predictions water level data for the station.*
datums 	datums data for the stations.
currents 	Currents data for currents stations.
currents_predictions 	Currents predictions data for currents predictions stations.






<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/grahamwaters/PyBuoy.svg?style=for-the-badge
[contributors-url]: https://github.com/grahamwaters/PyBuoy/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/grahamwaters/PyBuoy.svg?style=for-the-badge
[forks-url]: https://github.com/grahamwaters/PyBuoy/network/members
[stars-shield]: https://img.shields.io/github/stars/grahamwaters/PyBuoy.svg?style=for-the-badge
[stars-url]: https://github.com/grahamwaters/PyBuoy/stargazers
[issues-shield]: https://img.shields.io/github/issues/grahamwaters/PyBuoy.svg?style=for-the-badge
[issues-url]: https://github.com/grahamwaters/PyBuoy/issues
[license-shield]: https://img.shields.io/github/license/grahamwaters/PyBuoy.svg?style=for-the-badge
[license-url]: https://github.com/grahamwaters/PyBuoy/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555

[nura-linkedin-url]:https://www.linkedin.com/in/nura-abuassaf/
[grahamwaters-linkedin-url]: https://linkedin.com/in/grahamwaters01
[product-screenshot]: images/screenshot.png
[iso1]: images/example_isometric_scene.png
[iso2]: images/example_isometricscene2.png
[iso_goal]: images/goal_visual_iso.jpg

[pybuoymain]: images/PyBuoy_Main.png
