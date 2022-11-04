
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

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/grahamwaters/PyBuoy">
    <img src="images/PyBuoy_Main.png" alt="Logo" width="1640" height="200">
  </a>

  <h1 align="center">PyBuoy</h3>

  <h4 align="center">
    Generating a sense of randomness from the world's oceans.
    <br />
    <a href="https://github.com/grahamwaters/PyBuoy"><strong>Explore the docs »</strong></a>
    <br />
    <a href="https://github.com/grahamwaters/PyBuoy/issues">Report Bug</a>
    ·
    <a href="https://github.com/grahamwaters/PyBuoy/issues">Request Feature</a>
  </p>
</div>

<div align='center'>

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][grahamwaters-linkedin-url]

</div>

Feel free to share this project on Twitter!

[![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/intent/tweet?text=Get%20over%20170%20free%20design%20blocks%20based%20on%20Bootstrap%204&url=https://froala.com/design-blocks&via=froala&hashtags=bootstrap,design,templates,blocks,developers)

## Table of Contents

- [Installation](#installation)
  - [Installing Dependencies](#installing-dependencies)
  - [About The Project](#about-the-project)
    - [Built With](#built-with)
  - [Acknowledgements](#acknowledgements)
  - [Getting Started](#getting-started)
    - [What Data is Available?](#what-data-is-available)
  - [See Buoy Data Retrieval Example](#see-buoy-data-retrieval-example)
  - [Usage](#usage)

# Installation

```bash
pip install pybuoy
```

## Installing Dependencies

```bash
pip install -r requirements.txt
```

<!-- ABOUT THE PROJECT -->
## About The Project

The PyBuoy module is a Python package that allows users to generate a sense of randomness from the world's oceans. The basic premise of the code is that randomness is best defined by nature, not by computers and thus the more we can take from nature to inform our *random* numbers, the more truely random they become. The implementation is fairly straight-forward. It is based on the concept of a buoy, which is a floating object that is used to measure the ocean's surface. The PyBuoy module uses the buoy's location to generate a random number.

There are several key data features that are used to generate the random number. These include:
* Latitude
* Longitude
* Wind Speed
* Wind Direction
* Wave Height

PyBuoy uses these data features to generate a random number in a range specified by the user. The random number could be much larger than the range but continuous operations are performed by the module on the result until it is effectively normalized and lies between the min and max given by that range. The random number is then returned to the user.

### Built With
<!-- This section should list any major frameworks that you built your project using. Leave any add-ons/plugins for the acknowledgements section. Here are a few examples.
 -->
* [Python](https://www.python.org/)
* [Numpy](https://numpy.org/)
* [Pandas](https://pandas.pydata.org/)
* [Matplotlib](https://matplotlib.org/)


<!-- Acknowledgements: -->
## Acknowledgements
* [Buoy Data](https://www.ndbc.noaa.gov/)
* [See Buoy Repository](https://github.com/nickc1/seebuoy) - This project has done a great job of providing a simple API for accessing buoy data. I have used this project as a reference for my work and am grateful for the work that has been done on it to enable this easy access.

<!-- GETTING STARTED -->
## Getting Started

The data we want to access is available through the National Data Buoy Center (NDBC). The NDBC is a part of the National Oceanic and Atmospheric Administration (NOAA).
### What Data is Available?
- There is a widget (html) that can be imbedded in a website that provides a graphical representation of the buoy's data.
- Photos from the buoy can be accessed (see this [link](https://www.ndbc.noaa.gov/images/buoycam/Z72A_2022_11_04_2210.jpg)).
-


```html
<iframe src="https://www.ndbc.noaa.gov/widgets/station_page.php?station=41004" style="border: solid thin #3366ff; width:300px; height:300px"></iframe>
```

## See Buoy Data Retrieval Example

The following is taken from the [See Buoy Repository](https://github.com/nickc1/seebuoy) and is a great example of how to access the buoy data.

```python
from seebuoy import ndbc

df = ndbc.real_time('41013')
df.head()
```
What I like about this tool is that it provides an easy method of accessing not just one data element but all of the data elements that are available. This is a great way to get a sense of what can be gathered. The result is a pandas dataframe with the date in datetime format on the left.

<div align = "center">
    <img src="images/PyBuoy_DataFrame.png" alt="Logo" width="1640" height="200">
</div>











## Usage

When using the PyBuoy module, the user must first import the module. The module is imported as follows:

```python
import pybuoy
```

The user can then create a `Buoy` object. The `Buoy` object takes the following arguments:

* `buoy_id` - The ID of the buoy. This is a string.
* `min` - The minimum value of the range. This is an integer.
* `max` - The maximum value of the range. This is an integer.

The `Buoy` object is created as follows:

```python
buoy = pybuoy.Buoy(buoy_id='44025', min=0, max=100)
```

The `Buoy` object has a method called `get_random_number`. This method returns a random number. The random number is generated using the data features listed above. The random number is returned as an integer.

```python
random_number = buoy.get_random_number()
```

The user then has a number that is generated from the ocean's surface. This number is random, but it is also influenced by the ocean's variant movements. So, pure mathematicians will tell you this is not true randomness, and they are technically correct. We posit that it is pretty close, though.




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

[grahamwaters-linkedin-url]: https://linkedin.com/in/grahamwaters01
[product-screenshot]: images/screenshot.png
[iso1]: images/example_isometric_scene.png
[iso2]: images/example_isometricscene2.png
[iso_goal]: images/goal_visual_iso.jpg

[pybuoymain]: images/PyBuoy_Main.png