# PySeas


![main](images/PySeasMain.png)


<div align="center">
<h1>


<h1 style= "color:blue; font-size: 50px; text-align: center;">
PySeas

</h1>
<p align="center">
  <!-- Typing SVG by DenverCoder1 - https://github.com/DenverCoder1/readme-typing-svg -->
  <a href="https://github.com/DenverCoder1/readme-typing-svg"><img src="https://readme-typing-svg.demolab.com/?lines=watch+12+at+2210;sunset+for+32+is+2010;Looking+at+Buoy+42001;sunset+for+20+is+2110;sunset+for+12+is+2210;sunset+for+13+is+2110;sunset+for+27+is+2110;sunset+for+3+is+2210;sunset+for+48+is+1610;watch+15+at+2110;sunset+for+7+is+2210;sunset+for+53+is+1510;sunset+for+8+is+2210;Looking+at+Buoy+42040;Looking+at+Buoy+46066;sunset+for+12+is+2210;watch+12+at+2210;Looking+at+Buoy+45003;watch+3+at+2210;sunset+for+20+is+2110;Looking+at+Buoy+41001;sunset+for+0+is+2310;watch+6+at+2210;watch+23+at+2110;sunset+for+37+is+1910;watch+18+at+2110;Looking+at+Buoy+46047;watch+10+at+2210;Looking+at+Buoy+42002;Looking+at+Buoy+42001;sunset+for+50+is+1610;sunset+for+37+is+1910;Looking+at+Buoy+42012;watch+5+at+2210;sunset+for+59+is+1510;watch+44+at+1610;watch+36+at+1910;sunset+for+56+is+1510;Looking+at+Buoy+46085;Looking+at+Buoy+51000;Looking+at+Buoy+46002;sunset+for+42+is+1710;sunset+for+2+is+2210;watch+17+at+2110;sunset+for+37+is+1910;sunset+for+24+is+2110;sunset+for+42+is+1710;watch+9+at+2210;sunset+for+32+is+2010;sunset+for+12+is+2210;watch+9+at+2210;watch+4+at+2210;Looking+at+Buoy+41008;watch+25+at+2110;sunset+for+20+is+2110;Looking+at+Buoy+44027;sunset+for+35+is+1910;sunset+for+40+is+1710;Looking+at+Buoy+46072;watch+58+at+1510;Looking+at+Buoy+46089;Looking+at+Buoy+46072;Looking+at+Buoy+51000;sunset+for+9+is+2210;watch+13+at+2110;Looking+at+Buoy+45003;sunset+for+38+is+1810;sunset+for+41+is+1710;sunset+for+54+is+1510;sunset+for+49+is+1610;Looking+at+Buoy+44007;watch+33+at+2010;Looking+at+Buoy+42060;sunset+for+47+is+1610;sunset+for+24+is+2110;sunset+for+12+is+2210;Looking+at+Buoy+51101;watch+27+at+2110;watch+18+at+2110;Looking+at+Buoy+46072;Looking+at+Buoy+41049;Looking+at+Buoy+51001;sunset+for+44+is+1610;Looking+at+Buoy+42002;sunset+for+13+is+2110;sunset+for+33+is+2010;watch+53+at+1510;Looking+at+Buoy+51002;sunset+for+35+is+1910;watch+29+at+2010;sunset+for+37+is+1910;sunset+for+36+is+1910;sunset+for+51+is+1510;Looking+at+Buoy+46066;Looking+at+Buoy+46059;sunset+for+29+is+2010;watch+17+at+2110;watch+31+at+2010;Looking+at+Buoy+46071;sunset+for+31+is+2010;The+optimal+time+for+sunset+at+buoy+31+is+2010;The+optimal+time+for+sunset+at+buoy+44+is+1610;Looking+at+Buoy+41046;&font=menlo%20Code&center=true&width=440&height=45&color=FFD43B&vCenter=true&size=22&pause=1500" /></a>
</p>

</div>

<div align="center">
<h1>
PySeas Purpose
</h1>
</div>
The world's oceans are an untapped wealth of information that we are only barely beginning to understand. More of the ocean has been untouched by man than any other place on earth.

# Using PySeas as a enVita Artist Agent
Our first application of this project is to create art with the images from these buoys, and use them to generate a tapestry of the beautiful oceans.

## Phase One: Sunrise over the Sea

Create sunsets over the sea using the images from the NOAA API.


## Phase Two: The Raging of the Storm

Find images of storms and hurricanes, and create a time-lapse of the storm.



## The Functions of PySeas

1. get_image_size(img): This function accepts an image and returns its width and height.

2. mse_between_arrays(arr1, arr2): This function calculates the mean squared error (MSE) between two arrays (arr1 and arr2) and returns the result.

3. crop_the_bottom_off(images): This function accepts one or multiple images and crops 20 pixels off the bottom of each image.

4. download_image(image_url): This function downloads an image from the specified URL and checks if it's too white. If the image is too white, it's considered a failing image and is recorded in a CSV file. The function returns the image if it's not too white.

5. resize_image_to_standard_height(image, target_height): This function resizes the input image to the target height while maintaining its aspect ratio.

6. split_image_into_panels(resized_image, num_panels): This function takes a resized image and the number of panels to split it into. It returns a list of panels.

7. check_unusual_panels(panels, mse_threshold): This function accepts a list of panels and an MSE threshold value. It checks if any of the panels have an MSE value greater than the threshold and returns a tuple containing a list of unusual panels and a list of rich-color panels.

8. detect_horizon_line(img): This function detects the horizon line of an image and returns the angle of the detected line.

9. align_horizon_line(img): This function aligns the input image by rotating it to correct the tilt angle of the horizon line.

10. stitch_aligned_images(aligned_images): This function stitches a list of aligned images into a single panoramic image.

# These functions work together in the following ways:

An image is downloaded using the download_image() function.
The image is resized to a standard height using resize_image_to_standard_height().
The resized image is split into panels using split_image_into_panels().
Unusual panels are checked using the check_unusual_panels() function.
If there are unusual panels, the horizon line is detected and aligned for each panel using detect_horizon_line() and align_horizon_line(), respectively.
The aligned images are stitched together into a panoramic image using stitch_aligned_images().







# License
PySeas is licensed under the MIT License. See [LICENSE](LICENSE) for more information.

# Contributing
Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for more information.













# Acknowledgements


# The Sunset Tapestry

<div align="center">

![The Tapestry of the Ocean's Life](images/master_stitch.png)

</div>