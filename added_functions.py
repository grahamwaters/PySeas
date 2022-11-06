import pandas as pd




# The end goal is to be able to classify images from the buoy cameras in real time as either:
# (1) Beautiful
# (2) Unuseable (due to bad weather)
# (3) Unuseable (due to bad camera angle)
# (4) Unuseable (due to dark conditions)
# (5) Sunset shot
# (6) Sunrise shot
# (7) Useable for a Panorama
# (8) Contains a boat
# (9) Contains a person
# (10) Contains a bird
# (11) Contains a whale
# (12) Contains something unusual


"""
We need training data in order to do this and we can get this from the buoy cameras.
We can get the images from the buoy cameras and then manually classify them as one of the above categories.


_extended_summary_
"""

import cv2
import numpy as np
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
