from cam_backend import *

print('Starting the download loop')
last_time_fetched = time.time() # get the current time
first_run = True # set a flag to indicate that this is the first run of the loop (for the first run, we will download rss feeds for all the buoys)

# import the necessary packages
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2


def buoy_links():
    links = ["https://www.ndbc.noaa.gov/buoycam.php?station=42001","https://www.ndbc.noaa.gov/buoycam.php?station=46059","https://www.ndbc.noaa.gov/buoycam.php?station=41044","https://www.ndbc.noaa.gov/buoycam.php?station=46071","https://www.ndbc.noaa.gov/buoycam.php?station=42002","https://www.ndbc.noaa.gov/buoycam.php?station=46072","https://www.ndbc.noaa.gov/buoycam.php?station=46066","https://www.ndbc.noaa.gov/buoycam.php?station=41046","https://www.ndbc.noaa.gov/buoycam.php?station=46088","https://www.ndbc.noaa.gov/buoycam.php?station=44066","https://www.ndbc.noaa.gov/buoycam.php?station=46089","https://www.ndbc.noaa.gov/buoycam.php?station=41043","https://www.ndbc.noaa.gov/buoycam.php?station=42012","https://www.ndbc.noaa.gov/buoycam.php?station=42039","https://www.ndbc.noaa.gov/buoycam.php?station=46012","https://www.ndbc.noaa.gov/buoycam.php?station=46011","https://www.ndbc.noaa.gov/buoycam.php?station=42060","https://www.ndbc.noaa.gov/buoycam.php?station=41009","https://www.ndbc.noaa.gov/buoycam.php?station=46028","https://www.ndbc.noaa.gov/buoycam.php?station=44011","https://www.ndbc.noaa.gov/buoycam.php?station=41008","https://www.ndbc.noaa.gov/buoycam.php?station=46015","https://www.ndbc.noaa.gov/buoycam.php?station=42059","https://www.ndbc.noaa.gov/buoycam.php?station=44013","https://www.ndbc.noaa.gov/buoycam.php?station=44007","https://www.ndbc.noaa.gov/buoycam.php?station=46002","https://www.ndbc.noaa.gov/buoycam.php?station=51003","https://www.ndbc.noaa.gov/buoycam.php?station=46027","https://www.ndbc.noaa.gov/buoycam.php?station=46026","https://www.ndbc.noaa.gov/buoycam.php?station=51002","https://www.ndbc.noaa.gov/buoycam.php?station=51000","https://www.ndbc.noaa.gov/buoycam.php?station=42040","https://www.ndbc.noaa.gov/buoycam.php?station=44020","https://www.ndbc.noaa.gov/buoycam.php?station=46025","https://www.ndbc.noaa.gov/buoycam.php?station=41010","https://www.ndbc.noaa.gov/buoycam.php?station=41004","https://www.ndbc.noaa.gov/buoycam.php?station=51001","https://www.ndbc.noaa.gov/buoycam.php?station=44025","https://www.ndbc.noaa.gov/buoycam.php?station=41001","https://www.ndbc.noaa.gov/buoycam.php?station=51004","https://www.ndbc.noaa.gov/buoycam.php?station=44027","https://www.ndbc.noaa.gov/buoycam.php?station=41002","https://www.ndbc.noaa.gov/buoycam.php?station=42020","https://www.ndbc.noaa.gov/buoycam.php?station=46078","https://www.ndbc.noaa.gov/buoycam.php?station=46087","https://www.ndbc.noaa.gov/buoycam.php?station=51101","https://www.ndbc.noaa.gov/buoycam.php?station=46086","https://www.ndbc.noaa.gov/buoycam.php?station=45002","https://www.ndbc.noaa.gov/buoycam.php?station=46053","https://www.ndbc.noaa.gov/buoycam.php?station=46047","https://www.ndbc.noaa.gov/buoycam.php?station=46084","https://www.ndbc.noaa.gov/buoycam.php?station=46085","https://www.ndbc.noaa.gov/buoycam.php?station=45003","https://www.ndbc.noaa.gov/buoycam.php?station=45007","https://www.ndbc.noaa.gov/buoycam.php?station=46042","https://www.ndbc.noaa.gov/buoycam.php?station=45012","https://www.ndbc.noaa.gov/buoycam.php?station=42019","https://www.ndbc.noaa.gov/buoycam.php?station=46069","https://www.ndbc.noaa.gov/buoycam.php?station=46054","https://www.ndbc.noaa.gov/buoycam.php?station=41049","https://www.ndbc.noaa.gov/buoycam.php?station=45005"]
    return links

# Notes:
# Buoy 42002 Has good sunsets



def ocean_stitching(imagePaths, pano_path):
    images = []
    # loop over the image paths, load each one, and add them to our
    # images to stitch list
    # open each image with cv2 and append to images list
    for imagePath in imagePaths:
        try:
            # add the full path to the image
            # '/Volumes/Backups of Grahams IMAC/PythonProjects/PySeas_Master_Folder/PySeas/images/panels/44020/2022_11_6_10_54/panel_1.jpg'
            # read the image
            full_path =  '/Volumes/Backups of Grahams IMAC/PythonProjects/PySeas_Master_Folder/PySeas/' + imagePath
            imagePath = full_path # this is the full path to the image
            assert(os.path.exists(imagePath)) # check if file exists
            image = cv2.imread(imagePath, cv2.IMREAD_COLOR) # read image
            #cv2.imshow("image", image)
            #cv2.waitKey(0) # show image
            images.append(image) # append to list
            if image is None:
                print("Error loading image: " + imagePath)
        except:
            print("Error reading image: " + imagePath)
            continue
    # getting error: OpenCV(4.6.0) /Users/runner/work/opencv-python/opencv-python/opencv/modules/imgproc/src/resize.cpp:4052: error: (-215:Assertion failed) !ssize.empty() in function 'resize'
    # how to fix this:
    # The error is saying that the image is empty. This is because you are trying to read an image that doesn't exist. Check the path to the image and make sure it is correct.

    # initialize OpenCV's image stitcher object and then perform the image
    # stitching
    print("[INFO] stitching images...")
    stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
    (status, stitched) = stitcher.stitch(images)
    # if the status is '0', then OpenCV successfully performed image
    # stitching
    if status == 0:
        # write the output stitched image to disk
        cv2.imwrite(pano_path, stitched) # save the stitched image
        # display the output stitched image to our screen
        #!cv2.imshow("Stitched", stitched)
        #!cv2.waitKey(0)
    # otherwise the stitching failed, likely due to not enough keypoints)
    # being detected
    else:
        print("[INFO] image stitching failed ({})".format(status))


def refine_view(stitched_image):
    stitched = cv2.copyMakeBorder(stitched, 10, 10, 10, 10,
			cv2.BORDER_CONSTANT, (0, 0, 0))
    # convert the stitched image to grayscale and threshold it
    # such that all pixels greater than zero are set to 255
    # (foreground) while all others remain 0 (background)
    gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
    # find all external contours in the threshold image then find
    # the *largest* contour which will be the contour/outline of
    # the stitched image
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    # allocate memory for the mask which will contain the
    # rectangular bounding box of the stitched image region
    mask = np.zeros(thresh.shape, dtype="uint8")
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)




def chunk_images(buoy_id,foldername):
    """
    chunk_images takes a folder of images and splits them into sets of 6 images

    _extended_summary_

    :param buoy_id: The id of the buoy
    :type buoy_id: int
    :param foldername: The name of the folder containing the images
    :type foldername: str
    # """
    # #buoy_id = str(foldername)
    # # There are many images in the folder. We want to parse these in sets of six.
    # # images/panels/41002/2022_11_5_15_44_panel_1.png is an example of the naming convention for the images.
    # # the first number (folder under panels) is the buoy id, then the filename begins with the year, the second number is the month, the third number is the day, the fourth number is the hour, the fifth number is the minute, and the sixth number is the panel number.
    # # we want to select the images that have the same buoy id, year, month, day, and hour AND end in the panel numbers: 1,2,3,4,5,6.

    # # for the example file above 2022_11_5_15_44_panel_1.png
    # # we want to put it into a subfolder named 2022_11_5_15 and put all the images that match the naming convention into that folder. There should only be 6.
    # # then return the list of lists of images that are in the subfolders.

    # # if the name of the

    # # foldername[0] --> '2022_11_5_15_44.jpg_panel_1.png'
    # # this is the kind of dirty names we have to clean.

    # # we want to get the first 4 numbers and the last number.
    # # we want to put the images into a folder named 2022_11_5_15
    # # we want to return a list of lists of images that are in the subfolders.

    # # we want to get the first 4 numbers and the last number.
    # # if type(foldername) == list:
    # #     foldername = foldername[0]
    # # else:
    # #     foldername = foldername
    # for file_name in tqdm(foldername):

    #     first_four = file_name.split('_')[0:4] # i.e. ['2022', '11', '5', '15']
    #     last_num = file_name.split('_')[-1] # i.e. '1.png'
    #     last_num = last_num.split('.')[0] # i.e. '1'


    #     # we want to put the images into a folder named 2022_11_5_15
    #     new_folder_name = '_'.join(first_four) # i.e. '2022_11_5_15'

    #     # the photos that belong with the new folder name (2022_11_5_15) contain the foldername's numbers.
    #     # Sort the images in the folder by the date they were added to the folder.
    #     #sorted_foldername = sorted(foldername, key=lambda x: os.path.getmtime(x)) # sort the images by the date they were added to the folder.

    #     # find all the files with the first 4 numbers (2022_11_5_15) and .jpg only once in the name
    #     files_with_last_four = [x for x in foldername if new_folder_name in x and 'jpg_panel' not in x]
    #     # find all the files with the last number (1)
    #     files_with_last_num = [x for x in files_with_last_four if str('panel_')+last_num in x]


    #     # are the images in the foldername in the correct order?
    #     # we want to check if the images are in the correct order.
    #     # They should now have the word panel 1, in the first image, panel 2 in the second image, etc. down to panel 6 and then the next image should be panel 1 again.
    #     # now we can just take each six images and put them into a folder.

    #     # logic
    #     # as long as the image contains "panel_n" where n is the next number in the order of the panels we add the image to the folder name that matches the first four numbers in their name.

    #      # go through the images in the foldername and put them into the correct folder.
    #     # we want to put the images into a folder named 2022_11_5_15 or whatever the first four numbers are.


    #     # make a folder in the panels folder with the name of the first four numbers.
    #     # if the folder doesn't exist, make it.
    #     print('new_folder_name',new_folder_name)
    #     try:
    #         os.mkdir(f'images/panels/{buoy_id}/batches/{new_folder_name}')
    #     except Exception as e:
    #         print(e)
    #         pass

    #     # go through the images in the foldername and put them into the correct folder.
    #     # we want to put the images into a folder named 2022_11_5_15 or whatever the first four numbers are.
    #     for image in files_with_last_num:
    #         # move the image into the directory f'images/panels/{buoy_id}/{new_folder_name}'
    #         # full path of the image
    #         full_path = f'images/panels/{buoy_id}/{image}'
    #         shutil.move(full_path, f'images/panels/{buoy_id}/batches/{new_folder_name}')
    #         pass



    #     # now we want to delete the images that are in the foldername but not in the subfolder.
    #     # we want to delete the images that are in the foldername but not in the subfolder.

    #     # get the list of images in the foldername
    #     images_in_foldername = [x for x in foldername if buoy_id in x]
    #     # get the list of images in the subfolder
    #     images_in_subfolder = [x for x in os.listdir(f'images/panels/{buoy_id}/batches/{new_folder_name}') if buoy_id in x]
    #     # get the list of images that are in the foldername but not in the subfolder
    #     #!images_to_delete = [x for x in images_in_foldername if x not in images_in_subfolder]
    #     # delete the images that are in the foldername but not in the subfolder
    #     #for image in images_to_delete:
    #     #    os.remove(f'images/panels/{buoy_id}/{image}')





    #     # img1 = os.path.join('images','panels',str(buoy_id),file_name) # the image we want to move
    #     # img2 = os.path.join('images','batches',str(buoy_id),new_folder_name) # the folder we want to move the image to.
    #     # # i.e. move the image from images/panels/41002/2022_11_5_15_44_panel_1.png to images/panels/41002/2022_11_5_15/2022_11_5_15_44_panel_1.png
    #     # shutil.copy(img1, img2) # copy the image to the folder.
    pass


# 51101,51000,51003,51002,51004 - Hawaii
# 46006 - Amazing Sunset
# 46089 - Tillamook Oregon.


cam_urls = buoy_links() # get the links to the cameras
stitch_switch = False # make false if you don't want to stitch the images.


while True:
    try:
        # turn on at 4 am CST and turn off at 11 pm CST
        if datetime.datetime.now().hour < 4 or datetime.datetime.now().hour > 22: # if it is before 4 am or after 11 pm
            # wait to turn on until 4 am CST
            # keep the computer awake
            print('The computer is sleeping')
            time.sleep(240) # sleep for 4 minutes
            continue
        # # if the time is between 4 am and 11 am pacific time, then your wait_period is 100 seconds
        # if datetime.datetime.now().hour >= 4 and datetime.datetime.now().hour < 11:
        #     wait_period = 100
        # # if the time is between 11 am and 11 pm pacific time, then your wait_period is 600 seconds
        # if datetime.datetime.now().hour >= 11 and datetime.datetime.now().hour < 13:
        #     wait_period = 600 # 10 minutes
        # wait for 15 minutes
        wait_period = 600 # 10 minutes
        start_time = datetime.datetime.now() # use this to calculate the next time to download images (every ten minutes)
        #!print('Starting the download loop at {}'.format(start_time))
        # print('I can still see things! Downloading images...')
        for cam_url in tqdm(cam_urls):
            # get the buoy id from the camera url
            buoy_id = re.search('station=(.*)', cam_url).group(1)
            # get the current time
            now = datetime.datetime.now()
            # create a directory for the buoy id if it doesn't already exist
            if not os.path.exists('images/buoys/{}'.format(buoy_id)):
                os.makedirs('images/buoys/{}'.format(buoy_id))
                ##logging.info("Created directory for buoy {}".format(buoy_id))

            # get the image
            ##logging.info("Checking buoy {}".format(buoy_id)) # log the buoy id
            if 'images/buoys/{}/{}_{}_{}_{}_{}.jpg'.format(buoy_id, now.year, now.month, now.day, now.hour, now.minute) not in os.listdir('images/buoys/{}'.format(buoy_id)): # if the image has not already been downloaded
                time.sleep(0.25) # wait 0.25 seconds to avoid getting blocked by the server
                img = requests.get(cam_url) # get the image
                # save the image
                with open('images/buoys/{}/{}_{}_{}_{}_{}.jpg'.format(buoy_id, now.year, now.month, now.day, now.hour, now.minute), 'wb+') as f:
                    f.write(img.content) # write the image to the file
                # check if the image is daytime or nighttime
                # ##logging.WARNING("Skipped night detection model for buoy {}".format(buoy_id))
                #if not is_it_daytime('images/buoys/{}/{}_{}_{}_{}_{}.jpg'.format(buoy_id, now.year, now.month, now.day, now.hour, now.minute)): # if it is nighttime
                    # then we will delete the image
                    #*print(f'Deleting image for buoy {buoy_id} because it is nighttime.')
                    #*os.remove('images/buoys/{}/{}_{}_{}_{}_{}.jpg'.format(buoy_id, now.year, now.month, now.day, now.hour, now.minute))
                #    pass
            else:
                pass # if the image already exists, don't download it again

        ##logging.INFO("Beginning to panel images (line 24)") #! at {}".format(datetime.datetime.now()))
        # Save the panels to the images/panels directory
        list_of_buoys = os.listdir('images/buoys') # get the list of buoy ids by their directory names
        for buoy_id in tqdm(list_of_buoys):
            # get the list of images for the buoy
            #print(f'Paneling images for buoy {buoy_id}')
            if buoy_id != '.DS_Store' and '.' not in buoy_id: # if the buoy id is not a hidden file
                images = os.listdir('images/buoys/{}'.format(buoy_id))
                # if the image has not already been used to create panels, create the panels and save them to the images/panels directory
                ##logging.info("Saving panels for buoy {}".format(buoy_id))
                for image in images:
                    # print(f'    Paneling image {image}')
                    # if the image is not None
                    if image == '.DS_Store' or image != 'None':
                        continue
                    # If the panels directory for the buoy doesn't exist, create it.
                    if not os.path.exists('images/panels/{}'.format(buoy_id)):
                        os.makedirs('images/panels/{}'.format(buoy_id))
                    if 'images/buoys/{}/{}'.format(buoy_id, image) in os.listdir('images/panels/{}'.format(buoy_id)):
                        print('This image has already been used to create panels.')
                        continue
                    if image == '.DS_Store' and buoy_id != '.DS_Store':
                        continue # skip the .DS_Store file
                    #print('Processing image: {}'.format(image))

                    # get the panels
                    panel_1, panel_2, panel_3, panel_4, panel_5, panel_6 = divide_into_panels(buoy_id, 'images/buoys/{}/{}'.format(buoy_id, image))
                    ##logging.info("Saved panels for buoy {}".format(buoy_id))
                    # print('Saving panels...')
                    # save the panels to the images/panels directory


        # Stage 4: save buoy_update_rates_dict to a csv file
        buoy_update_rates_dict_df = pd.DataFrame.from_dict(buoy_update_rates_dict, orient='index')
        buoy_update_rates_dict_df.to_csv('data/buoy_update_rates_dict.csv')

        # Stage 5: Remove any duplicate images in the images/buoys directory with DifPy



        # Remove duplicate images (preferably before paneling but for now after)
        for folder in os.listdir('images/buoys'):
            if folder == '.DS_Store':
                continue
            # get the list of images in the folder
            # sort the images by date
            # make folder_path variable from relative path
            folder_path = 'images/buoys/{}'.format(folder)
            search = dif(folder_path, similarity='high', show_output=False, show_progress=True, silent_del=True, delete=True)

        ignoring_panel_optimimal = True # note: this is a temporary fix to the problem of the panel images not being generated
        # final step: make sure that all the previous buoy images have been panelled and saved to the images/panels directory
        for folder in tqdm(os.listdir('images/buoys')):
            #print('Checking if all images have been panelled for buoy {}'.format(folder))
            try:
                if folder == '.DS_Store':
                    continue
                images = os.listdir('images/buoys/{}'.format(folder))
                # if the folder is not in the images/panels directory, then we need to panel the images
                # if not os.path.exists('images/panels/{}'.format(folder)):
                if ignoring_panel_optimimal:
                    #print('The images in the images/buoys/{} directory have not been panelled yet.'.format(folder))
                    # panelling the images
                    try:
                        os.mkdir('images/panels/{}'.format(folder))
                        print('made directory for buoy {}'.format(folder) + ' in images/panels')
                    except:
                        pass
                    batch_id = 1
                    for image in images:
                        # make a folder for the batch that has the same name as the image without the extension
                        try:
                            i_name = image[:-4]
                            directory_save_path = f'images/panels/{folder}/{i_name}' # make the directory path
                            os.mkdir(directory_save_path)
                        except FileExistsError:
                            pass
                        # get the panels
                        # if the folder is not empty skip it
                        if len(os.listdir(directory_save_path)) > 0:
                            continue
                        try:
                            if image == '.DS_Store':
                                continue
                            # get the panels and save them to directory_save_path
                            panel_1, panel_2, panel_3, panel_4, panel_5, panel_6 = divide_into_panels(folder, 'images/buoys/{}/{}'.format(folder, image))
                            # save the panels to the directory_save_path
                            # panel_1.save(f'{directory_save_path}/panel_1.jpg')
                            # panel_2.save(f'{directory_save_path}/panel_2.jpg')
                            # panel_3.save(f'{directory_save_path}/panel_3.jpg')
                            # panel_4.save(f'{directory_save_path}/panel_4.jpg')
                            # panel_5.save(f'{directory_save_path}/panel_5.jpg')
                            # panel_6.save(f'{directory_save_path}/panel_6.jpg')
                            ##logging.info('Saved panels for image: {}'.format(image))

                        except:
                            ##logging.ERROR('Error while saving panels for image: {}'.format(image))
                            # print('Could not create panels for image: {}'.format(image))
                            continue

                        #note: trying to add in the vincent code here
                        # stitch the images together
                        if stitch_switch:
                            files_to_stitch = [f'{directory_save_path}/panel_1.jpg', f'{directory_save_path}/panel_2.jpg', f'{directory_save_path}/panel_3.jpg', f'{directory_save_path}/panel_4.jpg', f'{directory_save_path}/panel_5.jpg', f'{directory_save_path}/panel_6.jpg'] # list of files to stitch

                            # Stitch the images together with OpenCV and save the stitched image to the panoramas directory
                            print('Stitching images...')
                            try:
                                ocean_stitching(files_to_stitch, f'images/panoramas/{folder}/{i_name}.jpg') # stitch the images together and save the stitched image to the panoramas directory
                            except Exception as f:
                                print(f)
                                print('Could not stitch images for image: {}'.format(image))
                            # > Overload resolution failed:
                            # >  - Can't parse 'images'. Sequence item with index 0 has a wrong type
                            # >  - Can't parse 'images'. Sequence item with index 0 has a wrong type
                            # >  - Stitcher.stitch() missing required argument 'masks' (pos 2)
                            # >  - Stitcher.stitch() missing required argument 'masks' (pos 2)
                            # fix: https://stackoverflow.com/questions/6380057/python-cv2-error-215-overload-resolution-failed











                        # try:
                        #     print('Stitching images for image set {}'.format(files_to_stitch))
                        #     panorama = vincent.make_panorama(images)
                        #     #remove the .jpg from the image name
                        #     image_name = image[:-4] # remove the .jpg from the image name

                        #     cv2.imwrite(f'images/panoramas/{folder}_{image_name}_panorama.png', panorama)
                        # except Exception as e:
                        #     print(e)
                        #     print('Could not stitch images together for buoy: {}'.format(folder))
                        #     continue


                        batch_id += 1
            except Exception as e:
                print(e)
                #print('Could not create panels for buoy: {}'.format(folder))
                #print('The images in the images/buoys/{} directory have not been panelled yet.'.format(folder))
                #print('line 139') # line 139
                continue
        # for each folder in the images/panels folder, stitch the images together and save them to the images/panoramas folder with the same name as the folder + panorama.png

        #//: the for loop below does not account for the fact that there are multiple captures with 6 frames per capture. This means that the images will be stitched together incorrectly. This is a problem that needs to be fixed. Find a way to select only the sets of 6 images that go together to stitch together.

        print('stage 5 complete')
        # Stage 6: Create the buoy dataframes
            # if it has been ten minutes since the last time the data was fetched, fetch the data again
        if time.time() - last_time_fetched > 600 or first_run:
            latest_data = get_latest_data() # get the latest data from the RSS feed (updates every 10 minutes)
            # save the master dataframe to a csv file
            run_date = time.strftime("%Y%m%d_%H%M%S")
            latest_data.to_csv(f'data/rss/rss_buoy_data_{run_date}.csv', index=False)
            print('Done with this run')
            time_last_fetched_rss = time.time() # get the time of the last fetch
        print('stage 6 complete')

        #* ======= show the last buoy image captured in this run
        try:
            display_last_image(list_of_buoys) # display the last image captured in this run
        except:
            pass
        #* ====== End Show Buoy Image Snippet

        # how much time has passed since the start of the loop?
        time_elapsed = datetime.datetime.now() - start_time
        # wait until the time elapsed is 15 minutes from the start of the loop
        print("Waiting for the remainder of the minutes...")
        # wait_period = 100 # was 900 (15 minutes)
        for i in tqdm(range(wait_period - time_elapsed.seconds)):
            time.sleep(1)
        iteration_counter += 1
    except Exception as e:
        print(e)
        print('Error occurred.')
        continue








    # # now we have all the images in the subfolders
    # # we want to return a list of lists of images that are in the subfolders

    # list_of_lists = []
    # for folder in os.listdir(f'images/batches/{buoy_id}'):
    #     try:
    #         # Get the list of images in the folder, and append it to the list of lists (these are the relative paths to the images)
    #         list_of_lists.append(os.listdir(f'images/batches/{buoy_id}/{folder}')) # this will be a list of 6 images
    #     except NotADirectoryError:
    #         # there is a file in the folder
    #         continue
    #     except FileNotFoundError:
    #         # there is a missing folder
    #         print(f'FileNotFoundError: {folder}')
    #         continue
    #     except Exception as e:
    #         print(e)
    #         continue
    # return list_of_lists # return a list of sets of images that have the same buoy id, year, month, day, hour, and panel numbers 1,2,3,4,5,6



# Next Steps

"""

The next steps are to:
Use the Keras model saved in the models folder to predict the content of each panel image as it is created, and save the predictions to a csv file.
These predictions will determine which buoys the model will follow closest and which buoys it will eventually ignore. This will be done by finding which of the buoy cameras record the most interesting content (i.e. the most interesting content will be the content that the model predicts as being the most interesting).





"""