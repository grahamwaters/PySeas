import numpy as np
import cv2
import os
import imutils
from PIL import Image

class BuoyImage:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        self.average_color = self.get_average_color()

    def get_average_color(self):
        image = Image.fromarray(self.image)
        img_width, img_height = image.size
        average_color = image.getpixel((img_width // 2, img_height // 2))
        return average_color


class PanoramicImage:
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.panorama = self.make_panorama()

    def artist_eval(self):
        img = Image.fromarray(self.panorama)
        width, height = img.size
        panels = [img.getpixel((int(width * i / 12), int(height / 2))) for i in [1, 3, 6, 9, 10, 11]]
        mses = [np.mean((np.array(panels[i]) - np.array(panels[i + 1])) ** 2) for i in range(len(panels) - 1)]
        mse = np.mean(mses)

        return mse < 100

    def make_panorama(self):
        scale_percent = 70
        if '.DS_Store' in self.image_paths:
            self.image_paths.remove('.DS_Store')
        images_opened = [cv2.resize(cv2.imread(image), None, fx=scale_percent / 100, fy=scale_percent / 100, interpolation=cv2.INTER_AREA) for image in self.image_paths]
        stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
        stitcher.setPanoConfidenceThresh(0.1)
        stitcher.setRegistrationResol(0.6)
        stitcher.setSeamEstimationResol(0.1)
        stitcher.setCompositingResol(0.6)
        stitcher.setFeaturesFinder(cv2.ORB_create())
        stitcher.setFeaturesMatcher(cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING))
        stitcher.setBundleAdjuster(cv2.detail_BundleAdjusterRay())
        stitcher.setExposureCompensator(cv2.detail.ExposureCompensator_createDefault(cv2.detail.ExposureCompensator_GAIN_BLOCKS))
        stitcher.setBlender(cv2.detail.Blender_createDefault(cv2.detail.Blender_NO))
        stitcher.setSeamFinder(cv2.detail.SeamFinder_createDefault(cv2.detail.SeamFinder_VORONOI_SEAM))
        stitcher.setWarper(cv2.PyRotationWarper(cv2.ROTATION_WARP_PERSPECTIVE))
        stitcher.setInterpolationFlags(cv2.INTER_LINEAR_EXACT)
        status, stitched = stitcher.stitch(images_opened)

        if status == 0:
            stitched = cv2.resize(stitched, None, fx=scale_percent / 100, fy=scale_percent / 100, interpolation=cv2.INTER_AREA)
            return stitched
        else:
            return None


def main():
    image_directory = "images"
    image_paths = [os.path.join(image_directory, image) for image in os.listdir(image_directory) if image.endswith(('.jpg', '.jpeg', '.png'))]

    # Create BuoyImage instances
    buoy_images = [BuoyImage(image_path) for image_path in image_paths]

    # Create a PanoramicImage instance
    panoramic_image = PanoramicImage(image_paths)

    # Check if the panoramic_image is a consistent panorama
    is_panorama = panoramic_image.artist_eval()

    if is_panorama:
        print("The image is a consistent panorama.")
        # Save the panorama
        output_path = "images/panoramas/panorama.jpg"
        cv2.imwrite(output_path, panoramic_image.panorama)
    else:
        print("The image is not a consistent panorama.")

if __name__ == "__main__":
    main()
