import requests
import os

def scrape_noaa_buoycams(image_directory):
    # URL of the buoycam image
    buoycam_url = "https://www.ndbc.noaa.gov/data/cameras/{buoycam_id}.jpg"

    # List of buoycam IDs
    buoycam_ids = ["44025", "44027", "44065"]  # Replace with the desired buoycam IDs

    # Create the image directory if it doesn't exist
    os.makedirs(image_directory, exist_ok=True)

    # Scrape images from each buoycam
    for buoycam_id in buoycam_ids:
        # Construct the URL for the buoycam image
        url = buoycam_url.format(buoycam_id=buoycam_id)

        # Send a GET request to retrieve the image data
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            # Save the image to the image directory
            image_path = os.path.join(image_directory, f"{buoycam_id}.jpg")
            with open(image_path, "wb") as f:
                f.write(response.content)
            print(f"Image saved: {image_path}")
        else:
            print(f"Failed to retrieve image from buoycam {buoycam_id}")

# Example usage
image_directory = "path/to/save/images"
scrape_noaa_buoycams(image_directory)