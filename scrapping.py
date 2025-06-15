from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import time
import sys
import io
import os
import requests
import pandas as pd
import logging
from datetime import datetime

### What is tonight's dinner Project ###

# UberEats URL Configuration
UBEREATS_URLS = {
    'ikebukuro': "https://www.ubereats.com/jp/neighborhood/ikebukuro-toshima-tokyo?pl=JTdCJTIyYWRkcmVzcyUyMiUzQSUyMiVFMyU4MiVCNiVFMyU4MyVCQiVFMyU4MyVBOSVFMyU4MiVBNCVFMyU4MiVBQSVFMyU4MyVCMyVFMyU4MiVCQSVFNiVCMSVBMCVFOCVBMiU4QiUyMiUyQyUyMnJlZmVyZW5jZSUyMiUzQSUyMkNoSUoxZFFOSUdxVEdHQVJxZHRTcTdPenQ2RSUyMiUyQyUyMnJlZmVyZW5jZVR5cGUlMjIlM0ElMjJnb29nbGVfcGxhY2VzJTIyJTJDJTIybGF0aXR1ZGUlMjIlM0EzNS43MzU3ODQ4JTJDJTIybG9uZ2l0dWRlJTIyJTNBMTM5LjcwNjU5OTklN0Q%3D&slr=1&app_clip=false&campaign=signin_universal_link&effect=&guest_mode=false&marketing_vistor_id=9f92f82e-26c8-479c-98c7-b832176f401c&source_cta=undefined&source_flow=undefined",
    'near_me': "https://www.ubereats.com/jp-en/near-me"
}

# Chrome options configuration
CHROME_OPTIONS = {
    'lang': 'ja_JP',
    'headless': True,
    'disable-gpu': True,
    'no-sandbox': True,
    'disable-dev-shm-usage': True,
    'charset': 'UTF-8',
    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'window-size': '1920,1080'
}


def scrape_restaurant_data(location='ikebukuro'):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # To be able to print in utf-8
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    # Set up Chrome options for proper encoding
    chrome_options = Options()
    for option, value in CHROME_OPTIONS.items():
        chrome_options.add_argument(f'--{option}={value}' if value is not True else f'--{option}')

    # Get UberEats URL based on location
    url = UBEREATS_URLS.get(location)
    if not url:
        logger.error(f"Invalid location: {location}")
        sys.exit(1)

    driver = webdriver.Chrome(options=chrome_options)  # Make sure the ChromeDriver is installed

    # Wait for the content to load
    wait = WebDriverWait(driver, 10)
    driver.get(url)


    script = r"""
    return Array.from(document.querySelectorAll('[data-test="store-link"]')).map(storeLink => {
        // Find the title
        const titleElement = storeLink.querySelector('[data-test="store-title"]');
        
        // Get rating
        const ratingElement = storeLink.querySelector('div[aria-hidden="true"]');

        // Get image URL from source tag
        const sourceElement = storeLink.querySelector('source[type="image/webp"]');
        const imageUrl = sourceElement?.getAttribute('srcset') || '';
        
        // Get all spans with the specific class that contains types and address
        const targetSpans = Array.from(storeLink.querySelectorAll('span'));
        
        // Function to check if text contains numeric digits
        const isAddress = (text) => {
            // Check if the text contains any digits (0-9)
            return /\d/.test(text);
        };
        
        // Separate types and address
        const types = targetSpans
            .map(span => span.textContent)
            .filter(text => !isAddress(text) && text.trim() !== '•' && text.trim() !== '・' && text.trim() !== '');
            
        const address = targetSpans
            .find(span => isAddress(span.textContent))?.textContent || '';
        
        return {
            name: titleElement?.textContent || '',
            rating: ratingElement?.textContent || 0.0,
            types: types,
            address: address,
            imageUrl: imageUrl
        }
    });
    """

    # Wait for store-link elements to be present
    wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, '[data-test="store-link"]')))

    # Scroll to bottom of page gradually until no new content loads
    last_height = driver.execute_script("return document.body.scrollHeight")

    initial_height = 1000
    driver.execute_script(f"window.scrollTo(0, {initial_height});")

    while True:
        # increase the height by 1000 if the difference is greater than 1000
        if last_height - initial_height > 1000:
            new_height = initial_height + 1000
        else:
            # if the difference is less than 1000, set the new height to the last height
            new_height = last_height

        driver.execute_script(f"window.scrollTo({initial_height}, {new_height});")

        time.sleep(1)
        # update the initial height
        initial_height = new_height

        # Break the loop if no new content loaded (heights equal)
        if new_height == last_height:
            break


    # read restaurant_database.csv to see if the restaurant is already in the database
    try:
        db_df = pd.read_csv('restaurant_database.csv')
        logger.info("Loaded restaurant_database.csv")
    except FileNotFoundError:
        logger.error("restaurant_database.csv not found")
        #interrupt the program
        sys.exit()

    # Create a directory to store images if it doesn't exist
    if not os.path.exists('restaurant_images'):
        os.makedirs('restaurant_images')

    # keep track of the number of updates and new records
    update_count = 0
    new_records_count = 0

    restaurants_data = driver.execute_script(script)
    for idx, restaurant in enumerate(restaurants_data):
        # Check if the information is scraped correctly
        # print(f"\nRestaurant: {restaurant['name']}")
        # print("Rating:", restaurant['rating'])
        # print("Types:", ' | '.join(restaurant['types']))
        # print("Address:", restaurant['address'])

        found_restaurant_flag = False
        update_index = 99999
        for index, row in db_df.iterrows():

            mask = (row['restaurant_name'] == restaurant['name']) & (row['restaurant_address'] == restaurant['address'])

            if mask == True:
                found_restaurant_flag = True
                update_count += 1
                update_index = index
                
                # To update the rating, open_status, last_time_scrappd, restaurant_types
                db_df.at[index, 'restaurant_rating'] = float(restaurant['rating'])
                db_df.at[index, 'is_open'] = 1 # 1 means open if we can scrape the website information
                db_df.at[index, 'last_time_scraped'] = datetime.now()
                db_df.at[index, 'restaurant_types'] = restaurant['types']

                type_columns = [col for col in db_df.columns if col not in ['restaurant_name', 'restaurant_rating', 'is_open', 'last_time_scraped', 'restaurant_address', 'restaurant_img', 'restaurant_types']]
                for type in restaurant['types']:
                    for feature in type_columns:
                        if type == feature:
                            db_df.at[index, feature] = 1
                        else:
                            # if not in the type the rest should be 0
                            db_df.at[index, feature] = 0

                # logger.info(f"Updated restaurant: {restaurant['name']}")
                break

        if found_restaurant_flag == False:
            # logger.info(f"Restaurant not found in the database: {restaurant['name']}")
            
            new_records_count += 1

            # To create a new record as we cannot find the restaurant in the database
            new_record = {
                'restaurant_name': restaurant['name'], 
                'restaurant_rating': restaurant['rating'],
                'last_time_scraped': datetime.now(),
                'is_open': 1,
                'restaurant_address': restaurant['address'],
                'restaurant_img': '',
                'restaurant_types': restaurant['types']
            }

            for type_col in [col for col in db_df.columns if col not in ['restaurant_name', 'restaurant_rating', 'is_open', 'last_time_scraped', 'restaurant_address', 'restaurant_img', 'restaurant_types']]:
                new_record[type_col] = 0

            for restaurant_type in restaurant['types']:
                new_record[restaurant_type] = 1

            # add the new record to the database
            # Convert new_record to Series with matching index
            new_record_series = pd.Series(new_record, index=db_df.columns)
            db_df.loc[len(db_df)] = new_record_series

            # logger.info(f'added new record: {restaurant["name"]} to the database')

        # store restaurant types in csv. This is to make sure no unique types are missed
        with open('unique_types.csv', 'a') as f:
            for type in restaurant['types']:
                f.write(f"{type}\n")
            f.close()

        # Download and save the image
        if restaurant['imageUrl']:
            try:
                # Create a safe filename from the restaurant name
                safe_name = ''.join(c for c in restaurant['name'] if c.isalnum() or c in (' ', '-', '_')).rstrip()
                image_filename = f"restaurant_images/{safe_name}.webp"  # Changed extension to .webp
                
                # Download the image
                response = requests.get(restaurant['imageUrl'])
                if response.status_code == 200:
                    with open(image_filename, 'wb') as f:
                        f.write(response.content)
                    # Check if the image is saved
                    if found_restaurant_flag == True:
                        db_df.at[update_index, 'restaurant_img'] = image_filename
                    else:
                        db_df.at[len(db_df)-1, 'restaurant_img'] = image_filename
                else:
                    print(f"Failed to download image. Status code: {response.status_code}")
            except Exception as e:
                print(f"Error downloading image: {e}")
        else:
            # Based on observation, the image url should be always present.
            # if it is not present, scraping is not workin properly.
            print("No image URL found")


    # read unique_types.csv and make a unique list of types
    # currently have 439 types
    # unique_types_used_feature.csv is sotred and used to create the original restaurant_database.csv
    # unique_types.csv will be used to keep collecting types and make sure no unique types are missed
    with open('unique_types.csv', 'r') as f:
        lines_list = f.readlines()
        unique_lines_list = sorted(set([line.strip() for line in lines_list]))

        f.close()

    # clean unique_types.csv to remove duplicates
    with open('unique_types.csv', 'w', encoding='utf-8', newline='') as f:
        for type in unique_lines_list:
            f.write(f"{type}\n")
        f.close()

    # save the updated database
    db_df.to_csv('restaurant_database.csv', index=False)

    logger.info(f"Updated {update_count} records and added {new_records_count} new records")

    logger.info(f"Scraping completed")

if __name__ == "__main__":
    
    # Change the location parameter to scrape different areas
    scrape_restaurant_data(location='ikebukuro')


