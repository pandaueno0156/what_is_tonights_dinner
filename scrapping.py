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

### Uber eats Food Choice Project ###

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# To be able to print in utf-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("Hello World")

# Set up Chrome options for proper encoding
chrome_options = Options()
chrome_options.add_argument('--lang=ja_JP')
chrome_options.add_argument('--headless')  # Optional: run in headless mode
chrome_options.add_argument('--disable-gpu')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
chrome_options.add_argument('--charset=UTF-8')
chrome_options.add_argument('--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36')
chrome_options.add_argument('--window-size=1920,1080')

# ubereats url
# url = "https://www.ubereats.com/jp-en/neighborhood/ikebukuro-toshima-tokyo?srsltid=AfmBOoq53JdGTyBSsCf2qYmQATDOvWZ4iisqVwKVrBlPrS0uy2bp3ox2"
# url = "https://www.ubereats.com/jp-en/near-me"
url = "https://www.ubereats.com/jp/neighborhood/ikebukuro-toshima-tokyo?pl=JTdCJTIyYWRkcmVzcyUyMiUzQSUyMiVFMyU4MiVCNiVFMyU4MyVCQiVFMyU4MyVBOSVFMyU4MiVBNCVFMyU4MiVBQSVFMyU4MyVCMyVFMyU4MiVCQSVFNiVCMSVBMCVFOCVBMiU4QiUyMiUyQyUyMnJlZmVyZW5jZSUyMiUzQSUyMkNoSUoxZFFOSUdxVEdHQVJxZHRTcTdPenQ2RSUyMiUyQyUyMnJlZmVyZW5jZVR5cGUlMjIlM0ElMjJnb29nbGVfcGxhY2VzJTIyJTJDJTIybGF0aXR1ZGUlMjIlM0EzNS43MzU3ODQ4JTJDJTIybG9uZ2l0dWRlJTIyJTNBMTM5LjcwNjU5OTklN0Q%3D&slr=1&app_clip=false&campaign=signin_universal_link&effect=&guest_mode=false&marketing_vistor_id=9f92f82e-26c8-479c-98c7-b832176f401c&source_cta=undefined&source_flow=undefined"

driver = webdriver.Chrome(options=chrome_options)  # Make sure you have ChromeDriver installed

# Wait for the content to load
wait = WebDriverWait(driver, 10)
driver.get(url)


script = """
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
    
    // Address keywords to check (both Japanese and English)
    const addressKeywords = [
        // General Tokyo terms
        '東京都', '东京都', 'Tokyo', 'Tōkyō-To', 'Tōkyō-to',
        
        // 23 Special Wards in Japanese
        '千代田区', '中央区', '港区', '新宿区', '文京区', '台東区', '墨田区',
        '江東区', '品川区', '目黒区', '大田区', '世田谷区', '渋谷区', '中野区',
        '杉並区', '豊島区', '北区', '荒川区', '板橋区', '練馬区', '足立区',
        '葛飾区', '江戸川区',
        
        // 23 Special Wards in English
        'Chiyoda-ku', 'Chūō-ku', 'Minato-ku', 'Shinjuku-ku', 'Bunkyō-ku', 
        'Taitō-ku', 'Sumida-ku', 'Kōtō-ku', 'Shinagawa-ku', 'Meguro-ku',
        'Ōta-ku', 'Setagaya-ku', 'Shibuya-ku', 'Nakano-ku', 'Suginami-ku',
        'Toshima-ku', 'Kita-ku', 'Arakawa-ku', 'Itabashi-ku', 'Nerima-ku',
        'Adachi-ku', 'Katsushika-ku', 'Edogawa-ku',
        
        // Common alternative spellings without macrons
        'Chuo-ku', 'Bunkyo-ku', 'Taito-ku', 'Koto-ku', 'Ota-ku',
        
        // Keep existing specific area names
        '池袋', 'Ikebukuro', '高田馬場', '早稲田', '西早稲田', 
        '渋谷', 'Shibuya', '代代木', '代代木上原', '代代木神園', '代代木国立',
        '目白', '目白台', '目白駅', '目白駅前', '目白駅前通り',
        '中野', '中野駅', '中野駅前', '中野駅前通り',
        
        // Postal code prefix
        '171',
        // Japan
        '日本', 'Japan'
    ];
    
    // Function to check if text contains any address keywords
    const isAddress = (text) => {
        return addressKeywords.some(keyword => text.includes(keyword));
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

# print(f'last_height: {last_height}')

while True:
    # increase the height by 1000 if the difference is greater than 1000
    if last_height - initial_height > 1000:
        new_height = initial_height + 1000
    else:
        # if the difference is less than 1000, set the new height to the last height
        new_height = last_height

    driver.execute_script(f"window.scrollTo({initial_height}, {new_height});")

    time.sleep(0.5)
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
        # print(f'index: {index}')
        # print(row)
        mask = (row['restaurant_name'] == restaurant['name']) & (row['restaurant_address'] == restaurant['address'])
        # print()
        # print(f'mask: {mask}')
        if mask == True:
            found_restaurant_flag = True
            update_count += 1
            update_index = index

            db_df.at[index, 'restaurant_name'] = restaurant['name']
            db_df.at[index, 'restaurant_rating'] = float(restaurant['rating'])
            db_df.at[index, 'is_open'] = 1 # 1 means open if we can scrape the website information
            db_df.at[index, 'last_time_scraped'] = datetime.now()
            db_df.at[index, 'restaurant_address'] = restaurant['address']

            type_columns = [col for col in db_df.columns if col not in ['restaurant_name', 'restaurant_rating', 'is_open', 'last_time_scraped', 'restaurant_address', 'restaurant_img']]
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

        new_record = {
            'restaurant_name': restaurant['name'], 
            'restaurant_rating': restaurant['rating'],
            'last_time_scraped': datetime.now(),
            'is_open': 1,
            'restaurant_address': restaurant['address'],
            'restaurant_img': ''
        }

        for type_col in [col for col in db_df.columns if col not in ['restaurant_name', 'restaurant_rating', 'is_open', 'last_time_scraped', 'restaurant_address', 'restaurant_img']]:
            new_record[type_col] = 0

        for restaurant_type in restaurant['types']:
            new_record[restaurant_type] = 1

        # add the new record to the database
        db_df.loc[len(db_df)] = new_record

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
                # print(f"Image saved as: {image_filename}")
                # print("-" * 50)

                if found_restaurant_flag == True:
                    db_df.at[update_index, 'restaurant_img'] = image_filename
                else:
                    db_df.at[len(db_df)-1, 'restaurant_img'] = image_filename
            else:
                print(f"Failed to download image. Status code: {response.status_code}")
        except Exception as e:
            print(f"Error downloading image: {e}")
    else:
        # based on observation, the image url is always present.
        # if it is not present, scraping is not working.
        print("No image URL found")
        # print("-" * 50)


# read unique_types.csv and make a unique list of types
# currently have 439 types
# unique_types_used_feature.csv is sotred and used to create the original restaurant_database.csv
# unique_types.csv will be used to keep collecting types and make sure no unique types are missed
with open('unique_types.csv', 'r') as f:
    lines_list = f.readlines()
    unique_lines_list = sorted(set([line.strip() for line in lines_list]))

    f.close()

# # need to clean unique_types.csv to remove duplicates
with open('unique_types.csv', 'w', encoding='utf-8', newline='') as f:
    for type in unique_lines_list:
        f.write(f"{type}\n")
    f.close()

# save the updated database
db_df.to_csv('restaurant_database.csv', index=False)

logger.info(f"Updated {update_count} records and added {new_records_count} new records")




