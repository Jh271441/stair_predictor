from selenium import webdriver
import time
import os
import requests
from PIL import Image

def fetch_image_urls(query, max_links_to_fetch, wd, sleep_between_interactions=1):
    def scroll_to_end(wd):
        wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(sleep_between_interactions)

    search_url = f"https://www.google.com/search?q={query}&tbm=isch"

    wd.get(search_url)
    image_urls = set()
    image_count = 0
    results_start = 0

    while image_count < max_links_to_fetch:
        scroll_to_end(wd)
        thumbnail_results = wd.find_elements("css selector", "img.Q4LuWd")
        number_results = len(thumbnail_results)

        for img in thumbnail_results[results_start:number_results]:
            try:
                img.click()
                time.sleep(sleep_between_interactions)
            except Exception:
                continue

            actual_images = wd.find_elements("css selector", 'img.n3VNCb')
            for actual_image in actual_images:
                if actual_image.get_attribute('src') and 'http' in actual_image.get_attribute('src'):
                    image_urls.add(actual_image.get_attribute('src'))

            image_count = len(image_urls)

            if len(image_urls) >= max_links_to_fetch:
                break
        else:
            load_more_button = wd.find_element("css selector", ".mye4qd")
            if load_more_button:
                wd.execute_script("document.querySelector('.mye4qd').click();")

        results_start = len(thumbnail_results)

    return image_urls

def download_image(folder_path, url, file_name):
    try:
        image_content = requests.get(url).content
        image_file = os.path.join(folder_path, file_name)
        with open(image_file, 'wb') as f:
            f.write(image_content)
        print(f"Downloaded {file_name} from {url}")
    except Exception as e:
        print(f"Failed to download {url} - {e}")

def main():
    query = "staircase"
    max_images = 50
    folder_path = "staircase_images"

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    wd = webdriver.Chrome()
    urls = fetch_image_urls(query, max_images, wd)
    wd.quit()

    for i, url in enumerate(urls):
        download_image(folder_path, url, f"staircase_{i+1}.jpg")

if __name__ == "__main__":
    main()