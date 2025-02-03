from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import csv

def youtube_scraper(video_url, max_comments=100):
    # Set up Selenium WebDriver with Chrome
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run without opening a browser
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--no-sandbox")

    # Set ChromeDriver Path (Make sure chromedriver.exe is available)
    chrome_driver_path = "chromedriver.exe"  # Update path if needed
    service = Service(chrome_driver_path)
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        # Load YouTube video page
        driver.get(video_url)
        time.sleep(5)  # Allow page to load

        # Extract Video Metadata
        wait = WebDriverWait(driver, 10)
        title = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "h1.title yt-formatted-string"))).text
        channel = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "yt-formatted-string.ytd-channel-name a"))).text
        views = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "span.view-count"))).text.split()[0]

        print(f"🎥 Video Title: {title}")
        print(f"📺 Channel: {channel}")
        print(f"👀 Views: {views}")

        # Scroll to the comments section
        driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
        time.sleep(3)

        comments = []
        last_height = driver.execute_script("return document.documentElement.scrollHeight")

        while len(comments) < max_comments:
            # Scroll down multiple times to load new comments
            driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
            time.sleep(3)

            # Wait until comments are loaded
            wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "ytd-comment-thread-renderer")))

            # Extract comments dynamically using Selenium
            comment_elements = driver.find_elements(By.CSS_SELECTOR, "ytd-comment-thread-renderer")

            for comment in comment_elements:
                try:
                    author = comment.find_element(By.CSS_SELECTOR, "#author-text").text.strip()
                    content = comment.find_element(By.CSS_SELECTOR, "#content-text").text.strip()
                    likes = comment.find_element(By.CSS_SELECTOR, "#vote-count-middle").text.strip() or "0"
                    time_posted = comment.find_element(By.CSS_SELECTOR, ".published-time-text").text.strip() or "Unknown"

                    if author and content:
                        comments.append({
                            "author": author,
                            "content": content,
                            "likes": likes,
                            "time_posted": time_posted
                        })
                except:
                    continue  # Skip if any element is missing

            print(f"🔄 Collected {len(comments)} comments...")

            # Stop scrolling if no new comments are loaded
            new_height = driver.execute_script("return document.documentElement.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

        # Save to CSV
        filename = "youtube_comments.csv"
        with open(filename, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Video Title", "Channel", "Views", "Author", "Comment", "Likes", "Time Posted"])

            for comment in comments[:max_comments]:  # Limit comments
                writer.writerow([title, channel, views, comment["author"], comment["content"], comment["likes"], comment["time_posted"]])

        print(f"✅ Data saved to '{filename}'")

    except Exception as e:
        print(f"❌ Error: {e}")

    finally:
        driver.quit()

# Usage Example:

youtube_scraper("https://youtu.be/kRR1K3q5nlg", max_comments=50)
