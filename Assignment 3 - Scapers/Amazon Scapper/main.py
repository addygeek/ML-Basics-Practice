#amazon scapper
import requests
#import beautifulsoup as bs4
import pandas as pd

#define the request from amazon
amazon = "https://www.amazon.in/s?k="
product_name = input("Enter the product name : ")
querry = amazon+product_name
import requests
from bs4 import BeautifulSoup

def scrape_amazon_product(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }

    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract product details
    title = soup.find('span', {'id': 'productTitle'}).get_text().strip()
    price = soup.find('span', {'id': 'priceblock_ourprice'})
    price = price.get_text().strip() if price else 'Price not available'
    
    rating = soup.find('span', {'id': 'acrPopover'})
    rating = rating.get_text().strip() if rating else 'No ratings yet'
    
    description = soup.find('div', {'id': 'productDescription'})
    description = description.get_text().strip() if description else 'Description not available'
    
    # Display extracted information
    print(f"Product Title: {title}")
    print(f"Price: {price}")
    print(f"Rating: {rating}")
    print(f"Description: {description}")

# Example usage
url = "https://www.amazon.in/TCL-inches-Metallic-Bezel-Less-55V6B/dp/B0CZ6NXNP8/ref=sr_1_1_sspa?dib=eyJ2IjoiMSJ9.-t44WfsfD86Oj4zXFdoFwxPsPLWTBsaBNTvrXmuKpIwpBOweLfuRqgp-KpRjrufY8Z3u1K-fXRHcFzoCnwSyNnKduzMFjlp_uKgSeavLd-3DE2CxTbp1GC8OjEtncLgAgKTWw1K5OCZmEv9vafze-x3E4iMce7UKWyV3sBcyQR6c_4JMDY4DdRzBSjcI45inpuKC795MzfewiHa-ITJqcy3rHgE4buW6COlP4JXG9qw.AbN7hOmyL7DiabpR891PkFeVDGxVVc1BfUi5V1g0MYo&dib_tag=se&keywords=tv&qid=1738313703&sr=8-1-spons&sp_csd=d2lkZ2V0TmFtZT1zcF9hdGY&psc=1"  # Replace with any Amazon product URL
scrape_amazon_product(url)
