import requests
from bs4 import BeautifulSoup
import pandas as pd

# Set headers to mimic a real browser
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

# Function to scrape Amazon product reviews
def scrape_amazon_reviews(url):
    try:
        # Send a GET request to the product page
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an error for bad status codes

        # Parse the page with BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract product details
        product_title = soup.select_one("#productTitle").text.strip() if soup.select_one("#productTitle") else "N/A"
        product_price = soup.select_one(".a-price .a-offscreen").text.strip() if soup.select_one(".a-price .a-offscreen") else "N/A"
        product_rating = soup.select_one(".a-icon-star .a-icon-alt").text.strip() if soup.select_one(".a-icon-star .a-icon-alt") else "N/A"
        product_reviews_count = soup.select_one("#acrCustomerReviewText").text.strip() if soup.select_one("#acrCustomerReviewText") else "N/A"

        # Extract reviews
        reviews = []
        review_elements = soup.select(".review")
        for review in review_elements:
            review_data = {
                "review_title": review.select_one(".review-title").text.strip() if review.select_one(".review-title") else "N/A",
                "review_text": review.select_one(".review-text").text.strip() if review.select_one(".review-text") else "N/A",
                "rating": review.select_one(".a-icon-alt").text.strip() if review.select_one(".a-icon-alt") else "N/A",
                "date": review.select_one(".review-date").text.strip() if review.select_one(".review-date") else "N/A",
                "helpful_votes": review.select_one(".cr-vote-text").text.strip() if review.select_one(".cr-vote-text") else "0",
            }
            reviews.append(review_data)

        return {
            "product_title": product_title,
            "product_price": product_price,
            "product_rating": product_rating,
            "product_reviews_count": product_reviews_count,
            "reviews": reviews,
        }
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None

# Example Amazon product URL
url = "https://www.amazon.in/ZdalaMit-Mi-Tv-Remote-Suitable/dp/B0CBV6QNMF?ref_=Oct_d_obs_d_1388929031_1&pd_rd_w=cHXzZ&content-id=amzn1.sym.a0183515-a55a-48ac-a863-406c0a598721&pf_rd_p=a0183515-a55a-48ac-a863-406c0a598721&pf_rd_r=FQ0N35GFNK74W5G0RKE1&pd_rd_wg=Omhqg&pd_rd_r=357a4165-dfa4-4075-bf88-df5cc38b3708&pd_rd_i=B0CBV6QNMF&th=1"  # Replace with the actual product URL

# Scrape reviews
print(f"Scraping {url}...")
product_data = scrape_amazon_reviews(url)

# Save data to a CSV file
if product_data:
    # Save product details
    product_details = {
        "Product Title": [product_data["product_title"]],
        "Price": [product_data["product_price"]],
        "Rating": [product_data["product_rating"]],
        "Total Reviews": [product_data["product_reviews_count"]],
    }
    df_product = pd.DataFrame(product_details)
    df_product.to_csv("amazon_product_details.csv", index=False)
    print("Product details saved to amazon_product_details.csv")

    # Save reviews
    if product_data["reviews"]:
        df_reviews = pd.DataFrame(product_data["reviews"])
        df_reviews.to_csv("amazon_product_reviews.csv", index=False)
        print("Reviews saved to amazon_product_reviews.csv")
    else:
        print("No reviews found.")
else:
    print("No data scraped.")