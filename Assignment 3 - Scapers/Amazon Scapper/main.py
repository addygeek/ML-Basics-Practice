#amazon scapper
import requests
#import beautifulsoup as bs4
import pandas as pd

#define the request from amazon
amazon = "https://www.amazon.in/"
product_name = "q=Mobile under 20000"
querry = amazon+product_name
print(querry)