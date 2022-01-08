import scrapy
from collections import OrderedDict
import pandas as pd

class CrawlerSpider(scrapy.Spider):
    # spider name
    name = "imdb_crawler"
    # domains & URL
    allowed_domains = ["imdb.com"]

    def start_requests(self):
        # get movie id and imdb from the original dataset
        df = pd.read_csv('../data/movies.csv')
        for index,row in df.iterrows():
            try:
                # create imdb movie profile url based with imdb_id and send request to scrape rating
                url = 'https://www.imdb.com/title/' + row['imdb_id'] + '/'
                yield scrapy.Request(url = url,callback=self.parse,meta={'id' : row['id']})
            except:
                pass

    # parsing function
    def parse(self, response):
        item = OrderedDict()
        # movie id
        item['id'] = response.meta.get('id')
        try:
            # get rating score from website using xpath selector
            rating = response.xpath('//span[@class="AggregateRatingButton__RatingScore-sc-1ll29m0-1 iTLWoV"]/text()').extract_first()
            # round rating score
            item['rating'] = str(round(float(rating)))
        except:
            item['rating'] = ''
        yield item