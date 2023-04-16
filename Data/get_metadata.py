from selenium.webdriver.chrome.options import Options
from selenium import webdriver
from tqdm import tqdm
import pandas as pd


def get_metadata():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    driver = webdriver.Chrome(executable_path='chromedriver.exe', options=chrome_options)
    stock_list = sorted(list(pd.read_csv('stock_tickers.csv', header=None).iloc[0]))
    metadata = pd.DataFrame(columns=stock_list)

    for stock in tqdm(stock_list):
        try:
            driver.get('https://www.marketwatch.com/investing/stock/' + stock + '/company-profile')
            column = [driver.find_element('xpath', '//*[@id="maincontent"]/div[6]/div[2]/div[1]/p').text,
                      driver.find_element('xpath', '//*[@id="maincontent"]/div[6]/div[1]/div[1]/div/ul/li[1]/span').text,
                      driver.find_element('xpath', '//*[@id="maincontent"]/div[6]/div[1]/div[1]/div/ul/li[2]/span').text]
            metadata[stock] = column

        except:
            continue

    metadata.index = ['Description', 'Industry', 'Sector']
    return metadata


if __name__ == '__main__':
    get_metadata().to_csv('t2k_metadata.csv')
