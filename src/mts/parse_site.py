import argparse
import os
import signal
import time
import urllib.parse
from typing import Optional, Any

import orjson
import pandas as pd
import undetected_chromedriver as uc
from pyvirtualdisplay import Display
from selenium.common import NoSuchElementException, TimeoutException, InvalidArgumentException, \
    UnexpectedAlertPresentException, WebDriverException
from selenium.webdriver.common.by import By
from tqdm import tqdm

driver: Optional[uc.Chrome] = None
display: Optional[Display] = None


def is_blocked() -> Optional[Any]:
    global driver
    curr_url = driver.current_url
    blocked_url = urllib.parse.urlparse(curr_url)
    if blocked_url.netloc == 'blocked.mgts.ru':
        return {'url': curr_url, 'reason': 'mgts'}
    return None


def create_driver():
    global driver
    options = uc.ChromeOptions()
    options.add_argument("--lang=ru")
    options.add_argument("--window-size=1500,1500")
    options.add_argument('--load-extension=data/uBlock0.chromium')
    options.experimental_options['prefs'] = {'intl.accept_languages': 'ru-RU,ru', 'safebrowsing.enabled': False}
    options.unhandled_prompt_behavior = 'accept'
    options.add_argument('--ignore-certificate-errors')
    options.add_argument('--ignore-urlfetcher-cert-requests')
    driver = uc.Chrome(options, driver_executable_path='data/undetected_chromedriver')
    driver.set_page_load_timeout(60)


def startup():
    global driver, display
    display = Display(visible=False, size=(1500, 1500))
    display.start()
    create_driver()
    os.makedirs('data/site_retry', exist_ok=True)
    # signal.signal(signal.SIGINT, lambda *args: shutdown())
    # signal.signal(signal.SIGTERM, lambda *args: shutdown())


def work(domain: str, debug: bool):
    global driver, display
    if debug:
        print(domain)
    site_dir = f'data/site_data/{domain}'
    os.makedirs(site_dir, exist_ok=True)
    try:
        driver.get(f'https://{domain}/')
    except TimeoutException:
        with open(f'data/site_retry/{domain}.json', 'wb') as f:
            f.write(orjson.dumps({'url': None, 'reason': 'timeout'}))
    except BaseException as e:
        with open(f'data/site_retry/{domain}.json', 'wb') as f:
            f.write(orjson.dumps({'url': None, 'reason': 'exception', 'exc_data': str(e)}))
    time.sleep(1)
    try:
        try:
            block_status = is_blocked()
        except UnexpectedAlertPresentException:
            driver.switch_to.alert.accept()
            block_status = is_blocked()
    except WebDriverException:
        with open(f'data/site_retry/{domain}.json', 'wb') as f:
            f.write(orjson.dumps({'url': None, 'reason': 'renderer'}))
        driver.quit()
        del driver
        create_driver()
    else:
        if block_status:
            with open(f'data/site_retry/{domain}.json', 'wb') as f:
                f.write(orjson.dumps(block_status))
        else:
            try:
                driver.save_screenshot(f'{site_dir}/screenshot.png')
            except BaseException as e:
                with open(f'data/site_retry/{domain}.json', 'wb') as f:
                    f.write(orjson.dumps({'url': None, 'reason': 'exception_screenshot', 'exc_data': str(e)}))
                try:
                    driver.quit()
                    del driver
                    create_driver()
                except:
                    pass
            else:
                title = driver.title
                try:
                    description = driver.find_element(By.XPATH, "//meta[@name='description']").get_attribute("content")
                except (NoSuchElementException, InvalidArgumentException):
                    description = None
                html = driver.page_source
                entry = {'title': title, 'description': description, 'html': html}
                with open(f'{site_dir}/data.json', 'wb') as f:
                    f.write(orjson.dumps(entry))
    return domain
