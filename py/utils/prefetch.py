import threading
import requests
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

data_dict = {}
in_progress_dict = {}

data_getter = requests.get


def set_data_getter(getter_func):
    global data_getter
    data_getter = getter_func


def prefetch_data(url):
    def fetch():
        response = data_getter(url)
        data_dict[url] = response
        del in_progress_dict[url]
        logging.debug(f"Fetched data for URL: {url}")
    in_progress_dict[url] = True
    thread = threading.Thread(target=fetch)
    thread.start()
    logging.debug(f"Started prefetch for URL: {url}")


def get_data(url):
    if url in data_dict:
        data = data_dict[url]
        del data_dict[url]
        return data
    elif url in in_progress_dict:
        while url in in_progress_dict:
            pass
        data = data_dict[url]
        del data_dict[url]
        return data
    else:
        prefetch_data(url)
        while url in in_progress_dict:
            pass
        data = data_dict[url]
        del data_dict[url]
        return data
