import threading
import requests

data_dict = {}
in_progress_dict = {}


def prefetch_data(url):
    def fetch():
        response = requests.get(url)
        data_dict[url] = response.content
        del in_progress_dict[url]
    in_progress_dict[url] = True
    thread = threading.Thread(target=fetch)
    thread.start()


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
