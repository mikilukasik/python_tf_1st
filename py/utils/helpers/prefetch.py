import threading
import logging

data_dict = {}
in_progress_dict = {}
data_getter = None

data_fetched_events = {}


def set_data_getter(getter_func):
    global data_getter
    data_getter = getter_func


def prefetch_data(url='default'):
    def fetch():
        response = data_getter(url)
        data_dict[url] = response
        del in_progress_dict[url]
        if url in data_fetched_events:
            data_fetched_events[url].set()
        logging.debug(f"Fetched data for URL: {url}")
    in_progress_dict[url] = True
    thread = threading.Thread(target=fetch)
    thread.start()
    logging.debug(
        f"Started prefetch for URL: {url}. in_progress_dict: {in_progress_dict}")


def get_data(url='default'):
    if url in data_dict:
        data = data_dict[url]
        del data_dict[url]
        logging.debug(
            f"Got prefetched data for URL: {url}. data_dict: {data_dict}")
        return data
    elif url in in_progress_dict:
        data_fetched_event = threading.Event()
        data_fetched_events[url] = data_fetched_event
        logging.debug(
            f"Event created for URL: {url}. data_fetched_events: {data_fetched_events}")
        data_fetched_event.wait()
        data = data_dict[url]
        del data_dict[url]
        del data_fetched_events[url]
        logging.debug(
            f"Got data after waiting for URL: {url}. data_dict: {data_dict}")
        return data
    else:
        logging.debug(f"Calling prefetch from get_data: {url}")
        prefetch_data(url)
        logging.debug(
            f"Finished prefetch in get_data: {url}. data_dict: {data_dict}")
        return get_data(url)
