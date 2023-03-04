import time
import unittest
import requests
import threading
from unittest.mock import patch
from utils import prefetch
import logging


def default_data_getter(url):
    time.sleep(0.2)
    return None


class MyModuleTest(unittest.TestCase):

    def test_prefetch_data(self):
        url = 'http://www.example.com/1'
        prefetch.set_data_getter(default_data_getter)
        prefetch.prefetch_data(url)
        self.assertIn(url, prefetch.in_progress_dict)
        time.sleep(0.4)
        self.assertNotIn(url, prefetch.in_progress_dict)
        self.assertIn(url, prefetch.data_dict)

    def test_get_data_prefetched(self):
        url = 'http://www.example.com/2'
        data = b'example data'
        prefetch.data_dict[url] = data
        result = prefetch.get_data(url)
        self.assertEqual(result, data)
        self.assertNotIn(url, prefetch.data_dict)

    def test_get_data_not_prefetched(self):
        url = 'http://www.example.com/3'
        data = b'example data'

        def data_getter(queryUrl):
            self.assertEqual(queryUrl, url)
            time.sleep(0.2)
            return data
        prefetch.set_data_getter(data_getter)

        result = prefetch.get_data(url)
        self.assertEqual(result, data)

    def test_get_data_prefetch_in_progress(self):
        url = 'http://www.example.com/4'
        data = b'example data'

        def data_getter(queryUrl):
            self.assertEqual(queryUrl, url)
            time.sleep(0.2)
            return data

        prefetch.set_data_getter(data_getter)

        def simulate_prefetch():
            prefetch.prefetch_data(url)

        # Start the prefetch process in a separate thread
        threading.Thread(target=simulate_prefetch).start()

        # Wait a little bit for the prefetch process to start
        time.sleep(0.1)

        # Try to retrieve the data while it's still being prefetched
        result = prefetch.get_data(url)

        # Check that the retrieved data matches the expected data
        self.assertEqual(result, data)

        # Check that the URL is not in the in-progress or data dictionary
        self.assertNotIn(url, prefetch.in_progress_dict)
        self.assertNotIn(url, prefetch.data_dict)

    def test_get_data_prefetch_not_started(self):
        url = 'http://www.example.com/5'
        data = b'example data'

        def data_getter(queryUrl):
            self.assertEqual(queryUrl, url)
            time.sleep(0.2)
            return data
        prefetch.set_data_getter(data_getter)

        prefetch.prefetch_data(url)
        result = prefetch.get_data(url)

        self.assertEqual(result, data)
        self.assertNotIn(url, prefetch.in_progress_dict)
        self.assertNotIn(url, prefetch.data_dict)

    def test_prefetch_does_not_block_thread(self):
        url = 'http://www.example.com/6'
        start_time = time.monotonic()
        prefetch.prefetch_data(url)
        prefetch_time = time.monotonic() - start_time

        data = prefetch.get_data(url)
        get_data_time = time.monotonic() - prefetch_time
        self.assertLess(prefetch_time, 0.001)
        self.assertGreater(get_data_time, 0.2)

    def test_get_data_after_prefetch(self):
        url = 'http://www.example.com/7'
        data = b'example data'

        def data_getter(queryUrl):
            self.assertEqual(queryUrl, url)
            time.sleep(0.2)
            return data
        prefetch.set_data_getter(data_getter)

        prefetch.prefetch_data(url)
        time.sleep(0.1)

        result = prefetch.get_data(url)
        self.assertEqual(result, data)
        self.assertNotIn(url, prefetch.in_progress_dict)
        self.assertNotIn(url, prefetch.data_dict)


if __name__ == '__main__':
    unittest.main()
