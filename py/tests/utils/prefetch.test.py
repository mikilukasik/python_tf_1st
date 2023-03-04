import time
import unittest
import requests
import threading
from unittest.mock import patch
from utils import prefetch


class MyModuleTest(unittest.TestCase):

    def test_prefetch_data(self):
        url = 'http://www.example.com/1'
        prefetch.prefetch_data(url)
        self.assertIn(url, prefetch.in_progress_dict)
        time.sleep(1)
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
        with patch.object(requests, 'get', return_value=type('Response', (), {'content': data})):
            result = prefetch.get_data(url)
        self.assertEqual(result, data)

    def test_get_data_prefetch_in_progress(self):
        url = 'http://www.example.com/4'
        data = b'example data'

        def simulate_prefetch():
            time.sleep(1)
            prefetch.data_dict[url] = data
            del prefetch.in_progress_dict[url]
        prefetch.in_progress_dict[url] = True
        threading.Thread(target=simulate_prefetch).start()
        result = prefetch.get_data(url)
        self.assertEqual(result, data)
        self.assertNotIn(url, prefetch.in_progress_dict)
        self.assertNotIn(url, prefetch.data_dict)

    def test_get_data_prefetch_not_started(self):
        url = 'http://www.example.com/5'
        data = b'example data'
        with patch.object(requests, 'get', return_value=type('Response', (), {'content': data})):
            # This will start the prefetch process in the background
            prefetch.prefetch_data(url)
            # This should retrieve the prefetched data
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
        self.assertGreater(get_data_time, 30000)

    def test_with_no_name(self):
        url = 'http://www.example.com/7'
        prefetch.prefetch_data(url)
        time.sleep(1)

        start_time = time.monotonic()
        data = prefetch.get_data(url)
        get_data_time = time.monotonic() - start_time
        self.assertLess(get_data_time, 0.001)


if __name__ == '__main__':
    unittest.main()
