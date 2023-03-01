import requests
import time


class DataLoader:
    def __init__(self, url, batch_size):
        self.url = url
        self.batch_size = batch_size
        self.data = []
        self.idx = 0

    def fetch_data(self):
        print('getting')
        response = requests.get(self.url)
        print('got')
        data = response.json()
        self.data.extend(data)
        return len(data)

    def get_batch(self):
        if not self.data:
            self.fetch_data()

        if self.idx >= len(self.data):
            self.idx = 0

        batch = self.data[self.idx:self.idx+self.batch_size]
        self.idx += self.batch_size
        return batch


if __name__ == '__main__':
    url = 'https://jsonplaceholder.typicode.com/posts'
    batch_size = 1
    data_loader = DataLoader(url, batch_size)

    while True:
        batch = data_loader.get_batch()
        print(f"Got batch: {batch}")
        time.sleep(5)
