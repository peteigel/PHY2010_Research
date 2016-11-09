import time

class Timer ():
    def __enter__(self):
        self.start_time = time.perf_counter()

    def __exit__(self, *args):
        print('Execution Time: {}s'.format(time.perf_counter() - self.start_time))
