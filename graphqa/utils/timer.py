from time import time


def timeit(method):
    def timed(*args, **kw):
        start_time = time.time()
        result = method(*args, **kw)
        end_time = time.time()
        time_delta = (end_time - start_time) * 1000
        print(f'{method.__name__} {time_delta:2.2f} ms')
        return result
    return timed