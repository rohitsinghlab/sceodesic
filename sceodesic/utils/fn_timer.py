import time
from functools import wraps


def fn_timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):

        # run and time function
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"{func.__name__} took {elapsed_time:.3f} seconds to run.")
        return result
    return wrapper


