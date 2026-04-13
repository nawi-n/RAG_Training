from functools import wraps

from tenacity import retry, stop_after_attempt, wait_exponential


def retry_with_exception(fn):
    @wraps(fn)
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def wrapper(state):
        try:
            return fn(state)
        except Exception as e:
            state["error"] = str(e)
            raise

    return wrapper
