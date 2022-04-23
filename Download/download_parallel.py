"""
This python script implements parallel downloads.

@ Authors: Domenico Riccardi & Viola Floris

@ Creation Date: 09/04/2022

@ Last Update: 22/04/2022
"""
# import libraries
import os
import time
import requests
import multiprocessing
import functools


def timer(func):
    """
    Print the runtime of the function (decorator).

    :param func: The input function.

    :return: The return values of function.
    """

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(f"The '{func.__name__}' function runtime is: {run_time:.4f}s")
        return value

    return wrapper_timer


@timer
def download(url):
    """
    The function makes a download request at the given URL.

    :param url: URL of file the user wants to download
    :param call: The function call.

    :return: None
    """
    response = requests.get(url, stream=True)
    file_name = url.split('/')[-1]
    print(f"The {file_name} download is assigned to the process with ID: {os.getpid()}")
    with open(file_name, "wb") as file:
        for data in response.iter_content(1024):
            file.write(data)


def parallel():
    """
    Using multiprocessing module, the download is made by defining two or
    more different processes executed with the resources available.

    :return: The "Done" string.
    """
    urls = [
        "https://root.cern/files/HiggsTauTauReduced/VBF_HToTauTau.root",
        "https://root.cern/files/HiggsTauTauReduced/GluGluToHToTauTau.root"
    ]
    start_time = time.time()
    print(f"ID parent process: {os.getpid()}")
    with multiprocessing.Pool() as pool:
        pool.map(download, urls)
    run_time = time.time() - start_time
    print(f"Downloaded {len(urls)} in {run_time:.4f} seconds")
    return "Done"


if __name__ == "__main__":
    DONE = parallel()
    print(DONE)
