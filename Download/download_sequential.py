"""
This python script implements sequential downloads.

@ Authors: Domenico Riccardi & Viola Floris

@ Creation Date: 09/04/2022

@ Last Update: 20/04/2022
"""
# import libraries
import time
import requests
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
def download(url, call):
    """
    The function makes a download request at the given URL.

    :param url: URL of file the user wants to download.
    :param call: The function call.

    :return: The downloaded file name.
    """
    response = requests.get(url, stream=True)
    file_name = url.split('/')[-1]
    print(f"The {file_name} download is assigned to the function call: {call}")
    with open(file_name, "wb") as file:
        for data in response.iter_content(1024):
            file.write(data)
    return file_name


@timer
def sequential():
    """
    This function calls the download function on files in the URLs list (one after the other).

    :return: The files list downloaded.
    """
    urls = [
        "https://root.cern/files/HiggsTauTauReduced/VBF_HToTauTau.root",
        "https://root.cern/files/HiggsTauTauReduced/GluGluToHToTauTau.root"
    ]
    files = []
    call = 0
    for url in urls:
        call += 1
        files.append(download(url, call))
    return files


if __name__ == "__main__":
    done = sequential()
    print(f"Downloaded files: {done}")
