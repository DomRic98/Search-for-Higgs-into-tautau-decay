"""
This python script implements a download procedure with threads.

@ Authors: Domenico Riccardi & Viola Floris

@ Creation Date: 09/04/2022

@ Last Update: 20/04/2022
"""
# import libraries
import time
import threading
import requests
import functools

def timer(func):

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"The '{func.__name__}' function ({threading.current_thread().name}) start time is: {start_time:.4f}s, end time is: {end_time:.4f}")
        return value

    return wrapper_timer

@timer
def download(url):
    """
    The function makes a download request at the given URL.

    :param url: URL of file the user wants to download.
    :param call: The function call.

    :return: None.
    """
    response = requests.get(url, stream=True)
    file_name = url.split('/')[-1]
    print(f"The {file_name} download is assigned to the thread: {threading.current_thread().name}")
    with open(file_name, "wb") as file:
        for data in response.iter_content(1024):
            file.write(data)


def threads():
    """
    Using multithreading module, the download use more threads to each file.

    :return: Files downloaded in multithreading mode with printed runtime.
    """
    urls = [
        "https://root.cern/files/HiggsTauTauReduced/VBF_HToTauTau.root",
        "https://root.cern/files/HiggsTauTauReduced/GluGluToHToTauTau.root"
    ]
    NUM_THREAD = len(urls)
    start_time = time.time()
    # define a list of threads
    threads = [threading.Thread(target=download, args=(urls[x],)) for x in range(NUM_THREAD)]
    # start threads
    for thread in threads:
        thread.start()
    # join threads
    for thread in threads:
        thread.join()
    run_time = time.time() - start_time
    print(f"Downloaded {NUM_THREAD} in {run_time:.4f} seconds")
    return "Done"


if __name__ == "__main__":
    DONE = threads()
    print(DONE)
