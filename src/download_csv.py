# author: Trevor Kinsey (inspired by Tiffany Timbers)
# date: 2020-11-20

"""Downloads data from a url to a file specified by filepath.
Usage: download_data.py <url> <filepath> 
 
Options:
<url>               URL from where to download the data in csv format 
<filepath>          Path to desired local file location
"""

import os
import pandas as pd
from docopt import docopt

opt = docopt(__doc__)

def main(url, filepath):
    """Retrieves a dataset from URL and saves to the specified filepath

    Parameters
    ----------
    url : str
        URL to the earthquake data set
    file_path: str
        Path to save the earthquake data set

    Returns
    -------
    None
    """
    data = pd.read_csv(url)
    try:
        data.to_csv(filepath, index=False)
    except:
        os.makedirs(os.path.dirname(filepath))
        data.to_csv(filepath, index=False)


def test():
    """Retrieves a dataset from URL and saves to the specified filepath

    Parameters
    ----------

    Raises
    -------
    AssertionError: If any test does not pass
    """
    UNIT_TEST_URL = 'https://raw.githubusercontent.com/fivethirtyeight/data/master/san-andreas/earthquake_data.csv'
    UNIT_TEST_PATH ='unit_test'
    UNIT_TEST_FILEPATH = os.path.join(UNIT_TEST_PATH, 'unit_test.csv') 
    main(UNIT_TEST_URL, UNIT_TEST_FILEPATH)
    assert os.path.isfile(UNIT_TEST_FILEPATH), "File is not created at the specified path"
    source_data = pd.read_csv(UNIT_TEST_URL)
    saved_data = pd.read_csv(UNIT_TEST_FILEPATH)
    assert saved_data.shape == (1013, 11), "Saved data does not match the shape of the original data"
    # Delete the downloaded file
    os.remove(UNIT_TEST_FILEPATH)
    # Delete the unit test directory if no other files exist
    if len(os.listdir(UNIT_TEST_PATH)) == 0:
    	os.rmdir(UNIT_TEST_PATH)

if __name__ == "__main__":
    test()
    main(opt["<url>"], opt["<filepath>"])

