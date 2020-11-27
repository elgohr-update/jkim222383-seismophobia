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
    data = pd.read_csv(url, header=None)
    try:
        data.to_csv(filepath, index=False, header=None)
    except:
        os.makedirs(os.path.dirname(filepath))
        data.to_csv(filepath, index=False, header=None)


UNIT_TEST_URL = 'https://raw.githubusercontent.com/fivethirtyeight/data/master/san-andreas/earthquake_data.csv'
UNIT_TEST_PATH ='unit_test'
UNIT_TEST_FILEPATH = os.path.join(UNIT_TEST_PATH, 'unit_test.csv') 
main(UNIT_TEST_URL, UNIT_TEST_FILEPATH)
assert os.path.isfile(UNIT_TEST_FILEPATH)
source_data = pd.read_csv(UNIT_TEST_URL, header=None)
saved_data = pd.read_csv(UNIT_TEST_FILEPATH, header=None)
assert saved_data.shape[0] > 0
assert saved_data.shape[1] > 0
assert source_data.shape == saved_data.shape
os.remove(UNIT_TEST_FILEPATH)
if len(os.listdir(UNIT_TEST_PATH)) == 0:
    os.rmdir(UNIT_TEST_PATH)

if __name__ == "__main__":
    main(opt["<url>"], opt["<filepath>"])

