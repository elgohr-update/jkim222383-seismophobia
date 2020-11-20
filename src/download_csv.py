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
        data.to_csv(filepath, index=False)
    except:
        os.makedirs(os.path.dirname(filepath))
        data.to_csv(filepath, index=False)



if __name__ == "__main__":
    main(opt["<url>"], opt["<filepath>"])


# TODO: test case