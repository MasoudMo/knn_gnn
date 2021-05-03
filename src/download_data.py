"""
Author: Masoud Mokhtari

Downloads the PTB Diagnostic ECG Database found at https://www.physionet.org/content/ptbdb/1.0.0/

"""

import argparse
import wfdb


def main():
    # Command line argument parser
    parser = argparse.ArgumentParser(description='Download PTB Diagnostic Database')
    parser.add_argument('--download_path',
                        type=str,
                        required=False,
                        default="../data/",
                        help='Path to download the dataset to.')
    args = parser.parse_args()

    download_path = args.download_path

    print("Downloading the PTB Diagnostic Database into {}".format(download_path))
    wfdb.dl_database('ptbdb', download_path)
    print("Dataset successfully downloaded!")


if __name__ == "__main__":
    main()