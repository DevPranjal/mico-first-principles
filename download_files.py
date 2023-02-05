import os
import urllib
import argparse

from torchvision.datasets.utils import download_and_extract_archive

parser = argparse.ArgumentParser()
parser.add_argument('--challenge', type=str, required=True, choices=['cifar10', 'purchase100'])

args = parser.parse_args()

CHALLENGE = args.challenge

url = "https://membershipinference.blob.core.windows.net/mico/cifar10.zip?si=cifar10&spr=https&sv=2021-06-08&sr=b&sig=d7lmXZ7SFF4ZWusbueK%2Bnssm%2BsskRXsovy2%2F5RBzylg%3D." if CHALLENGE == 'cifar10' else 'https://membershipinference.blob.core.windows.net/mico/purchase100.zip?si=purchase100&spr=https&sv=2021-06-08&sr=b&sig=YzJUTPoNndtIy0y2666XnPXS4WBF%2BbN7kbVM2soQNoU%3D'
filename = "cifar10.zip" if CHALLENGE == 'cifar10' else 'purchase100.zip'
md5 = "c615b172eb42aac01f3a0737540944b1" if CHALLENGE == 'cifar10' else '67eba1f88d112932fe722fef85fb95fd'

try:
    download_and_extract_archive(url=url, download_root=os.curdir, extract_root=None, filename=filename, md5=md5, remove_finished=False)
except urllib.error.HTTPError as e:
    print(e)
    print("Have you replaced the URL above with the one you got after registering?")
