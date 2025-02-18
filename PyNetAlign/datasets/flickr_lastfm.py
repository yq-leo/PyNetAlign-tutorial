from typing import Union, Optional
from pathlib import Path
import os

from PyNetAlign.data import Dataset
from .utils import download_file_from_google_drive


class FlickrLastFM(Dataset):
    r"""Flickr-LastFM dataset for alignment."""
    def __init__(self,
                 root: Union[str, Path],
                 download: Optional[bool] = False,
                 ratio: Optional[float] = 0.2,
                 precision: Optional[int] = 32,
                 seed: Optional[int] = None):

        if download:
            download_file_from_google_drive(
                remote_file_id='1NVQTs9a81r3SHQnto5FaV8Pu24efrbGK',
                save_filename='flickr-lastfm.pt',
                root=root)

        if not self._check_integrity(root):
            raise RuntimeError('Flickr-LastFM dataset not found or corrupted. You can use download=True to download it')

        super(FlickrLastFM, self).__init__(root=root, name='flickr-lastfm', ratio=ratio, precision=precision, seed=seed)

    def _check_integrity(self, root):
        return os.path.exists(os.path.join(root, 'flickr-lastfm.pt'))
