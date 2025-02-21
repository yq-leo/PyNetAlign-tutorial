from typing import Union, Optional
from pathlib import Path
import os

from PyNetAlign.data import Dataset
from .utils import download_file_from_google_drive


class FoursquareTwitter(Dataset):
    r"""Foursquare-Twitter dataset for alignment."""
    def __init__(self,
                 root: Union[str, Path],
                 download: Optional[bool] = False,
                 train_ratio: Optional[float] = 0.2,
                 precision: Optional[int] = 32,
                 seed: Optional[int] = 0):

        if download:
            download_file_from_google_drive(
                remote_file_id='1pTxudta3-Y65LGl92J1--5H6iztkOEbE',
                save_filename='foursquare-twitter.pt',
                root=root)

        if not self._check_integrity(root):
            raise RuntimeError('Foursquare-Twitter dataset not found or corrupted. You can use download=True to download it')

        super(FoursquareTwitter, self).__init__(root=root, name='foursquare-twitter', train_ratio=train_ratio, precision=precision, seed=seed)

    def _check_integrity(self, root):
        return os.path.exists(os.path.join(root, 'foursquare-twitter.pt'))
