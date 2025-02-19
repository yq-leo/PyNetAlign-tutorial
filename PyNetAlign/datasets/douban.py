from typing import Union, Optional
from pathlib import Path
import os

from PyNetAlign.data import Dataset
from .utils import download_file_from_google_drive


class Douban(Dataset):
    r"""Douban dataset for alignment."""
    def __init__(self,
                 root: Union[str, Path],
                 download: Optional[bool] = False,
                 ratio: Optional[float] = 0.2,
                 precision: Optional[int] = 32,
                 seed: Optional[int] = 0):

        if download:
            download_file_from_google_drive(
                remote_file_id='1KnP4yzQIZ9J36x-or8uViYVc1eyacn95',
                save_filename='douban.pt',
                root=root)

        if not self._check_integrity(root):
            raise RuntimeError('Douban dataset not found or corrupted. You can use download=True to download it')

        super(Douban, self).__init__(root=root, name='douban', ratio=ratio, precision=precision, seed=seed)

    def _check_integrity(self, root):
        return os.path.exists(os.path.join(root, 'douban.pt'))
