from typing import Union, Optional
from pathlib import Path
import os

from PyNetAlign.data import Dataset
from .utils import download_file_from_google_drive


class PPI(Dataset):
    r"""PPI dataset for alignment."""
    def __init__(self,
                 root: Union[str, Path],
                 download: Optional[bool] = False,
                 train_ratio: Optional[float] = 0.2,
                 precision: Optional[int] = 32,
                 seed: Optional[int] = 0):

        if download:
            download_file_from_google_drive(
                remote_file_id='14UhPGdSdxPjokZbPhTNgMbEkC6gLv-Pr',
                save_filename='PPI.pt',
                root=root)

        if not self._check_integrity(root):
            raise RuntimeError('PPI dataset not found or corrupted. You can use download=True to download it')

        super(PPI, self).__init__(root=root, name='PPI', train_ratio=train_ratio, precision=precision, seed=seed)

    def _check_integrity(self, root):
        return os.path.exists(os.path.join(root, 'PPI.pt'))
