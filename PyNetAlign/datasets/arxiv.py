from typing import Union, Optional
from pathlib import Path
import os

from PyNetAlign.data import Dataset
from .utils import download_file_from_google_drive


class ArXiv(Dataset):
    r"""ArXiv dataset for alignment."""
    def __init__(self,
                 root: Union[str, Path],
                 download: Optional[bool] = False,
                 train_ratio: Optional[float] = 0.2,
                 precision: Optional[int] = 32,
                 seed: Optional[int] = 0):

        if download:
            download_file_from_google_drive(
                remote_file_id='15IEZkaaK9nD1OBMWDvITN8UdRB_3s4G0',
                save_filename='arxiv.pt',
                root=root)

        if not self._check_integrity(root):
            raise RuntimeError('ArXiv dataset not found or corrupted. You can use download=True to download it')

        super(ArXiv, self).__init__(root=root, name='arxiv', train_ratio=train_ratio, precision=precision, seed=seed)

    def _check_integrity(self, root):
        return os.path.exists(os.path.join(root, 'arxiv.pt'))
