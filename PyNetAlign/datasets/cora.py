from typing import Union, Optional
from pathlib import Path
import os

from PyNetAlign.data import Dataset
from .utils import download_file_from_google_drive


class Cora(Dataset):
    r"""Cora dataset for alignment."""
    def __init__(self,
                 root: Union[str, Path],
                 download: Optional[bool] = False,
                 train_ratio: Optional[float] = 0.2,
                 precision: Optional[int] = 32,
                 seed: Optional[int] = 0):

        if download:
            download_file_from_google_drive(
                remote_file_id='1ES5o-WNHcV6iFpt2alutJt2EtssvB02g',
                save_filename='cora.pt',
                root=root)

        if not self._check_integrity(root):
            raise RuntimeError('Cora dataset not found or corrupted. You can use download=True to download it')

        super(Cora, self).__init__(root=root, name='cora', train_ratio=train_ratio, precision=precision, seed=seed)

    def _check_integrity(self, root):
        return os.path.exists(os.path.join(root, 'cora.pt'))
