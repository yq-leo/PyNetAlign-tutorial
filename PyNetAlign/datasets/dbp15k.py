from typing import Union, Optional
from pathlib import Path
import os

from PyNetAlign.data import Dataset
from .utils import download_file_from_google_drive


class DBP15K_FR_EN(Dataset):
    r"""DBP15K_FR-EN dataset for alignment."""
    def __init__(self,
                 root: Union[str, Path],
                 download: Optional[bool] = False,
                 train_ratio: Optional[float] = 0.2,
                 precision: Optional[int] = 32,
                 seed: Optional[int] = 0):

        if download:
            download_file_from_google_drive(
                remote_file_id='1xpZqBYtuLvAFrLESzXAvB7P-tPXmGWTi',
                save_filename='dbp15k_fr-en.pt',
                root=root)

        if not self._check_integrity(root):
            raise RuntimeError('DBP15K_FR-EN dataset not found or corrupted. You can use download=True to download it')

        super(DBP15K_FR_EN, self).__init__(root=root, name='dbp15k_fr-en', train_ratio=train_ratio, precision=precision, seed=seed)

    def _check_integrity(self, root):
        return os.path.exists(os.path.join(root, 'dbp15k_fr-en.pt'))


class DBP15K_JA_EN(Dataset):
    r"""DBP15K_JA-EN dataset for alignment."""
    def __init__(self,
                 root: Union[str, Path],
                 download: Optional[bool] = False,
                 train_ratio: Optional[float] = 0.2,
                 precision: Optional[int] = 32,
                 seed: Optional[int] = 0):

        if download:
            download_file_from_google_drive(
                remote_file_id='1RSf2tyx50zUpHnBGDKzg07CNkQqBekgL',
                save_filename='dbp15k_ja-en.pt',
                root=root)

        if not self._check_integrity(root):
            raise RuntimeError('DBP15K_JA-EN dataset not found or corrupted. You can use download=True to download it')

        super(DBP15K_JA_EN, self).__init__(root=root, name='dbp15k_ja-en', train_ratio=train_ratio, precision=precision, seed=seed)

    def _check_integrity(self, root):
        return os.path.exists(os.path.join(root, 'dbp15k_ja-en.pt'))


class DBP15K_ZH_EN(Dataset):
    r"""DBP15K_ZH-EN dataset for alignment."""
    def __init__(self,
                 root: Union[str, Path],
                 download: Optional[bool] = False,
                 train_ratio: Optional[float] = 0.2,
                 precision: Optional[int] = 32,
                 seed: Optional[int] = 0):

        if download:
            download_file_from_google_drive(
                remote_file_id='18f5zsUBWYsSw5ACWQGFdsHQAkafuHqs5',
                save_filename='dbp15k_zh-en.pt',
                root=root)

        if not self._check_integrity(root):
            raise RuntimeError('DBP15K_ZH-EN dataset not found or corrupted. You can use download=True to download it')

        super(DBP15K_ZH_EN, self).__init__(root=root, name='dbp15k_zh-en', train_ratio=train_ratio, precision=precision, seed=seed)

    def _check_integrity(self, root):
        return os.path.exists(os.path.join(root, 'dbp15k_zh-en.pt'))
