from typing import Union, Optional
from pathlib import Path

from PyNetAlign.data import Dataset


class FoursquareTwitter(Dataset):
    r"""Foursquare-Twitter dataset for alignment."""
    def __init__(self,
                 root: Union[str, Path],
                 ratio: Optional[float] = 0.2,
                 precision: Optional[int] = 32,
                 seed: Optional[int] = None):
        super(FoursquareTwitter, self).__init__(root=root, name='foursquare_twitter', ratio=ratio, precision=precision, seed=seed)
