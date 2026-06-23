"""Frame-source-backed track dataset (EgoSmith first-party).

EgoSmith's replacement for HaWoR's ``lib/datasets/track_dataset.py``: instead of a
list of image-file paths it reads frames lazily through the pipeline's
``frame_source`` abstraction (so the motion stage never materializes a frame list
and can overlap decode with GPU compute). It reuses HaWoR's obtained
``crop`` / ``boxes_2_cs`` from ``lib.utils.imutils`` (unmodified upstream) and adds a
guard so a degenerate box yields a blank crop instead of crashing.

Kept first-party (not a patch on the obtained HaWoR file) so ``lib/datasets/track_dataset.py``
and ``lib/utils/imutils.py`` ship as unmodified upstream symlinks.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize, ToTensor, Compose

from lib.core import constants
from lib.utils.imutils import crop, boxes_2_cs


class TrackDatasetEval(Dataset):
    """Load per-frame crops for the tracked boxes, reading frames via frame_source."""

    def __init__(self, frame_source, frame_indices, boxes,
                 crop_size=256, dilate=1.0,
                 img_focal=None, img_center=None, normalization=True,
                 item_idx=0, do_flip=False):
        super(TrackDatasetEval, self).__init__()

        self.frame_source = frame_source
        self.frame_indices = np.asarray(frame_indices)
        self.crop_size = crop_size
        self.normalization = normalization
        self.normalize_img = Compose([
                            ToTensor(),
                            Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
                        ])

        self.boxes = boxes
        self.box_dilate = dilate
        self.centers, self.scales = boxes_2_cs(boxes)

        self.img_focal = img_focal
        self.img_center = img_center
        self.item_idx = item_idx
        self.do_flip = do_flip

    def __len__(self):
        return len(self.frame_indices)

    def _crop(self, img, center, scale):
        # HaWoR's crop can raise on a degenerate box (zero/negative window); fall
        # back to a blank crop so one bad detection doesn't kill the whole clip.
        try:
            return crop(img, center, scale,
                        [self.crop_size, self.crop_size], rot=0).astype('uint8')
        except Exception:
            return np.zeros((self.crop_size, self.crop_size, 3), dtype='uint8')

    def __getitem__(self, index):
        item = {}
        frame_idx = int(self.frame_indices[index])
        scale = self.scales[index] * self.box_dilate
        center = self.centers[index].copy()

        img_focal = self.img_focal
        img_center = self.img_center

        img = self.frame_source.get_frame(frame_idx, rgb=True)
        if self.do_flip:
            img = img[:, ::-1, :]
            img_width = img.shape[1]
            center[0] = img_width - center[0] - 1
        img_crop = self._crop(img, center, scale)

        if self.normalization:
            img_crop = self.normalize_img(img_crop)
        else:
            img_crop = torch.from_numpy(img_crop)
        item['img'] = img_crop

        if self.do_flip:
            item['do_flip'] = torch.tensor(1).float()
        item['img_idx'] = torch.tensor(index).long()
        item['frame_idx'] = torch.tensor(frame_idx).long()
        item['scale'] = torch.tensor(scale).float()
        item['center'] = torch.tensor(center).float()
        item['img_focal'] = torch.tensor(img_focal).float()
        item['img_center'] = torch.tensor(img_center).float()

        return item
