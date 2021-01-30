import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint
import time
import lintel

DEBUG_FLAG = False


class VideoRecord(object):

    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class TSNDataSet(data.Dataset):

    def __init__(self, root_path, list_file,
                 num_segments=3, new_length=1,
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 force_grayscale=False, random_shift=True, test_mode=False,
                 read_mode='img', skip=0):

        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.read_mode = read_mode
        self.skip = skip

        self.need_length = new_length * (skip + 1) - skip

        self._parse_list()

    def _load_image(self, directory, idx):
        return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')]

    def _load_image_from_video(self, video_data, p):
        return [Image.fromarray(video_data[p])]

    def _parse_list(self):
        self.video_list = [VideoRecord(x.strip().split(' '))
                           for x in open(self.list_file)]

    def _sample_indices(self, record):
        """

        :param record: VideoRecord
        :return: list
        """

        average_duration = (record.num_frames -
                            self.need_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + \
                randint(average_duration, size=self.num_segments)
        elif record.num_frames > self.need_length:
            offsets = np.sort(randint(record.num_frames -
                                      self.need_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets

    def _get_val_indices(self, record):
        if record.num_frames > self.num_segments + self.need_length - 1:
            tick = (record.num_frames - self.need_length + 1) / \
                float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x)
                                for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets

    def _get_test_indices(self, record):

        return self._get_val_indices(record)

    def __getitem__(self, index):
        record = self.video_list[index]

        if not self.test_mode:
            segment_indices = self._sample_indices(
                record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)

        return self.get(record, segment_indices)

    def _get_full_indices(self, indices, num_frames):
        full_indices = []
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                full_indices.append(p)
                if p < num_frames - 1:
                    p += 1 + self.skip
                if p >= num_frames:
                    p = num_frames - 1
        full_indices = list(set(full_indices))
        full_indices.sort()
        return full_indices

    def get(self, record, indices):

        images = list()
        debug_info = []
        if self.read_mode == 'video':
            video_data = {}
            t1 = time.time()
            finish_flag = False
            full_indices = self._get_full_indices(indices, record.num_frames)
            with open(os.path.join(self.root_path, record.path), 'rb') as f:
                enc_vid = f.read()
            df, w, h = lintel.loadvid_frame_nums(
                enc_vid, frame_nums=full_indices)
            df = np.reshape(df, (len(full_indices), h, w, 3))
            for i in range(len(full_indices)):
                video_data[full_indices[i]] = df[i]
            t2 = time.time()
            debug_info.append('read video: {:.4f}s'.format(t2 - t1))

        t1 = time.time()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                if self.read_mode == 'video':
                    seg_imgs = self._load_image_from_video(video_data, p)
                else:
                    seg_imgs = self._load_image(
                        os.path.join(self.root_path, record.path), p)
                images.extend(seg_imgs)
                if p < record.num_frames - 1:
                    p += 1 + self.skip
                if p >= record.num_frames:
                    p = record.num_frames - 1
        t2 = time.time()
        debug_info.append('load image: {:.4f}s'.format(t2 - t1))

        t1 = time.time()
        process_data = self.transform(images)
        t2 = time.time()
        debug_info.append('transform data: {:.4f}s'.format(t2 - t1))
        if DEBUG_FLAG:
            print(debug_info)
        return process_data, record.label

    def __len__(self):
        return len(self.video_list)
