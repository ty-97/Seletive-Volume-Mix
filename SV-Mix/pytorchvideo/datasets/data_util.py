"""
Dataset of video classification task
By Zhaofan Qiu, Copyright 2022.
"""
import random
import lmdb
import os
import io
#import cv2
import decord
from PIL import Image


def load_list(list_root):
    with open(list_root, 'r') as f:
        samples = f.readlines()
    return samples


def parse_train_sample(sample, num_segments, clip_length, num_steps, equal_interval=True):
    ss = sample.split(' ')
    video_path = sample[:-len(ss[-1]) - 1 - len(ss[-2]) - 1]
    duration = int(ss[-2])
    label = int(ss[-1][:-1])

    if '48' in sample:
        video_path = video_path.replace('diving48/img','rgb')
    # sample frames offsets
    offsets = []
    length_ext = clip_length * num_steps
    ave_duration = duration // num_segments
    if ave_duration >= length_ext:
        offset = random.randint(0, ave_duration - length_ext)
        for i in range(num_segments):
            if equal_interval:
                offsets.append(offset + i * ave_duration)
            else:
                offsets.append(random.randint(0, ave_duration - length_ext) + i * ave_duration)
    else:
        if duration >= length_ext:
            float_ave_duration = float(duration - length_ext) / float(num_segments)
            offset = random.randint(0, int(float_ave_duration))
            for i in range(num_segments):
                offsets.append(offset + int(i * float_ave_duration))
        else:
            print(video_path, length_ext)
            raise NotImplementedError
    return video_path, offsets, label


def parse_test_sample(sample, num_segments, clip_length, num_steps, num_clips, clip_idx):
    ss = sample.split(' ')
    video_path = sample[:-len(ss[-1]) - 1 - len(ss[-2]) - 1]
    duration = int(ss[-2])
    label = int(ss[-1][:-1])
    
    if '48' in sample:
        video_path = video_path.replace('diving48/img','rgb')
    # sample frames offsets
    offsets = []
    length_ext = clip_length * num_steps
    ave_duration = duration // num_segments
    if ave_duration >= length_ext:
        for i in range(num_segments):
            offsets.append(int(float(ave_duration - length_ext) * clip_idx / num_clips) + i * ave_duration)
    else:
        if duration >= length_ext:
            float_ave_duration = float(duration - length_ext) / float(num_segments)
            for i in range(num_segments):
                offsets.append(
                    int(float_ave_duration * clip_idx / num_clips) + int(i * float_ave_duration))
        else:
            raise NotImplementedError
    return video_path, offsets, label


def load_rgb_lmdb(root_path, video_path, offsets, num_segments, clip_length, num_steps):
    """Return the clip buffer sample from video lmdb."""
    lmdb_env = lmdb.open(os.path.join(root_path, video_path), readonly=True, lock=False)

    with lmdb_env.begin() as lmdb_txn:
        image_list = []
        for offset in offsets:
            for frame_id in range(offset + 1, offset + num_steps * clip_length + 1, num_steps):
                bio = io.BytesIO(lmdb_txn.get('{:06d}.jpg'.format(frame_id).encode()))
                image = Image.open(bio).convert('RGB')
                image_list.append(image)
    lmdb_env.close()
    return image_list


def load_rgb_raw(root_path, video_path, offsets, num_segments, clip_length, num_steps):
    """Return the clip buffer sample from raw video."""
    cap = cv2.VideoCapture(os.path.join(root_path, video_path))

    image_list = []
    pos_list = []
    current_pos = 0
    for offset in offsets:
        for frame_id in range(offset + 1, offset + num_steps * clip_length + 1, num_steps):
            if (frame_id - 1) in pos_list:
                idx = pos_list.index(frame_id - 1)
                image_list.append(image_list[idx].copy())
            else:
                if not frame_id - 1 == current_pos:
                    if (frame_id - 1 - current_pos > 0) and (frame_id - 1 - current_pos <= 16):
                        while frame_id - 1 - current_pos > 0:
                            success, frame = cap.read()
                            current_pos += 1
                    else:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id - 1)
                        current_pos = frame_id - 1

                # assert frame_id - 1 == current_pos
                success, frame = cap.read()
                current_pos += 1
                if not success:
                    print(video_path)
                    print(frame_id - 1)
                    assert success

                image = Image.fromarray(frame[..., ::-1])
                image_list.append(image)
            pos_list.append(frame_id - 1)
    cap.release()
    return image_list

def load_rgb_decord(root_path, video_path, offsets, num_segments, clip_length, num_steps, suffix):
    """Return the clip buffer sample from raw video using decord"""
    if '_' in suffix:
        suffix = suffix.split('_')[1]
    else:
        suffix = ''
        
    # if 'ucf' in root_path:
    #     video_path = os.path.join('videos',*video_path.split('/')[1:])
    if 'EGTEA_Gaze' in root_path:
        video_path = os.path.join('cropped_clips',*video_path.split('/')[2:])
    vr = decord.VideoReader(os.path.join(root_path, video_path + suffix), num_threads=1)

    image_list = []
    pos_list = []
    for offset in offsets:
        pos_list.extend(range(offset, offset + num_steps * clip_length, num_steps))
    assert max(pos_list) < len(vr), f'Sample index: {max(pos_list)} out of bounds for video length: {len(vr)}'
    frames = vr.get_batch([pos_list]).asnumpy()
    image_list.extend([Image.fromarray(frame) for frame in frames])
    return image_list


def load_rgb_frame(root_path, video_path, offsets, num_segments, clip_length, num_steps):
    """Return the clip buffer sample from video frames."""
    if 'some_some' in root_path:
        root_path = os.path.join('/',*root_path.split('/')[:-1])
    image_list = []
    for offset in offsets:
        for frame_id in range(offset + 1, offset + num_steps * clip_length + 1, num_steps):
            if 'some_some_v2' in video_path:
                image = Image.open(os.path.join(root_path, video_path, '{:06d}'.format(frame_id) + '.jpg')).convert('RGB')
            else:
                image = Image.open(os.path.join(root_path, video_path, '{:05d}'.format(frame_id) + '.jpg')).convert('RGB')
            image_list.append(image)
    return image_list


def load_flow_lmdb(root_path, video_path, offsets, num_segments, clip_length, num_steps):
    """Return the clip buffer sample from video lmdb."""
    lmdb_env = lmdb.open(os.path.join(root_path, video_path), readonly=True)

    with lmdb_env.begin() as lmdb_txn:
        image_list = []
        for offset in offsets:
            for frame_id in range(offset + 1, offset + num_steps * clip_length + 1, num_steps):
                bio = io.BytesIO(lmdb_txn.get('{:06d}_c1.jpg'.format(frame_id).encode()))
                image = Image.open(bio)
                image_list.append(image)
                bio = io.BytesIO(lmdb_txn.get('{:06d}_c2.jpg'.format(frame_id).encode()))
                image = Image.open(bio)
                image_list.append(image)
    lmdb_env.close()
    return image_list
