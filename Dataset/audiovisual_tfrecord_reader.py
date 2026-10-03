from tfrecord.torch.dataset import TFRecordDataset
import torch
import numpy as np

description = {
    'video': 'byte',
    'video_shape': 'float',
    'audio': 'byte',
    'audio_shape': 'float',
    'label': 'float',
}

dataset=TFRecordDataset("drone_clips_test.tfrecord", index_path=None, description=description)

# Compare
def collate_fn(batch):
    videos, audios, labels=[], [], []
    for item in batch:
        v_shape = item['video_shape'].astype(int)
        a_shape = item['audio_shape'].astype(int)
        video = torch.frombuffer(item['video'], dtype=torch.uint8).reshape(*v_shape)
        audio = torch.frombuffer(item['audio'], dtype=torch.float32).reshape(*a_shape)
        videos.append(video)
        audios.append(audio)
        labels.append(item['label'][0])
    return torch.stack(videos), torch.stack(audios), torch.tensor(labels)

loader = torch.utils.data.DataLoader(dataset, batch_size=8, collate_fn=collate_fn)
print(loader)