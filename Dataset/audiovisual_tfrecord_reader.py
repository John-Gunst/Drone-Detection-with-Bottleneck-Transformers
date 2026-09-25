from tfrecord.torch.dataset import TFRecordDataset
import torch
import numpy as np

np.random.seed(42)
video_original = np.random.randint(0, 255, (10, 64, 64, 3), dtype=np.uint8)
audio_original = np.random.rand(128, 100).astype(np.float32)

description = {
    'video': 'byte',
    'video_shape': 'float',
    'audio': 'byte',
    'audio_shape': 'float',
    'label': 'float',
}

dataset=TFRecordDataset("drone_clips_test.tfrecord", index_path=None, description=description)

first_item = next(iter(dataset))

# Reconstruct video and audio using the stored shape info
v_shape = first_item['video_shape'].astype(int)
a_shape = first_item['audio_shape'].astype(int)

video_reconstructed = np.frombuffer(first_item['video'], dtype=np.uint8).reshape(*v_shape)
audio_reconstructed = np.frombuffer(first_item['audio'], dtype=np.float32).reshape(*a_shape)
video_match = np.array_equal(video_original, video_reconstructed)
audio_match = np.array_equal(audio_original, audio_reconstructed)


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
print("Video shapes:", video_original.shape, "vs", video_reconstructed.shape)
print("Audio shapes:", audio_original.shape, "vs", audio_reconstructed.shape)
print("Video matches:", video_match)
print("Audio matches:", audio_match)
print(loader)