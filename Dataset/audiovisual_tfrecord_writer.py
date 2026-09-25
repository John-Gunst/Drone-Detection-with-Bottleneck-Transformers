import tensorflow as tf
import numpy as np

#wraps raw bytes into the protobuf Feature types that TFRecord expects
#Byte_list wants list so value gets wrapped as [value]
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

#defines the function that turns video array + audio array + labels into sing TFRecord entry
def serialize_example(video_frames: np.ndarray, audio_spec: np.ndarray, label: int):
    # video_frames: (T, H, W, C) uint8, audio_spec: (F, T) float32
    #dictionary of fields
    feature = {
        'video': _bytes_feature(video_frames.tobytes()), #.tobytes() flatten array
        'video_shape': _float_feature(video_frames.shape), #flattening removes shape so shape is stored seperately
        'audio': _bytes_feature(audio_spec.tobytes()),
        'audio_shape': _float_feature(audio_spec.shape),
        'label': _float_feature([label]),
    }
    #wraps the feature dict into Example structure, that gets written to disk
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()
#opens .tfrecord file for writing
test_set = [
    (
        np.random.randint(0, 255, (10, 64, 64, 3), dtype=np.uint8),  # video_frames
        np.random.rand(128, 100).astype(np.float32),                 # audio_spec
        1                                                              # label
    ),
    (
        np.random.randint(0, 255, (10, 64, 64, 3), dtype=np.uint8),
        np.random.rand(128, 100).astype(np.float32),
        0
    ),
]

with tf.io.TFRecordWriter("./drone_clips_test.tfrecord") as writer:
     for video_frames, audio_spec, label in test_set:
        writer.write(serialize_example(video_frames, audio_spec, label))