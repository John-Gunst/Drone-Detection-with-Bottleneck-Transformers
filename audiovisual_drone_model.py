import torch
import torch.nn as nn

class SimpleDroneAVModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Flatten video (10 frames * 64 * 64 * 3 = 122,880) down to 64
        self.video_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(10 * 64 * 64 * 3, 64),
            nn.ReLU()
        )
        
        # Flatten audio (128 mel bins * 100 time steps = 12,800) down to 64
        self.audio_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 100, 64),
            nn.ReLU()
        )
        
        # Combine both (64 + 64 = 128) and output 1 probability (0 to 1)
        self.classifier = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, video, audio):
        # Convert video to float [0, 1]
        v = video.float() / 255.0
        
        # Extract features
        v_feat = self.video_net(v)
        a_feat = self.audio_net(audio)
        
        # Fuse simply by concatenating
        combined = torch.cat([v_feat, a_feat], dim=1)
        
        # Output probability [B]
        return self.classifier(combined).squeeze(-1)