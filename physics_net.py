import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import ConvEncoder, VelEncoder
from decoder import ConvSTDecoder
# from cells import RolloutCell


class PhysicsPredictor(nn.Module):
    def __init__(self, input_channels, hidden_dim, num_layers, n_objs, seq_len, input_steps, pred_steps, device='cpu'):
        super(PhysicsPredictor, self).__init__()
        self.device = device
        self.seq_len = seq_len
        self.input_steps = input_steps
        self.pred_steps = pred_steps
        self.n_objs = n_objs

        # LSTM for processing sequences
        self.lstm = nn.LSTM(input_size=input_channels * 36 * 36, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True)

        # Encoder and Decoder definitions
        self.encoder = ConvEncoder(n_objs=n_objs, conv_ch=input_channels, input_shape=(36, 36))  # Adjust parameters as needed
        # self.vel_est = VelEncoder(n_objs=n_objs, input_steps=input_steps, coord_units=2*n_objs, device='cpu')
        self.decoder = ConvSTDecoder(input_size=(36*36), n_objs=n_objs, conv_ch=3)  # Adjust parameters as needed
        #
        # # Assuming RolloutCell is a custom implementation for your specific task
        # self.rollout_cell = RolloutCell(hidden_dim, self.n_objs)  # Define this class based on your needs

    def forward(self, x):
        batch_size, seq_len, channels, height, width = x.size()
        encoded_frames = torch.zeros(batch_size, seq_len, self.n_objs * 2, device=self.device)
        decoded_frames = torch.zeros(batch_size, seq_len, self.n_objs * 2, device=self.device)

        # Process each frame through the encoder individually
        for t in range(seq_len):
            frame = x[:, t, :, :, :]  # Select each frame at time t
            # print(f"one frame as an input to the encoder: {frame.size()}")
            encoded_frame = self.encoder(frame)
            # Ensure the encoded_frame is correctly reshaped
            if encoded_frame.shape[1] != self.n_objs * 2:
                # Assuming encoded_frame is [batch_size, n_objs, 2]
                encoded_frame = encoded_frame.view(batch_size, -1)  # Reshape to [batch_size, n_objs * 2]
                # print(f"reshaped encoder output: {encoded_frame.size()}")
            decoded_frame = self.decoder.forward(encoded_frame)  # input: torch.Size([32, 6])
            print(f"decoded frames: {decoded_frame.size()}")

            encoded_frames[:, t, :] = encoded_frame
            # decoded_frames[:, t, :] = decoded_frame

        # for t in range(seq_len):


        pos_for_vel_est = encoded_frames[:, :self.input_steps, :].contiguous()
        # Estimate velocities using the position data
        # velocities = self.vel_est(pos_for_vel_est)
        # print(f"velocities: {velocities.size()}")

        # TODo: Do we want to concatenate encoded positions and estimated velocities??
        # For simplicity, assume we concatenate positions and velocities for input_steps, and use only positions for the rest
        # combined_input = torch.cat([encoded_frames, velocities.repeat(1, seq_len - self.input_steps + 1, 1)], dim=-1)
        # lstm_out, _ = self.lstm(combined_input)
        # Now encoded_frames is of shape [batch_size, seq_len, n_objs * 2]
        # Process sequences with LSTM
        lstm_out, _ = self.lstm(encoded_frames)
        print(f"after lstm: {lstm_out.size()}")

        return lstm_out

    def initialize_pos_vel(self, enc_out):
        # Simplified example: Use last encoded output as initial position, zeros for initial velocity
        pos = enc_out[:, -1, :]
        vel = torch.zeros_like(pos).to(self.device)
        return pos, vel