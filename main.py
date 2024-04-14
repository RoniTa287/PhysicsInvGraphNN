import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from physics_net import PhysicsPredictor

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = np.load('3bp_color/color_3bp_vx2_vy2_sl20_r2_g60_m1_dt05.npz')
    data = data['train_x']

    train_x = data.transpose(0, 1, 4, 2, 3).astype(np.float32)  # No need to reshape as before
    print(f"My data after reshaping: {train_x.shape}, {train_x.dtype}")
    train_x = torch.from_numpy(train_x).to(device)

    # Create dataset and dataloader
    train_dataset = TensorDataset(train_x)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # 5000/32 batches

    model = PhysicsPredictor(input_channels=3, hidden_dim=128, num_layers=1, n_objs=3, seq_len=20, input_steps=10,
                             pred_steps=5, device=device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.MSELoss()

    # model.train()
    # (32, 20, 3, 36, 36)
    for batch in train_loader:
        # print(enumerate(train_loader))
        images = batch[0]     # (32, 20, 3, 36, 36)
        # optimizer.zero_grad()
        output = model(images)

        # Here, using the same images tensor as a placeholder; adjust as necessary
        # loss = loss_function(output, images[:, :model.pred_steps])
        # loss.backward()
        # optimizer.step()
        # print(f"Loss: {loss.item()}")
        # print(output)
