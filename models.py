# %% TCN MODEL only
import torch
from torch import nn
from torch.nn import Module
from pytorch_tcn import TCN

# new tcn
class TCNModel(Module):
    def __init__(self,
                 num_sensors,
                 num_codes,
                 mean_sensors,
                 std_sensors,
                 num_channels=[16, 16, 16],
                 kernel_size=12,
                 dropout=.1,
                 causal=True):
        super(TCNModel, self).__init__()
        
        # Handle zero standard deviation (constant features)
        safe_std = torch.where(torch.FloatTensor(std_sensors) == 0, 
                              torch.ones_like(torch.FloatTensor(std_sensors)), 
                              torch.FloatTensor(std_sensors))
        
        # Register normalization parameters as buffers (won't be trained)
        self.register_buffer('sensor_mean', torch.FloatTensor(mean_sensors))
        self.register_buffer('sensor_std', safe_std)
        
        self.num_inputs = num_sensors + num_codes

        self.tcn = TCN(
            num_inputs = self.num_inputs,
            num_channels = num_channels,
            kernel_size= kernel_size,
            dropout= dropout,
            causal=causal,
            use_skip_connections=True,
            input_shape="NCL", # N: BatchSize + C: Features + L: sequence length
            use_norm="weight_norm",
            activation="relu"
        )

        self.output_layers = nn.Sequential(
            nn.Linear(num_channels[-1], 64),
            nn.Linear(64, 1)
          #  Tanh()
            # Sigmoid()
        )

    def forward(self, sensor_data, code_data):
        # Normalize sensor data using registered buffers
        # sensor_data shape: [batch, 50, 120]
        # sensor_mean/std shape: [50] -> need to reshape for broadcasting
        sensor_normalized = (sensor_data - self.sensor_mean.unsqueeze(0).unsqueeze(2)) / self.sensor_std.unsqueeze(0).unsqueeze(2)
        
        # Concatenate normalized sensors with codes
        x = torch.cat([sensor_normalized, code_data], dim=1)
        
        # Pass through TCN
        tcn_output = self.tcn(x)
        last_tcn_output = tcn_output[:, :, -1] # we only care for the last one
        
        return self.output_layers(last_tcn_output).squeeze()
        # tanh_output = self.output_layers(last_tcn_output)

        # scaled_output = (tanh_output + 1) / 2 # scale to 0 - 1
        # return scaled_output.squeeze(1)