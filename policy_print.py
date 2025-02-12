import torch
import os
import numpy as np

# If you copied the entire PolicyValueNet class from your main code:
# Just paste it here again or import it from your module.

class PolicyValueNet(torch.nn.Module):
    def __init__(self, board_size=9):
        super().__init__()
        self.board_size = board_size

        self.conv1 = torch.nn.Conv2d(2, 16, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 32, kernel_size=3, padding=1)

        self.policy_conv = torch.nn.Conv2d(32, 2, kernel_size=1)
        self.policy_fc = torch.nn.Linear(2 * board_size * board_size,
                                         board_size * board_size + 1)

        self.value_conv = torch.nn.Conv2d(32, 1, kernel_size=1)
        self.value_fc1 = torch.nn.Linear(board_size * board_size, 64)
        self.value_fc2 = torch.nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))

        # Policy head
        p = torch.relu(self.policy_conv(x))
        p = p.view(p.size(0), -1)  # flatten
        p = self.policy_fc(p)

        # Value head
        v = torch.relu(self.value_conv(x))
        v = v.view(v.size(0), -1)
        v = torch.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v


def main():
    # Change this to your board size if needed
    BOARD_SIZE = 13

    # Create an instance of the network
    net = PolicyValueNet(board_size=BOARD_SIZE)

    # Check if the file exists
    model_file = "policy_value_net.pth"
    if not os.path.isfile(model_file):
        print(f"Model file '{model_file}' not found.")
        return

    # Load the weights
    state_dict = torch.load(model_file, map_location=torch.device("cpu"))
    net.load_state_dict(state_dict)
    net.eval()

    print("==> Loaded policy_value_net.pth successfully!")
    print("==> Network parameter details:\n")

    # Loop over each parameter and print its name, shape, and optionally some values
    for name, param in net.named_parameters():
        print(f"Parameter: {name}")
        print(f" - shape: {tuple(param.size())}")
        
        # Convert param to numpy to print or inspect values
        data_np = param.detach().cpu().numpy()
        
        # Print a small subset of values if it's large
        if data_np.size <= 20:
            # If it's small, print the whole array
            print(" - values:", data_np)
        else:
            # If it's large, just print some samples
            print(" - first 10 values:", data_np.flatten()[:10])
            print(" - mean:", np.mean(data_np), "std:", np.std(data_np))
        print()

    print("==> Done printing model parameters.")

if __name__ == "__main__":
    main()
