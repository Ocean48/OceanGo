import torch
import pprint

file_path = "policy_value_net.pth"
model_state_dict = torch.load(file_path)

# Create a pprint object for more control (optional)
pp = pprint.PrettyPrinter(indent=4)  # Indent by 4 spaces

# Pretty print the entire state_dict
pp.pprint(model_state_dict.keys())  # Print the keys first for an overview

# Or, pretty print specific parts:
for key, value in model_state_dict.items():
    print(f"\nKey: {key}")
    if isinstance(value, torch.Tensor):
        print(f"  Shape: {value.shape}")
        # Print a few elements if it's a small tensor, otherwise, skip printing the data itself:
        if value.numel() < 500:  # Adjust this threshold as needed
            pp.pprint(value)
        else:
            print("  (Tensor data not printed for brevity)")
    elif isinstance(value, dict): # Handle nested dictionaries
        pp.pprint(value.keys()) # Print the keys of nested dictionaries
    else:
        pp.pprint(value) # Print other data types