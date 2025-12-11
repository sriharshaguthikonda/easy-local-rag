import torch
import winsound


print(torch.__version__)  # Should print the installed PyTorch version
print(torch.cuda.is_available())  # Should return True if CUDA is available

# Beep to indicate the script has ended


winsound.Beep(1000, 500)  # Frequency: 1000 Hz, Duration: 500 ms
