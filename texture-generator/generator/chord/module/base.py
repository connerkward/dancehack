import torch
import torch.nn as nn

class Base(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        self.setup()
    
    def setup(self):
        raise NotImplementedError
    
