from torch.utils.data import Dataset
from processor import HiFiGANProcessor
from typing import Optional
import pandas as pd

class HiFiGANDataset(Dataset):
    def __init__(self, manifest_path: str, processor: HiFiGANProcessor, num_examples: Optional[int] = None) -> None:
        super().__init__()
        self.prompts = pd.read_csv(manifest_path, sep="\t")

        if num_examples is not None:
            self.prompts = self.prompts[:num_examples]

        self.processor = processor

    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, index):
        audio_path = self.prompts.iloc[index]['path']

        signal = self.processor.load_audio(audio_path)

        return signal