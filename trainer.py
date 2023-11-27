import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from hifi_gan import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator
from processor import HiFiGANProcessor
import pandas as pd
from typing import Union, Optional
import itertools
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchsummary
from tqdm import tqdm

class HiFiGANTrainer:
    def __init__(self,
                processor: HiFiGANProcessor,
                upsample_rates: Union[list, tuple]=[8,8,2,2],
                upsample_kernel_sizes: Union[list, tuple]=[16,16,4,4],
                upsample_initial_channel: int=512,
                resblock_kernel_sizes: Union[list, tuple]=[3,7,11],
                resblock_dilation_sizes: Union[list, tuple]=[[1,3,5], [1,3,5], [1,3,5]],
                device: str = "cpu",
                lr: float = 0.00003,
                checkpoint: str = None) -> None:
        self.generator = Generator(
            n_mel_channels=processor.n_mel_channels,
            upsample_rates=upsample_rates,
            upsample_kernel_sizes=upsample_kernel_sizes,
            upsample_initial_channel=upsample_initial_channel,
            resblock_kernel_sizes=resblock_kernel_sizes,
            resblock_dilation_sizes=resblock_dilation_sizes
        )
        
        self.processor = processor

        self.multi_period_discriminator = MultiPeriodDiscriminator()
        self.multi_scale_discriminator = MultiScaleDiscriminator()

        self.loss = 0.0
        self.losses = []

        self.epoch = 0

        self.val_loss = 0.0
        self.val_losses = []

        self.optimizer_g = optim.Adam(params=self.generator.parameters(), lr=lr)
        self.optimizer_d = optim.Adam(params=itertools.chain(
            self.multi_period_discriminator.parameters(), self.multi_scale_discriminator.parameters()
        ), lr=lr)

        self.device = device

        self.generator.to(device)
        self.multi_period_discriminator.to(device)
        self.multi_scale_discriminator.to(device)

        self.checkpoint = checkpoint

        if self.checkpoint is not None:
            self.load_model(self.checkpoint)

    def build_dataloader(self, dataset: Dataset, batch_size: int):
        return DataLoader(dataset=dataset, batch_size=batch_size,shuffle=True, collate_fn=self.get_batch)

    def get_batch(self, batch):
        mels, signals = self.processor(batch)

        return mels, signals

    def summary(self):
        print("Generator: ")
        torchsummary.summary(self.generator)

        print("\nMulti Period Discriminator: ")
        torchsummary.summary(self.multi_period_discriminator)

        print("\nMulti Scale Discriminator: ")
        torchsummary.summary(self.multi_scale_discriminator)

    def __save_model(self, path: str):
        torch.save({
            ModelInfo.GENERATOR_STATE_DICT: self.generator.state_dict(),
            ModelInfo.MULTI_PERIOD_STATE_DICT: self.multi_period_discriminator.state_dict(),
            ModelInfo.MULTI_SCALE_STATE_DICT: self.multi_scale_discriminator.state_dict(),
            ModelInfo.GENERATOR_OPTIMIZER_STATE_DICT: self.optimizer_g.state_dict(),
            ModelInfo.DISCRIMINATOR_OPTIMIZER_STATE_DICT: self.optimizer_d.state_dict(),
            ModelInfo.EPOCH: self.epoch,
            ModelInfo.LOSS: self.losses,
            ModelInfo.VAL_LOSS: self.val_losses
        }, path)

    def save_model(self, path: str):
        try:
            self.__save_model(path)
        except:
            self.__save_model("./hifi_gan.pt")

    def __load_model(self, path: str):
        checkpoint = torch.load(path)

        self.generator.load_state_dict(checkpoint[ModelInfo.GENERATOR_STATE_DICT])
        self.multi_period_discriminator.load_state_dict(checkpoint[ModelInfo.MULTI_PERIOD_STATE_DICT])
        self.multi_scale_discriminator.load_state_dict(checkpoint[ModelInfo.MULTI_SCALE_STATE_DICT])

        self.optimizer_g.load_state_dict(checkpoint[ModelInfo.GENERATOR_OPTIMIZER_STATE_DICT])
        self.optimizer_d.load_state_dict(checkpoint[ModelInfo.DISCRIMINATOR_OPTIMIZER_STATE_DICT])

        self.epoch = checkpoint[ModelInfo.EPOCH]
        self.losses = checkpoint[ModelInfo.LOSS]
        self.val_losses = checkpoint[ModelInfo.VAL_LOSS]

    def load_model(self, path: str):
        if os.path.exists(path):
            self.__load_model(path)

    def train_step(self, inputs: torch.Tensor, labels: torch.Tensor):
        labels = labels.unsqueeze(1)
        self.optimizer_d.zero_grad()

        y_g_hat = self.generator(inputs)

        y_dp_hat_r, y_dp_hat_g, _, _ = self.multi_period_discriminator(labels, y_g_hat.detach())
        loss_disc_p = discriminator_loss(y_dp_hat_r, y_dp_hat_g)

        y_ds_hat_r, y_ds_hat_g, _, _ = self.multi_scale_discriminator(labels, y_g_hat.detach())
        loss_disc_s = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

        loss_disc_all = loss_disc_p + loss_disc_s

        loss_disc_all.backward()
        
        self.optimizer_d.step()

        self.optimizer_g.zero_grad()

        _, y_dp_hat_g_, fmap_p_r, fmap_p_g = self.multi_period_discriminator(labels, y_g_hat)

        _, y_ds_hat_g_, fmap_s_r, fmap_s_g = self.multi_scale_discriminator(labels, y_g_hat)

        gen_mel = self.processor.log_mel_spectrogram(y_g_hat.cpu()).to(self.device)

        mel_loss = mel_spectrogram_loss(gen_mel, F.pad(inputs, pad=(0,1), mode='reflect'))

        loss_fm_p = feature_loss(fmap_p_r, fmap_p_g)
        loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)

        loss_gen_p = generator_loss(y_dp_hat_g_)
        loss_gen_s = generator_loss(y_ds_hat_g_)

        final_loss = loss_gen_p + loss_fm_p + mel_loss + loss_gen_s + loss_fm_s
        final_loss.backward()

        self.optimizer_g.step()

        self.loss += final_loss.item()

    def validate(self, dataloader: DataLoader):
        print("\tValidating...")
        num_batches = len(dataloader)

        for _, data in enumerate(dataloader):
            mel = data[0].to(self.device)
            signal = data[1].to(self.device)

            with torch.no_grad():
                output = self.generator(mel)

                _, y_dp_hat_g, fmap_p_r, fmap_p_g = self.multi_period_discriminator(signal, output)
                _, y_ds_hat_g, fmap_s_r, fmap_s_g = self.multi_scale_discriminator(signal, output)

                gen_mel = torch.tensor(self.processor.mel_spectrogram(output.cpu().numpy())).to(self.device)

                loss_fm_p = feature_loss(fmap_p_r, fmap_p_g)
                loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)

                loss_gen_p = generator_loss(y_dp_hat_g)
                loss_gen_s = generator_loss(y_ds_hat_g)

                mel_loss = mel_spectrogram_loss(gen_mel, F.pad(mel, pad=(0,1), mode='reflect'))

                totol_loss = loss_gen_p  + loss_fm_p + mel_loss + loss_gen_s + loss_fm_s

                self.val_loss += totol_loss.item()
            
        val_loss = self.val_loss / num_batches

        print(f"Validation Loss: {(val_loss):.4f}")

    def fit(self, dataset: Dataset, num_epochs: int = 1, batch_size: int = 1, val_dataset: Dataset = None, val_batch_size: int = 1):
        self.summary()

        dataloader = self.build_dataloader(dataset, batch_size)
        num_batches = len(dataloader)

        if val_dataset is not None:
            val_dataloader = self.build_dataloader(val_dataset, val_batch_size)

        self.generator.train()
        self.multi_period_discriminator.train()
        self.multi_scale_discriminator.train()

        for _ in range(num_epochs):
            print(f"EPOCH {self.epoch + 1}")
            for _, data in enumerate(tqdm(dataloader)):
                mel = data[0].to(self.device)
                signal = data[1].to(self.device)

                self.train_step(mel, signal)
            
            epoch_loss = self.loss / num_batches

            print(f"Train Loss: {(epoch_loss):.4f}")

            self.losses.append(epoch_loss)

            if val_dataset is not None:
                self.validate(val_dataloader)

            print(f"DONE EPOCH {self.epoch + 1}\n")

            self.epoch += 1
        

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

def mel_spectrogram_loss(outputs: torch.Tensor, labels: torch.Tensor):
    return F.l1_loss(outputs, labels) * 45

def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))
    return loss*2

def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1-dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)

    return loss


def generator_loss(disc_outputs):
    loss = 0
    for dg in disc_outputs:
        l = torch.mean((1-dg)**2)
        loss += l

    return loss

class ModelInfo:
    GENERATOR_STATE_DICT = 'generator_state_dict'
    MULTI_PERIOD_STATE_DICT = 'multi_period_state_dict'
    MULTI_SCALE_STATE_DICT = 'multi_scale_state_dict'
    GENERATOR_OPTIMIZER_STATE_DICT = 'generator_optimizer_state_dict'
    DISCRIMINATOR_OPTIMIZER_STATE_DICT = 'discriminator_optimizer_state_dict'
    EPOCH = 'epoch'
    LOSS = 'loss'
    VAL_LOSS = 'val_loss'