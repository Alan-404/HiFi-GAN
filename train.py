import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from processor import HiFiGANProcessor
from trainer import HiFiGANTrainer, HiFiGANDataset

from argparse import ArgumentParser

import json

parser = ArgumentParser()

parser.add_argument("--config_path", type=str, default="./config.json")

args = parser.parse_args()

config = json.load(open(args.config_path, encoding='utf-8'))

processor_config = config['processor_config']
model_config = config['model_config']
training_config = config['training_config']
validation_config = config['validation_config']

processor = HiFiGANProcessor(
    sample_rate=processor_config['sample_rate'],
    n_mel_channels=processor_config['n_mel_channels'],
    fft_size=processor_config['fft_size'],
    hop_length=processor_config['hop_length'],
    window_size=processor_config['window_size'],
    fmin=processor_config['fmin'],
    fmax=processor_config['fmax'],
    htk=processor_config['htk']
)


trainer = HiFiGANTrainer(
    processor=processor,
    upsample_rates=model_config['upsample_rates'],
    upsample_kernel_sizes=model_config['upsample_kernel_sizes'],
    upsample_initial_channel=model_config['upsample_initial_channel'],
    resblock_dilation_sizes=model_config['resblock_dilation_sizes'],
    resblock_kernel_sizes=model_config['resblock_kernel_sizes'],
    device=training_config['device'],
    lr=training_config['init_lr'],
    checkpoint=training_config['checkpoint']
)

dataset = HiFiGANDataset(manifest_path=training_config['train_path'], processor=processor)

val_dataset = None
if validation_config['use_validation']:
    val_dataset = HiFiGANDataset(manifest_path=validation_config['val_path'], processor=processor)

trainer.fit(
    dataset=dataset,
    num_epochs=training_config['num_epochs'],
    batch_size=training_config['batch_size'],
    val_dataset=val_dataset,
    val_batch_size=validation_config['val_batch_size']
)
