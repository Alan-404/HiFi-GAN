import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchsummary
import itertools

from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage
from ignite.handlers import Checkpoint, DiskSaver, EarlyStopping
from ignite.contrib.handlers.tqdm_logger import ProgressBar

from hifi_gan import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator

from dataset import HiFiGANDataset
from processor import HiFiGANProcessor

import wandb

#####################################
from argparse import ArgumentParser

parser = ArgumentParser()

# Processor Config
parser.add_argument("--n_mel_channels", type=int, default=80)
parser.add_argument("--sample_rate", type=int, default=22050)
parser.add_argument("--fft_size", type=int, default=1024)
parser.add_argument("--hop_length", type=int, default=256)
parser.add_argument("--window_size", type=int, default=1024)
parser.add_argument("--fmin", type=float, default=0.0)
parser.add_argument("--fmax", type=float, default=8000.0)
parser.add_argument("--htk", type=bool, default=True)

# Model Config
parser.add_argument("--upsample_rates", type=int, nargs="+", default=[8,8,2,2])
parser.add_argument("--upsample_kernel_sizes", type=int, nargs="+", default=[16,16,4,4])
parser.add_argument("--upsample_initial_channel", type=int, default=512)
parser.add_argument("--resblock_kernel_sizes", type=int, nargs="+", default=[3,7,11])
parser.add_argument("--resblock_dilation_sizes", type=list, nargs="+", default=[[1,3,5], [1,3,5], [1,3,5]])

# Dataset Config
parser.add_argument("--train_path", type=str, default="./datasets/train.csv")

parser.add_argument("--use_validation", type=bool, default=True)
parser.add_argument("--val_path", type=str, default="./datasets/val.csv")

# Training Config
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--num_epochs", type=int, default=1)
parser.add_argument("--device", type=str, default='cpu')

parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--num_train", type=int, default=None)

parser.add_argument("--val_batch_size", type=int, default=1)
parser.add_argument("--num_val", type=int, default=None)

# Checkpoint Config
parser.add_argument("--checkpoint", type=str, default="./checkpoints")
parser.add_argument("--checkpoint_step", type=str, default=None)

# Early Stopping Config
parser.add_argument("--early_stopping_patience", type=int, default=5)

# WanDB Config
parser.add_argument("--wandb_project_name", type=str, default="(TTS) HiFi - GAN")
parser.add_argument("--wandb_name", type=str, default='trind18')

args = parser.parse_args()
#####################################

wandb.init(project=args.wandb_project_name, name=args.wandb_name)

# Setup Checkpoint
checkpoint = None
def find_latest_checkpoint(checkpoints: list):
    latest = 0
    for checkpoint in checkpoints:
        index = int(checkpoint.replace("checkpoint_", "").replace(".pt", ""))
        if latest < index:
            latest = index
    return latest
if os.path.exists(args.checkpoint) and os.listdir(args.checkpoint) != 0:
    if args.checkpoint_step is None:
        args.checkpoint_step = find_latest_checkpoint(os.listdir(args.checkpoint))
    checkpoint = f"{args.checkpoint}/checkpoint_{args.checkpoint_step}.pt"

# Device Config
device = 'cpu'
if args.device == 'cuda' or args.device == 'gpu':
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

processor = HiFiGANProcessor(
    n_mel_channels=args.n_mel_channels,
    sample_rate=args.sample_rate,
    fft_size=args.fft_size,
    hop_length=args.hop_length,
    window_size=args.window_size,
    fmax=args.fmax,
    fmin=args.fmin,
    htk=args.htk
)

generator = Generator(
    n_mel_channels=processor.n_mel_channels,
    upsample_rates=args.upsample_rates,
    upsample_kernel_sizes=args.upsample_kernel_sizes,
    upsample_initial_channel=args.upsample_initial_channel,
    resblock_kernel_sizes=args.resblock_kernel_sizes,
    resblock_dilation_sizes=args.resblock_dilation_sizes
)
multi_period_discriminator = MultiPeriodDiscriminator()
multi_scale_discriminator = MultiScaleDiscriminator()

optimizer_g = optim.Adam(params=generator.parameters(), lr=args.lr)
optimizer_d = optim.Adam(params=itertools.chain(
                multi_period_discriminator.parameters(), multi_scale_discriminator.parameters()
        ), lr=args.lr)

scheduler_g = lr_scheduler.ExponentialLR(optimizer=optimizer_g, gamma=0.999)
scheduler_d = lr_scheduler.ExponentialLR(optimizer=optimizer_d, gamma=0.999)

def get_batch(batch: list):
    mels, signals = processor(batch)
    return mels, signals

train_dataset = HiFiGANDataset(manifest_path=args.train_path, processor=processor, num_examples=args.num_train)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=get_batch)

if args.use_validation:
    val_dataset = HiFiGANDataset(manifest_path=args.val_path, processor=processor, num_examples=args.num_val)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.val_batch_size, shuffle=True, collate_fn=get_batch)

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

def train_step(engine: Engine, inputs: torch.Tensor, labels: torch.Tensor):
    labels = labels.unsqueeze(1)
    optimizer_d.zero_grad()

    y_g_hat = generator(inputs)

    y_dp_hat_r, y_dp_hat_g, _, _ = multi_period_discriminator(labels, y_g_hat.detach())
    loss_disc_p = discriminator_loss(y_dp_hat_r, y_dp_hat_g)

    y_ds_hat_r, y_ds_hat_g, _, _ = multi_scale_discriminator(labels, y_g_hat.detach())
    loss_disc_s = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

    loss_disc_all = loss_disc_p + loss_disc_s

    loss_disc_all.backward()
        
    optimizer_d.step()

    optimizer_g.zero_grad()

    _, y_dp_hat_g_, fmap_p_r, fmap_p_g = multi_period_discriminator(labels, y_g_hat)

    _, y_ds_hat_g_, fmap_s_r, fmap_s_g = multi_scale_discriminator(labels, y_g_hat)

    gen_mel = processor.log_mel_spectrogram(y_g_hat.cpu()).to(device)

    mel_loss = mel_spectrogram_loss(gen_mel, F.pad(inputs, pad=(0,1), mode='reflect'))

    loss_fm_p = feature_loss(fmap_p_r, fmap_p_g)
    loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)

    loss_gen_p = generator_loss(y_dp_hat_g_)
    loss_gen_s = generator_loss(y_ds_hat_g_)

    final_loss = loss_gen_p + loss_fm_p + mel_loss + loss_gen_s + loss_fm_s
    final_loss.backward()

    optimizer_g.step()

    return final_loss.item(),  loss_disc_all.item()

def val_step(engine: Engine, mels: torch.Tensor, labels: torch.Tensor):
    with torch.no_grad():
        outputs = generator(mels)

    y_dp_hat_r, y_dp_hat_g, fmap_p_r, fmap_p_g = multi_period_discriminator(labels, outputs.detach())
    loss_disc_p = discriminator_loss(y_dp_hat_r, y_dp_hat_g)

    y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = multi_scale_discriminator(labels, outputs.detach())
    loss_disc_s = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

    loss_disc_all = loss_disc_p + loss_disc_s

    gen_mel = torch.tensor(processor.log_mel_spectrogram(outputs.cpu())).to(device)

    loss_fm_p = feature_loss(fmap_p_r, fmap_p_g)
    loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)

    loss_gen_p = generator_loss(y_dp_hat_g)
    loss_gen_s = generator_loss(y_ds_hat_g)

    mel_loss = mel_spectrogram_loss(gen_mel, F.pad(mels, pad=(0,1), mode='reflect'))

    total_loss = loss_gen_p  + loss_fm_p + mel_loss + loss_gen_s + loss_fm_s

    return total_loss.item(), loss_disc_all.item()

def early_stopping_condition(engine):
    return -(engine.state.metrics['generator_loss'] + engine.state.metrics['discriminator_loss'])
    
trainer = Engine(train_step)
train_generator_loss = RunningAverage(output_transform=lambda x: x[0])
train_generator_loss.attach(trainer, 'generator_loss')

train_discriminator_loss = RunningAverage(output_transform=lambda x: x[1])
train_discriminator_loss.attach(trainer, 'disciminator_loss')
ProgressBar().attach(trainer)

validator = Engine(val_step)
val_generator_loss = RunningAverage(output_transform=lambda x: x[0])
val_generator_loss.attach(validator, 'generator_loss')

val_discriminator_loss = RunningAverage(output_transform=lambda x: x[1])
val_discriminator_loss.attach(validator, 'disciminator_loss')

ProgressBar().attach(validator)
early_stopping_handler = EarlyStopping(patience=args.early_stopping_patience, score_function=early_stopping_condition, trainer=trainer)

to_save = {
    'generator': generator,
    'multi_period_discriminator': multi_period_discriminator,
    'multi_scale_discriminator': multi_scale_discriminator,
    'generator_optimizer': optimizer_g,
    'discriminator_optimizer': optimizer_d,
    'trainer': trainer,
    'early_stopping': early_stopping_handler
}
checkpoint_manager = Checkpoint(to_save, save_handler=DiskSaver(dirname=args.checkpoint, create_dir=True, require_empty=False), n_saved=args.early_stopping_patience)

@trainer.on(Events.STARTED)
def start_training(engine: Engine):
    print("Generator")
    torchsummary.summary(generator)
    print("\n")

    print("Multi - Period Discriminator")
    torchsummary.summary(multi_period_discriminator)
    print("\n")

    print("Multi - Scale Discriminator")
    torchsummary.summary(multi_scale_discriminator)
    print("\n")

    generator.train()
    multi_period_discriminator.train()
    multi_scale_discriminator.train()

@trainer.on(Events.EPOCH_STARTED)
def start_epoch(engine: Engine):
    train_generator_loss.reset()
    train_discriminator_loss.reset()

    val_generator_loss.reset()
    val_discriminator_loss.reset()
    
    print(f"============ {engine.state.epoch} =============")

@trainer.on(Events.EPOCH_COMPLETED)
def finish_epoch(engine: Engine):
    print(f"Train: Generator Loss {(engine.state.metrics['generator_loss']):.4f} Discriminator Loss {(engine.state.metrics['discriminator_loss']):.4f}")
    wandb.log({
            'train_generator_loss': engine.state.metrics['generator_loss'], 
            "train_discriminator_loss": engine.state.metrics['discriminator_loss'],
            "generator_learning_rate": optimizer_g.param_groups[0]['lr'],
            'discriminator_learning_rate': optimizer_d.param_groups[0]['lr'],
            'generator_gradient_norm': torch.nn.utils.clip_grad.clip_grad_norm_(generator.parameters(), max_norm=float('-inf')).item(),
            'multi_period_discriminator_gradient_norm': torch.nn.utils.clip_grad.clip_grad_norm_(multi_period_discriminator.parameters(), max_norm=float('-inf')).item(),
            'multi_scale_discriminator_gradient_norm': torch.nn.utils.clip_grad.clip_grad_norm_(multi_scale_discriminator.parameters(), max_norm=float('-inf')).item(),
            'training_time': engine.state.times['EPOCH_COMPLETED']
        })
    scheduler_g.step()
    scheduler_d.step()
    if args.user_validation:
        validator.run(val_dataloader, max_epochs=1)

trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_manager)
trainer.add_event_handler(Events.COMPLETED, checkpoint_manager)

@validator.on(Events.EPOCH_COMPLETED)
def finish_validating(engine: Engine):
    print(f"Validation: Generator Loss {(engine.state.metrics['generator_loss']):.4f} Discriminator Loss {(engine.state.metrics['discriminator_loss']):.4f}")
    wandb.log({
        'val_generator_loss': engine.state.metrics['generator_loss'], 
        "val_discriminator_loss": engine.state.metrics['discriminator_loss'],
        'early_stopping_patience': early_stopping_handler.counter,
        'val_time': engine.state.times['EPOCH_COMPLETED']
    })

validator.add_event_handler(Events.EPOCH_COMPLETED, early_stopping_handler)

if checkpoint is not None:
    Checkpoint.load_objects(to_load=to_save, checkpoint=torch.load(checkpoint, map_location=device))

if trainer.state.max_epochs is not None:
    args.num_epochs += trainer.state.max_epochs

trainer.run(train_dataloader, max_epochs=args.num_epochs)