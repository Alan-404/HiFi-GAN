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

wandb.init(project="HiFi_GAN - Text to Speech", name='trind18')

lr = 3e-4
device = 'cuda'
checkpoint_dir = "./checkpoints"
checkpoint_step = None
checkpoint = None
num_epochs = 1
tensorboard_dir = "./log_dir"

save_step = 15000
scheduler_step = 15000

processor = HiFiGANProcessor()

generator = Generator(
    n_mel_channels=processor.n_mel_channels,
    upsample_rates=[8,8,2,2],
    upsample_kernel_sizes=[16,16,4,4],
    upsample_initial_channel=512,
    resblock_kernel_sizes=[3,7,11],
    resblock_dilation_sizes=[[1,3,5], [1,3,5], [1,3,5]]
)
multi_period_discriminator = MultiPeriodDiscriminator()
multi_scale_discriminator = MultiScaleDiscriminator()

optimizer_g = optim.Adam(params=generator.parameters(), lr=lr)
optimizer_d = optim.Adam(params=itertools.chain(
                multi_period_discriminator.parameters(), multi_scale_discriminator.parameters()
        ), lr=lr)

scheduler_g = lr_scheduler.CosineAnnealingLR(optimizer=optimizer_g, T_max=500000)
scheduler_d = lr_scheduler.CosineAnnealingLR(optimizer=optimizer_d, T_max=500000)

def get_batch(batch: list):
    mels, signals = processor(batch)
    return mels, signals

train_dataset = HiFiGANDataset(manifest_path="", processor=processor, num_examples=None)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True, collate_fn=get_batch)

val_dataset = HiFiGANDataset(manifest_path="", processor=processor, num_examples=None)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=True, collate_fn=get_batch)

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
early_stopping_handler = EarlyStopping(patience=10, score_function=early_stopping_condition, trainer=trainer)

to_save = {
    'generator': generator,
    'multi_period_discriminator': multi_period_discriminator,
    'multi_scale_discriminator': multi_scale_discriminator,
    'generator_optimizer': optimizer_g,
    'discriminator_optimizer': optimizer_d,
    'trainer': trainer
}
checkpoint_manager = Checkpoint(to_save, save_handler=DiskSaver(dirname=checkpoint_dir, create_dir=True, require_empty=False), n_saved=3)

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
    wandb.log({'train_generator_loss': engine.state.metrics['generator_loss'], "train_discriminator_loss": engine.state.metrics['discriminator_loss']})
    validator.run(val_dataloader, max_epochs=1)

@trainer.on(Events.ITERATION_COMPLETED(every=(scheduler_step)))
def update_scheduler(engine: Engine):
    scheduler_d.step()
    scheduler_g.step()

trainer.add_event_handler(Events.ITERATION_COMPLETED(event_filter=save_step), checkpoint_manager)

trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_manager)

@validator.on(Events.EPOCH_COMPLETED)
def finish_validating(engine: Engine):
    print(f"Validation: Generator Loss {(engine.state.metrics['generator_loss']):.4f} Discriminator Loss {(engine.state.metrics['discriminator_loss']):.4f}")
    wandb.log({'val_generator_loss': engine.state.metrics['generator_loss'], "val_discriminator_loss": engine.state.metrics['discriminator_loss']})

validator.add_event_handler(Events.EPOCH_COMPLETED, early_stopping_handler)

if checkpoint_step is not None and os.listdir(checkpoint_dir) != 0 and os.path.exists(checkpoint):
    Checkpoint.load_objects(to_load=to_save, checkpoint=torch.load(checkpoint))

if trainer.state.max_epochs is not None:
    num_epochs += trainer.state.max_epochs

trainer.run(train_dataloader, max_epochs=num_epochs)