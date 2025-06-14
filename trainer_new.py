#!/usr/bin/env python3

import argparse
import json
import os
import shutil
from glob import glob

from torch.nn.utils.rnn import pad_sequence
import h5py
import hdf5plugin
import numpy as np
import safetensors.torch
import torch
import torch.nn.functional as F
import torch.utils.data
from natsort import natsorted
from peft import LoraConfig
from prodigyopt import Prodigy
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core import LightningModule
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist # ADDED: Import torch.distributed for barrier
from pytorch_lightning.callbacks import RichProgressBar

from acestep.pipeline_ace_step import ACEStepPipeline
from acestep.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)

if torch.cuda.is_bf16_supported():
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
# torch._dynamo.config.recompile_limit = 64


def augment_tags(text_token_ids, mask, shuffle, dropout):
    if not shuffle and not dropout:
        return text_token_ids, mask

    COMMA = 275
    bos = text_token_ids[-1:]
    text_token_ids = text_token_ids[:-1]

    tags = []
    start_idx = 0
    _len = len(text_token_ids)
    for idx in range(_len):
        if text_token_ids[idx] == COMMA:
            if start_idx < idx:
                tags.append(text_token_ids[start_idx:idx])
            start_idx = idx + 1
    if start_idx < _len:
        tags.append(text_token_ids[start_idx:_len])

    if shuffle:
        # Shuffle tags using torch's random seed
        perm = torch.randperm(len(tags))
        tags = [tags[i] for i in perm]

    if dropout:
        tags = [x for x in tags if torch.rand(()) > dropout]

    comma = torch.tensor([COMMA], dtype=text_token_ids.dtype)
    tags_and_commas = []
    for x in tags:
        tags_and_commas.append(x)
        tags_and_commas.append(comma)
    if tags_and_commas:
        tags_and_commas[-1] = bos
    else:
        tags_and_commas.append(bos)

    text_token_ids = torch.cat(tags_and_commas)
    mask = mask[: len(text_token_ids)]
    return text_token_ids, mask

def pytree_to_dtype(x, dtype):
    if isinstance(x, list):
        return [pytree_to_dtype(y, dtype) for y in x]
    elif isinstance(x, dict):
        return {k: pytree_to_dtype(v, dtype) for k, v in x.items()}
    elif isinstance(x, torch.Tensor) and x.dtype.is_floating_point:
        return x.to(dtype)
    else:
        return x


class HDF5Dataset(Dataset):
    def __init__(self, dataset_path, dtype, tag_shuffle, tag_dropout):
        self.dataset_path = dataset_path
        self.dtype = dtype
        self.tag_shuffle = tag_shuffle
        self.tag_dropout = tag_dropout

        # --- CORRECTED CODE ---
        # List all items in the directory
        all_items = os.listdir(dataset_path)
        # Filter the list to only include files that end with .h5 or .hdf5
        self.filenames = natsorted([
            f for f in all_items 
            if os.path.isfile(os.path.join(dataset_path, f)) and (f.endswith('.h5') or f.endswith('.hdf5'))
        ])
        # I used natsorted to match your original code's probable intent for numeric sorting

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # The rest of your __getitem__ is correct and doesn't need to be changed.
        file_path = os.path.join(self.dataset_path, self.filenames[idx])
        with h5py.File(file_path, "r") as f:
            # torch.tensor(f[k]) is slow
            sample = {
                k: torch.from_numpy(np.asarray(f[k])) for k in f.keys() if k != "keys"
            }
        sample["text_token_ids"], sample["text_attention_mask"] = augment_tags(
            sample["text_token_ids"],
            sample["text_attention_mask"],
            self.tag_shuffle,
            self.tag_dropout,
        )
        sample["text_attention_mask"] = sample["text_attention_mask"].float()
        sample = pytree_to_dtype(sample, self.dtype)
        return sample


class Pipeline(LightningModule):
    def __init__(
        self,
        # Model
        checkpoint_dir: str = None,
        T: int = 1000,
        shift: float = 3.0,
        timestep_densities_type: str = "logit_normal",
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        lora_config_path: str = None,
        last_lora_path: str = None,
        output_dir: str = "./checkpoints",
        # Data
        dataset_path: str = "./data/your_dataset_path",
        batch_size: int = 1,
        num_workers: int = 0,
        tag_dropout: float = 0.5,
        speaker_dropout: float = 0.0,
        lyrics_dropout: float = 0.0,
        # Optimizer
        ssl_coeff: float = 1.0,
        optimizer: str = "adamw",
        learning_rate: float = 1e-4,
        beta1: float = 0.9,
        beta2: float = 0.99,
        weight_decay: float = 1e-2,
        max_steps: int = 2000,
        warmup_steps: int = 10,
        # Others
        adapter_name: str = "lora_adapter",
        save_last: int = 5,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Moved device and dtype initialization to setup() where self.trainer is available.
        self.to_dtype = None
        self.to_device = None
        self.scheduler = None # Will be initialized in setup()

        # Models will also be initialized in setup()
        self.transformer = None
        self.text_encoder_model = None

    def setup(self, stage: str):
        # Initialize device and dtype based on local_rank
        if torch.cuda.is_bf16_supported():
            self.to_dtype = torch.bfloat16
        else:
            self.to_dtype = torch.float16

        # Use local_rank to assign device for each process
        self.to_device = torch.device(f"cuda:{self.trainer.local_rank}")

        # Initialize scheduler
        self.scheduler = self.get_scheduler()

        # --- CRITICAL FIX FOR DDP MODEL LOADING ---
        # Only global_rank 0 downloads the model, others wait.
        # This prevents all processes from hammering the network simultaneously.
        if self.trainer.global_rank == 0:
            print(f"Rank 0: Initiating model download/load to cache from {self.hparams.checkpoint_dir}")
            # Create a temporary pipeline instance for download/initial load on rank 0
            temp_pipeline = ACEStepPipeline(
                checkpoint_dir=self.hparams.checkpoint_dir,
                device_id=self.trainer.local_rank # Pass local rank, though it's primarily for setup on rank 0
            )
            # This call will internally trigger snapshot_download on rank 0.
            temp_pipeline.load_checkpoint(temp_pipeline.checkpoint_dir)
            del temp_pipeline # Release memory for the temporary instance
        
        # All processes wait here. Rank 0 waits for its download to finish.
        # Other ranks wait for Rank 0 to finish its download, so the cache is populated.
        if self.trainer.world_size > 1: # Only barrier if truly in DDP mode
            dist.barrier() # Use torch.distributed.barrier()

        # Now, all processes can safely load the model from the local (already downloaded/cached) path.
        print(f"Rank {self.trainer.global_rank}: Loading model from local cache...")
        acestep_pipeline = ACEStepPipeline(
            checkpoint_dir=self.hparams.checkpoint_dir,
            device_id=self.trainer.local_rank, # Pass the correct local_rank for this process
            # torch_compile=False # Uncomment and set to False for debugging if issues persist
        )
        # This load_checkpoint call will now find models in the local cache, downloaded by rank 0.
        acestep_pipeline.load_checkpoint(acestep_pipeline.checkpoint_dir)

        self.transformer = acestep_pipeline.ace_step_transformer.to(
            self.to_device, self.to_dtype
        )
        self.transformer.eval()
        self.transformer.requires_grad_(False)
        self.transformer.enable_gradient_checkpointing()

        self.text_encoder_model = acestep_pipeline.text_encoder_model.to(
            self.to_device, self.to_dtype
        )
        self.text_encoder_model.eval()
        self.text_encoder_model.requires_grad_(False)

        # Load LoRA
        assert self.hparams.lora_config_path, "Please provide a LoRA config path"
        with open(self.hparams.lora_config_path, encoding="utf-8") as f:
            lora_config = json.load(f)
        lora_config = LoraConfig(**lora_config)
        self.transformer.add_adapter(
            adapter_config=lora_config, adapter_name=self.hparams.adapter_name
        )

        if self.hparams.last_lora_path:
            state_dict = safetensors.torch.load_file(self.hparams.last_lora_path)
            state_dict = {
                k.replace(".weight", f".{self.hparams.adapter_name}.weight"): v
                for k, v in state_dict.items()
            }
            self.transformer.load_state_dict(state_dict, strict=False)

        # Apply torch.compile after moving to device and setting up LoRA
        for module in self.transformer.projectors:
            module.forward = torch.compile(module.forward, dynamic=True)
        self.transformer.encode = torch.compile(self.transformer.encode, dynamic=True)
        self.text_encoder_model = torch.compile(self.text_encoder_model, dynamic=True)


    def get_scheduler(self):
        return FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=self.hparams.T,
            shift=self.hparams.shift,
        )
    
    def collate_fn(self, batch):
        # Target lengths. We will pad or trim to these exact sizes.
        TARGET_LATENT_LEN = 323
        TARGET_MERT_LEN = 2244

        collated_batch = {}
        all_keys = list(batch[0].keys())

        for key in all_keys:
            tensors = [sample[key] for sample in batch]

            # --- Logic for mert_ssl_hidden_states ---
            if key == "mert_ssl_hidden_states":
                # Shape is (length, features). We need to pad/trim the first dimension (dim=0).
                target_len = TARGET_MERT_LEN
                
                padded_tensors = []
                for tensor in tensors:
                    original_len = tensor.shape[0]
                    if original_len > target_len:
                        # Randomly crop the first dimension (length)
                        start_idx = torch.randint(0, original_len - target_len + 1, (1,)).item()
                        tensor = tensor[start_idx : start_idx + target_len, :]
                    elif original_len < target_len:
                        # Pad the first dimension (length)
                        pad_amount = target_len - original_len
                        # Pad shape is (0, 0, 0, pad_amount) to pad the second-to-last dim (which is dim 0 for a 2D tensor)
                        tensor = F.pad(tensor, (0, 0, 0, pad_amount), "constant", 0)
                    
                    padded_tensors.append(tensor)
                collated_batch[key] = torch.stack(padded_tensors)

            # --- Logic for latents and attention_mask ---
            elif key in ["target_latents", "attention_mask"]:
                # Shape is (..., length). We need to pad/trim the last dimension (dim=-1).
                target_len = TARGET_LATENT_LEN

                padded_tensors = []
                for tensor in tensors:
                    original_len = tensor.shape[-1]
                    if original_len > target_len:
                        # Randomly crop the last dimension (length)
                        start_idx = torch.randint(0, original_len - target_len + 1, (1,)).item()
                        tensor = tensor[..., start_idx : start_idx + target_len]
                    elif original_len < target_len:
                        # Pad the last dimension (length)
                        pad_amount = target_len - original_len
                        tensor = F.pad(tensor, (0, pad_amount), "constant", 0)

                    padded_tensors.append(tensor)
                collated_batch[key] = torch.stack(padded_tensors)

            # --- Logic for Text and Other Tensors ---
            elif key in ["lyric_token_ids", "lyric_mask", "text_token_ids", "text_attention_mask"]:
                collated_batch[key] = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=0)
            
            else: # For speaker_embds, etc.
                collated_batch[key] = torch.stack(tensors)

        return collated_batch

    def configure_optimizers(self):
        trainable_params = [
            p for name, p in self.transformer.named_parameters() if p.requires_grad
        ]

        if self.hparams.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                params=trainable_params,
                lr=self.hparams.learning_rate,
                betas=(self.hparams.beta1, self.hparams.beta2),
                weight_decay=self.hparams.weight_decay,
            )
            lr_multiplier_final = 0
        elif self.hparams.optimizer == "prodigy":
            if self.hparams.learning_rate < 0.1:
                print(
                    "Warning: With Prodigy optimizer, we can usually set learning_rate = 1"
                )
            if self.hparams.warmup_steps > 0:
                print(
                    "Warning: With Prodigy optimizer, we can usually set warmup_steps = 0"
                )
            optimizer = Prodigy(
                params=trainable_params,
                lr=self.hparams.learning_rate,
                betas=(self.hparams.beta1, self.hparams.beta2),
                weight_decay=self.hparams.weight_decay,
                use_bias_correction=True,
                safeguard_warmup=True,
                slice_p=11,
            )
            lr_multiplier_final = 0.1
        else:
            raise ValueError(f"Unknown optimizer: {self.hparams.optimizer}")

        max_steps = self.hparams.max_steps
        warmup_steps = self.hparams.warmup_steps  # New hyperparameter for warmup steps

        # Create a scheduler that first warms up linearly, then decays linearly
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                # Linear warmup from 0 to learning_rate
                return float(current_step) / float(max(1, warmup_steps))
            else:
                # Linear decay from learning_rate to 0
                progress = float(current_step - warmup_steps) / float(
                    max(1, max_steps - warmup_steps)
                )
                multiplier = max(0.0, 1.0 - progress)
                multiplier = (
                    multiplier * (1 - lr_multiplier_final) + lr_multiplier_final
                )
                return multiplier

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda, last_epoch=-1
        )
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

    def train_dataloader(self):
        ds = HDF5Dataset(
            dataset_path=self.hparams.dataset_path,
            dtype=self.to_dtype,
            tag_shuffle=True,
            tag_dropout=self.hparams.tag_dropout,
        )
        return DataLoader(
            ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            # pin_memory=True,
            # persistent_workers=True,
            collate_fn=self.collate_fn,
        )

    def get_sd3_sigmas(self, timesteps, device, n_dim, dtype):
        sigmas = self.scheduler.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = self.scheduler.timesteps.to(device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def get_timestep(self, bsz, device):
        if self.hparams.timestep_densities_type == "logit_normal":
            # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
            # In practice, we sample the random variable u from a normal distribution u ∼ N (u; m, s)
            # and map it through the standard logistic function
            u = torch.normal(
                mean=self.hparams.logit_mean,
                std=self.hparams.logit_std,
                size=(bsz,),
                device="cpu",
            )
            u = torch.nn.functional.sigmoid(u)
            indices = (u * self.scheduler.config.num_train_timesteps).long()
            indices = torch.clamp(
                indices, 0, self.scheduler.config.num_train_timesteps - 1
            )
            timesteps = self.scheduler.timesteps[indices].to(device)
        else:
            raise ValueError(
                f"Unknown timestep_densities_type: {self.hparams.timestep_densities_type}"
            )
        return timesteps

    def run_step(self, batch, batch_idx):
        target_latents = batch["target_latents"]
        attention_mask = batch["attention_mask"]
        text_token_ids = batch["text_token_ids"]
        text_attention_mask = batch["text_attention_mask"]
        speaker_embds = batch["speaker_embds"]
        lyric_token_ids = batch["lyric_token_ids"]
        lyric_mask = batch["lyric_mask"]
        mert_ssl_hidden_states = batch["mert_ssl_hidden_states"]

        target_image = target_latents
        device = self.to_device
        dtype = self.to_dtype

        with torch.no_grad():
            outputs = self.text_encoder_model(
                input_ids=text_token_ids, attention_mask=text_attention_mask
            )
            encoder_text_hidden_states = outputs.last_hidden_state

        if (
            self.hparams.speaker_dropout
            and torch.rand(()) < self.hparams.speaker_dropout
        ):
            speaker_embds = torch.zeros_like(speaker_embds)

        if self.hparams.lyrics_dropout and torch.rand(()) < self.hparams.lyrics_dropout:
            lyric_token_ids = torch.zeros_like(lyric_token_ids)
            lyric_mask = torch.zeros_like(lyric_mask)

        # Step 1: Generate random noise, initialize settings
        noise = torch.randn_like(target_image)
        bsz = target_image.shape[0]
        timesteps = self.get_timestep(bsz, device)

        # Add noise according to flow matching.
        sigmas = self.get_sd3_sigmas(
            timesteps=timesteps, device=device, n_dim=target_image.ndim, dtype=dtype
        )
        noisy_image = sigmas * noise + (1.0 - sigmas) * target_image

        # This is the flow-matching target for vanilla SD3.
        target = target_image

        # SSL constraints for CLAP and vocal_latent_channel2
        all_ssl_hiden_states = []
        if mert_ssl_hidden_states is not None:
            all_ssl_hiden_states.append(mert_ssl_hidden_states)

        # N x H -> N x c x W x H
        x = noisy_image
        # Step 5: Predict noise
        transformer_output = self.transformer(
            hidden_states=x,
            attention_mask=attention_mask,
            encoder_text_hidden_states=encoder_text_hidden_states,
            text_attention_mask=text_attention_mask,
            speaker_embeds=speaker_embds,
            lyric_token_idx=lyric_token_ids,
            lyric_mask=lyric_mask,
            timestep=timesteps.to(device, dtype),
            ssl_hidden_states=all_ssl_hiden_states,
        )
        model_pred = transformer_output.sample
        proj_losses = transformer_output.proj_losses

        # Follow: Section 5 of https://arxiv.org/abs/2206.00364.
        # Preconditioning of the model outputs.
        model_pred = model_pred * (-sigmas) + noisy_image

        # Compute loss. Only calculate loss where chunk_mask is 1 and there is no padding
        # N x T x 64
        # N x T -> N x c x W x T
        mask = (
            attention_mask.unsqueeze(1)
            .unsqueeze(1)
            .expand(-1, target_image.shape[1], target_image.shape[2], -1)
        )

        # TODO: Check if the masked mean is correct
        selected_model_pred = (model_pred * mask).reshape(bsz, -1).contiguous()
        selected_target = (target * mask).reshape(bsz, -1).contiguous()

        loss = F.mse_loss(selected_model_pred, selected_target, reduction="none")
        loss = loss.mean(1)
        loss = loss * mask.reshape(bsz, -1).mean(1)
        loss = loss.mean()

        prefix = "train"

        self.log(f"{prefix}/denoising_loss", loss, on_step=True, on_epoch=False)

        total_proj_loss = 0.0
        for k, v in proj_losses:
            self.log(f"{prefix}/{k}_loss", v, on_step=True, on_epoch=False)
            total_proj_loss += v

        if len(proj_losses) > 0:
            total_proj_loss = total_proj_loss / len(proj_losses)

        if self.hparams.ssl_coeff:
            loss += total_proj_loss * self.hparams.ssl_coeff
        self.log(f"{prefix}/loss", loss, on_step=True, on_epoch=False)

        # Log learning rate if scheduler exists
        if self.lr_schedulers() is not None:
            learning_rate = self.lr_schedulers().get_last_lr()[0]
            self.log(
                f"{prefix}/learning_rate", learning_rate, on_step=True, on_epoch=False
            )

        if self.hparams.optimizer == "prodigy":
            prodigy_d = self.optimizers().param_groups[0]["d"]
            self.log(f"{prefix}/prodigy_d", prodigy_d, on_step=True, on_epoch=False)

        return loss

    def training_step(self, batch, batch_idx):
        return self.run_step(batch, batch_idx)

    def on_save_checkpoint(self, checkpoint):
        if self.trainer.global_rank != 0:
            return 

        checkpoint_dir = self.hparams.output_dir 

        epoch = self.current_epoch
        step = self.global_step
        lora_name = f"epoch={epoch}-step={step}_lora"
        lora_path = os.path.join(checkpoint_dir, lora_name)
        os.makedirs(lora_path, exist_ok=True)
        self.transformer.save_lora_adapter(
            lora_path, adapter_name=self.hparams.adapter_name
        )

        # Clean up old loras and only save the last few loras
        lora_paths = glob(os.path.join(checkpoint_dir, "*_lora"))
        lora_paths = natsorted(lora_paths)
        if len(lora_paths) > self.hparams.save_last:
            shutil.rmtree(lora_paths[0])

        # Don't save the full model
        checkpoint.clear()
        return checkpoint


def main(args):
    model = Pipeline(
        output_dir=args.output_dir,
        # Model
        checkpoint_dir=args.checkpoint_dir,
        shift=args.shift,
        lora_config_path=args.lora_config_path,
        last_lora_path=args.last_lora_path,
        # Data
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        tag_dropout=args.tag_dropout,
        speaker_dropout=args.speaker_dropout,
        lyrics_dropout=args.lyrics_dropout,
        # Optimizer
        ssl_coeff=args.ssl_coeff,
        optimizer=args.optimizer,
        learning_rate=args.learning_rate,
        beta1=args.beta1,
        beta2=args.beta2,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        # Others
        adapter_name=args.exp_name,
        save_last=args.save_last,
    )
    
    os.environ["WANDB_DIR"] = "./gitignore"
    
    # --- STEP 1: Uncomment the logger ---
    logger_callback = WandbLogger(
        project="ace_step_lora",
        name=args.exp_name,
    )

    # Define all callbacks you want to use
    checkpoint_callback = ModelCheckpoint(
        every_n_train_steps=args.save_every_n_train_steps,
    )
    progress_bar = RichProgressBar()

    trainer = Trainer(
        accelerator="gpu",
        strategy="ddp",
        devices=args.devices,
        precision=args.precision,
        log_every_n_steps=1,
        
        # --- STEP 2: Add the logger to the Trainer ---
        logger=logger_callback,
        
        # Combine all callbacks into a single list
        callbacks=[checkpoint_callback, progress_bar],
        max_epochs=args.epochs,
        max_steps=args.max_steps,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        gradient_clip_algorithm=args.gradient_clip_algorithm,
    )

    trainer.fit(model)


if __name__ == "__main__":
    args = argparse.ArgumentParser()

    args.add_argument("--output_dir", type=str, default=None)
    
    # Model
    args.add_argument("--checkpoint_dir", type=str, default=None)
    args.add_argument("--shift", type=float, default=3.0)
    args.add_argument("--lora_config_path", type=str, default="./config/lora_config_transformer_only.json")
    args.add_argument("--last_lora_path", type=str, default=None)

    # Data
    args.add_argument("--dataset_path", type=str, default="hdf5")
    args.add_argument("--batch_size", type=int, default=8)
    args.add_argument("--num_workers", type=int, default=0)
    args.add_argument("--tag_dropout", type=float, default=0.1)
    args.add_argument("--speaker_dropout", type=float, default=0.1)
    args.add_argument("--lyrics_dropout", type=float, default=0.0)

    # Optimizer
    args.add_argument("--ssl_coeff", type=float, default=1.0)
    args.add_argument("--optimizer", type=str, default="adamw")
    args.add_argument("--learning_rate", type=float, default=0.001)
    args.add_argument("--beta1", type=float, default=0.9)
    args.add_argument("--beta2", type=float, default=0.99)
    args.add_argument("--epochs", type=int, default=-1)
    args.add_argument("--max_steps", type=int, default=10000)
    args.add_argument("--warmup_steps", type=int, default=100)
    args.add_argument("--accumulate_grad_batches", type=int, default=1)
    args.add_argument("--gradient_clip_val", type=float, default=0.0)
    args.add_argument("--gradient_clip_algorithm", type=str, default="norm")

    # Others
    args.add_argument("--devices", type=int, default=5)
    # args.add_argument("--num_nodes", type=int, default=1)
    args.add_argument("--exp_name", type=str, default="ace_step_lora")
    args.add_argument("--precision", type=str, default="bf16-mixed")
    args.add_argument("--save_every_n_train_steps", type=int, default=100)
    args.add_argument("--save_last", type=int, default=99999)

    args = args.parse_args()
    main(args)