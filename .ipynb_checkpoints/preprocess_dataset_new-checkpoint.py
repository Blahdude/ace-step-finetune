#!/usr/bin/env python3

import argparse
import os

import h5py
import hdf5plugin
import torch
import torch.nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, Wav2Vec2FeatureExtractor

from acestep.pipeline_ace_step import ACEStepPipeline
from acestep.text2music_dataset import Text2MusicDataset

if torch.cuda.is_bf16_supported():
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

class Preprocessor(torch.nn.Module):
    """
    A class to preprocess audio files and their corresponding text prompts,
    extracting features like audio latents, text embeddings, and MERT SSL features.
    This version processes entire files at once, without chunking.
    """
    def __init__(self, checkpoint_dir=None):
        super().__init__()

        if torch.cuda.is_bf16_supported():
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float16
        self.device = torch.device("cuda:0")

        # Initialize ACEStep pipeline components
        acestep_pipeline = ACEStepPipeline(checkpoint_dir)
        acestep_pipeline.load_checkpoint(acestep_pipeline.checkpoint_dir)
        self.dcae = acestep_pipeline.music_dcae
        self.dcae.dcae.encoder = torch.compile(self.dcae.dcae.encoder, dynamic=True)
        self.text_tokenizer = acestep_pipeline.text_tokenizer
        del acestep_pipeline

        # Initialize MERT model for SSL feature extraction
        self.mert_model = AutoModel.from_pretrained(
            "m-a-p/MERT-v1-330M", trust_remote_code=True, cache_dir=checkpoint_dir
        )
        self.resampler_mert = torchaudio.transforms.Resample(
            orig_freq=48000, new_freq=24000
        )
        # Note: The Wav2Vec2FeatureExtractor is loaded but not explicitly used in the SSL inference,
        # as the MERT model handles raw waveform inputs after manual normalization.
        self.processor_mert = Wav2Vec2FeatureExtractor.from_pretrained(
            "m-a-p/MERT-v1-330M", trust_remote_code=True, cache_dir=checkpoint_dir
        )

        self.to(self.device, self.dtype)
        self.eval()

    def infer_mert_ssl(self, target_wavs, wav_lengths):
        """
        Infers MERT SSL features for a batch of audio waveforms.
        CHANGE: This function now processes the entire audio waveform at once,
        removing the previous internal chunking mechanism. This may require
        significant GPU memory for very long audio files.
        """
        # Resample to 24kHz mono for the MERT model
        mert_input_wavs_mono_24k = self.resampler_mert(target_wavs.mean(dim=1))
        bsz = target_wavs.shape[0]
        actual_lengths_24k = wav_lengths // 2

        # Normalize each waveform individually
        means = torch.stack(
            [
                mert_input_wavs_mono_24k[i, : actual_lengths_24k[i]].mean()
                for i in range(bsz)
            ]
        )
        _vars = torch.stack(
            [
                mert_input_wavs_mono_24k[i, : actual_lengths_24k[i]].var()
                for i in range(bsz)
            ]
        )
        mert_input_wavs_mono_24k = (
            mert_input_wavs_mono_24k - means.view(-1, 1)
        ) / torch.sqrt(_vars.view(-1, 1) + 1e-7)

        # Process the entire audio file without chunking
        with torch.no_grad():
            mert_ssl_hidden_states = self.mert_model(
                mert_input_wavs_mono_24k
            ).last_hidden_state

        # Trim padding from the hidden states based on original audio length
        # MERT model has a downsampling factor of 320
        num_features = (actual_lengths_24k + 319) // 320
        
        mert_ssl_hidden_states_list = [
            mert_ssl_hidden_states[i, :num_features[i], :] for i in range(bsz)
        ]

        return mert_ssl_hidden_states_list

    def get_text_embeddings(self, texts, text_max_length=256):
        """
        Tokenizes text prompts and returns input IDs and attention masks.
        """
        inputs = self.text_tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=text_max_length,
        )
        return inputs["input_ids"], inputs["attention_mask"]

    def process(self, batch):
        """
        Processes a single file (as a batch of 1) to extract all features.
        """
        dtype = self.dtype
        device = self.device

        target_wavs = batch["target_wavs"].to(device, dtype)
        wav_lengths = batch["wav_lengths"].to(device)
        
        # SSL constraints (output list will have 1 item for batch size 1)
        mert_ssl_hidden_states = self.infer_mert_ssl(target_wavs, wav_lengths)[0].float().cpu()

        # Text embeddings
        texts = batch["prompts"]
        text_token_ids, text_attention_mask = self.get_text_embeddings(texts)
        text_attention_mask = text_attention_mask.float()

        # Encode the audio to latents
        target_latents, _ = self.dcae.encode(target_wavs, wav_lengths)
        target_latents = target_latents.float().cpu()[0] # Get the single item from the batch
        attention_mask = torch.ones(target_latents.shape[-1])

        # Return a dictionary for the single processed file
        return {
            "target_latents": target_latents,
            "attention_mask": attention_mask,
            "text_token_ids": text_token_ids[0],
            "text_attention_mask": text_attention_mask[0],
            "speaker_embds": batch["speaker_embs"][0],
            "lyric_token_ids": batch["lyric_token_ids"][0],
            "lyric_mask": batch["lyric_masks"][0],
            "mert_ssl_hidden_states": mert_ssl_hidden_states,
        }

def save_file(out_path, sample):
    """
    Saves a dictionary of processed features to an HDF5 file.
    """
    with h5py.File(out_path, "w") as f:
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                f.create_dataset(k, data=v.numpy(), compression=hdf5plugin.Zstd(), shuffle=True)
            else:
                f.create_dataset(k, data=v)


def do_files(input_name, output_dir, checkpoint_dir):
    """
    Main processing loop. Iterates through the dataset and processes each file.
    CHANGE: The file-based chunking loop has been removed. Each audio file is
    now processed in its entirety and saved to a single HDF5 file.
    """
    ds = Text2MusicDataset(
        train_dataset_path=input_name,
    )
    # Use batch_size=1 to process one full file at a time
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=ds.collate_fn)
    
    prep = Preprocessor(checkpoint_dir)
    os.makedirs(output_dir, exist_ok=True)

    for batch in tqdm(dl, desc="Processing files"):
        # Each 'batch' contains one full audio file from the dataset
        stem = batch["keys"][0]
        
        # Define the output path for the processed file
        out_path = os.path.join(output_dir, f"{stem}.hdf5")
        if os.path.exists(out_path):
            continue
            
        # Preprocess the full audio file
        processed_file = prep.process(batch)

        # Save the processed data to its HDF5 file
        save_file(out_path, processed_file)


@torch.inference_mode
def main():
    """
    Parses command-line arguments and starts the preprocessing.
    """
    parser = argparse.ArgumentParser(description="Preprocess audio files for the ACEStep model without chunking.")
    parser.add_argument("--input_name", type=str, required=True, help="The path to the text file containing audio filenames.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the preprocessed HDF5 files.")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory to cache model checkpoints.")
    args = parser.parse_args()

    do_files(
        input_name=args.input_name,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()
