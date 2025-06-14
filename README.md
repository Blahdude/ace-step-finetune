# [ACE-Step](https://github.com/ace-step/ACE-Step) fork

## Progress

* Separate data preprocessing (music and text encoding) and training
* Enable gradient checkpointing
* Cast everything to bf16

Now I can run the training on a single RTX 3080 with < 10 GB VRAM and 0.3 it/s speed, using music duration < 360 seconds and LoRA rank = 64.

I've trained some LoRAs at https://huggingface.co/woctordho/ACE-Step-v1-LoRA-collection

## Usage

1. Collect some audios, for example, in the directory `C:\data\audio`.

2. Generate prompts using Qwen2.5-Omni-7B:
    ```pwsh
    python generate_prompts_lyrics.py --data_dir C:\data\audio
    ```
    Each prompt is a list of tags separated by comma space `, ` without EOL. The order of tags will be randomly shuffled in the training. (TODO: Check how natural language prompts affect the performance.)

    **(Experimental)** The above script uses gptqmodel. Alternatively, you can use llama.cpp:
    <details>
    <summary>Expand</summary>

    Start llama-server (by default it listens host 127.0.0.1, port 8080)
    ```pwsh
    llama-server -m Qwen2.5-Omni-7B-Q8_0.gguf --mmproj mmproj-Qwen2.5-Omni-7B-Q8_0.gguf -c 32768 -fa -ngl 999 --cache-reuse 256
    ```
    Then run
    ```pwsh
    python generate_prompts_lyrics_llamacpp.py --data_dir C:\data\audio
    ```
    After this step, you can shut down llama-server to save VRAM.

    Unfortunately, for now llama.cpp did not reproduce the original model with enough accuracy, so tags may not be accurate and lyrics almost does not work at all.
    </details>

    **(Experimental)** You can also generate lyrics:
    <details>
    <summary>Expand</summary>

    ```pwsh
    python generate_prompts_lyrics.py --data_dir C:\data\audio --lyrics
    ```
    It seems Qwen2.5-Omni-7B works well for Chinese lyrics, but not so well for English and other languages.
    </details>

    TODO: Besides using an AI model to transcribe lyrics, we can also extract lyrics embedded in the audio file, or query online databases such as [163MusicLyrics](https://github.com/jitwxs/163MusicLyrics), [LyricsGenius](https://github.com/johnwmillr/LyricsGenius), [LyricWiki](https://archive.org/details/lyricsfandomcom-20200216-patched.7z).

    For music without vocal, just use `[instrumental]` for the lyrics.

    At this point, the directory `C:\data\audio` should be like:
    ```
    audio1.wav
    audio1_lyrics.txt
    audio1_prompt.txt
    audio2.mp3
    audio2_lyrics.txt
    audio2_prompt.txt
    ...
    ```

4. Create a dataset that only contains the filenames, not the audio data:
    ```pwsh
    python convert2hf_dataset_new.py --data_dir C:\data\audio --output_name C:\data\audio_filenames
    ```

5. Load the audios, do the preprocessing, save to a new dataset:
    ```pwsh
    python preprocess_dataset_new.py --input_name C:\data\audio_filenames --output_dir C:\data\audio_prep
    ```
    The preprocessed dataset takes ~0.2 MB for every second of input audio.

    TODO: If you have a lot of training data and want to reduce disk space requirement, we can add a switch to move MERT and mHuBERT from preprocessing to training.

7. Do the training:
    ```pwsh
    python trainer_new.py --dataset_path C:\data\audio_prep
    ```
    The LoRA will be saved to the directory `checkpoints`. Make sure to clear this directory before training, otherwise the LoRA may not be correctly saved.

    If you have a lot of VRAM, you can remove `self.transformer.enable_gradient_checkpointing()` for faster training speed.

    My script uses Wandb rather than TensorBoard. If you don't need it, you can remove the `WandbLogger`.

9. LoRA strength:

    At this point, when loading the LoRA in ComfyUI, you need to set the LoRA strength to `alpha / sqrt(rank)` (for rsLoRA) or `alpha / rank` (for non-rsLoRA). For example, if rank = 64, alpha = 1, rsLoRA is enabled, then the LoRA strength should be `1 / sqrt(64) = 0.125`.

    To avoid manually setting this, you can run:
    ```pwsh
    python add_alpha_in_lora.py --input_name checkpoints/epoch=0-step=100_lora/pytorch_lora_weights.safetensors --output_name out.safetensors --lora_config_path config/lora_config_transformer_only.json
    ```
    Then load `out.safetensors` in ComfyUI and set the LoRA strength to 1.

## Tips

* If you don't have experience, you can first try to train with a single audio and make sure that it can be overfitted. This is a sanity check of the training pipeline
* You can freeze the lyrics decoder and only train the transformer using `config/lora_config_transformer_only.json`. I think training the lyrics decoder is needed only when adding a new language
* In the LoRA config, you can add
    ```
    "projectors.0.0",
    "projectors.0.2",
    "projectors.0.4",
    "projectors.1.0",
    "projectors.1.2",
    "projectors.1.4",
    ```
    to `target_modules`. This may help the model learn the music style
* When using an Adam-like optimizer (including AdamW and Prodigy), you should not let `1 - beta2` be much smaller than `1 / max_steps`
* When using Prodigy optimizer, make sure that `d` rises to a large value (such as 1e-4, should be much larger than the initial 1e-6) after `1 / (1 - beta2)` steps
* After training, you can prune the LoRA using SVD. This can be done with Kohya's `resize_lora.py` after applying [this patch](https://github.com/kohya-ss/sd-scripts/pull/2057). If the dynamic pruning tells you that the LoRA rank can be much smaller without changing the output quality, then next time you can train the LoRA using a smaller rank

## TODO

* Support batch size > 1, maybe bucketing samples with similar lengths
* How to normalize the audio loudness before preprocessing? It seems the audios generated by ACE-Step usually have loudness in -16 .. -12 LUFS, and they don't follow prompts like 'loud' and 'quiet'
* To generate the tags, maybe a specialized tagger can perform better than Qwen2.5-Omni-7B, such as [OpenJMLA](https://huggingface.co/UniMus/OpenJMLA)
    * The statistics of the tags used to train the base model is shared on [Discord](https://discord.com/channels/1369256267645849741/1372633881215500429/1374037211145830442)
* When an audio is cropped because it's too long, also crop the lyrics
* I would not include BPM in the AI-generated tags, because it's much more accurate to detect BPM using traditional methods than AI. Also, to control the BPM of the generated audio, I guess it's more adhesive to use a control net than the prompt, similar to the Canny control net for images.
