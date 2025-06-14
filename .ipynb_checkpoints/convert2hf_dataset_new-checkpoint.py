#!/usr/bin/env python3

import argparse
import os

from datasets import Dataset


def create_dataset(data_dir, repeat_count, output_name, default_tag):
    # Formats supported by torchaudio
    extensions = {
        ".aac",
        ".flac",
        ".m4a",
        ".mp3",
        ".ogg",
        ".wav",
    }

    all_examples = []
    for file in sorted(os.listdir(data_dir)):
        stem, ext = os.path.splitext(file)
        if ext.lower() not in extensions:
            continue

        file_path = os.path.join(data_dir, file)
        stem_path = os.path.join(data_dir, stem)
        prompt_path = stem_path + "_prompt.txt"
        lyric_path = stem_path + "_lyrics.txt"

        if os.path.exists(prompt_path):
            with open(prompt_path, "r", encoding="utf-8") as f:
                prompt = f.read().strip()
        else:
            prompt = default_tag

        if os.path.exists(lyric_path):
            with open(lyric_path, "r", encoding="utf-8") as f:
                lyrics = f.read().strip()
        else:
            lyrics = "[instrumental]"

        tags = [x.strip() for x in prompt.split(",")]
        example = {
            "keys": stem,
            "filename": file_path,
            "tags": tags,
            "speaker_emb_path": "",
            "norm_lyrics": lyrics,
            "recaption": {},
        }
        all_examples.append(example)

    # repeat specified times
    ds = Dataset.from_list(all_examples * repeat_count)
    ds.save_to_disk(output_name)


def main():
    parser = argparse.ArgumentParser(description="Create a dataset from audio files.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=r"C:\data\audio",
        help="Directory containing the audio files.",
    )
    parser.add_argument(
        "--repeat_count",
        type=int,
        default=1,
        help="Number of times to repeat the dataset.",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default=r"C:\data\audio_filenames",
        help="Name of the output dataset.",
    )
    parser.add_argument(
        "--default_tag",
        type=str,
        default="",
        help="Default tag when there is no prompt file.",
    )
    args = parser.parse_args()

    create_dataset(
        data_dir=args.data_dir,
        repeat_count=args.repeat_count,
        output_name=args.output_name,
        default_tag=args.default_tag,
    )


if __name__ == "__main__":
    main()
