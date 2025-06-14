import click
import os
import json

from acestep.pipeline_ace_step import ACEStepPipeline
from acestep.data_sampler import DataSampler # Assuming this class primarily finds the json path

@click.command()
@click.option(
    "--checkpoint_path", 
    type=str, 
    default=None, 
    help="Path to the base model checkpoint directory"
)
@click.option(
    "--json_path",
    type=str,
    default="acestep/infer_params/infer.json", # Default path to your JSON file
    help="Path to the JSON file containing inference parameters."
)
@click.option("--bf16", type=bool, default=True, help="Whether to use bfloat16")
@click.option(
    "--torch_compile", type=bool, default=False, help="Whether to use torch compile"
)
@click.option(
    "--cpu_offload", type=bool, default=False, help="Whether to use CPU offloading"
)
@click.option(
    "--overlapped_decode", type=bool, default=False, help="Whether to use overlapped decoding"
)
@click.option("--device_id", type=int, default=0, help="Device ID to use")
@click.option("--output_path", type=str, default=None, help="Path to save the output")
def main(
    checkpoint_path,
    json_path,
    bf16,
    torch_compile,
    cpu_offload,
    overlapped_decode,
    device_id,
    output_path,
):
    """
    This script loads all inference parameters from a JSON file and runs the model.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

    # --- Step 1: Load all parameters from the JSON file ---
    try:
        with open(json_path, 'r') as f:
            params = json.load(f)
        logger.info(f"Successfully loaded inference parameters from: {json_path}")
    except FileNotFoundError:
        logger.error(f"FATAL: The JSON file was not found at {json_path}")
        return
    except json.JSONDecodeError:
        logger.error(f"FATAL: The JSON file at {json_path} is not formatted correctly.")
        return

    # --- Step 2: Initialize the pipeline ---
    model_demo = ACEStepPipeline(
        checkpoint_dir=checkpoint_path,
        dtype="bfloat16" if bf16 else "float32",
        torch_compile=torch_compile,
        cpu_offload=cpu_offload,
        overlapped_decode=overlapped_decode,
    )

    # --- Step 3: Call the model with all parameters from the JSON file ---
    logger.info(f"Starting inference with LoRA: '{params.get('lora_name_or_path', 'None')}' and weight: {params.get('lora_weight', 1.0)}")

    # The 'params' dictionary is passed directly using the **kwargs syntax.
    # This automatically maps keys in your JSON to the arguments of the __call__ method.
    model_demo(
        save_path=output_path,
        **params 
    )

if __name__ == "__main__":
    from loguru import logger # Add logger for better messages
    main()