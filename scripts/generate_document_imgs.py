import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch
from torchvision.transforms import ToPILImage
import argparse
from tqdm import tqdm
from loc_utils.capture_outputs import capture_all_output
import json

# from loc_utils.crop_faces import FaceCropper


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate document-like images from given prompts.", add_help=False
    )  # Disable default help to avoid conflicts with ComfyUI's help

    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--resolution", type=int, default=1024, help="Resolution of the output images.")

    # Use `parse_known_args()` to separate unknown arguments (ComfyUI's args)
    args, remaining_args = parser.parse_known_args()

    return args, remaining_args


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.insert(0, comfyui_path)  # Insert at the beginning to prioritize it
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    try:
        from ComfyUI.main import load_extra_path_config
    except ImportError:
        print("Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead.")
        from ComfyUI.utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import ComfyUI.execution as execution
    from ComfyUI.nodes import init_extra_nodes
    import ComfyUI.server as server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_extra_nodes()


def generate_image_for_json(
    json_file_path,
    cliptextencode,
    fluxguidance,
    ksampler,
    vaedecode,
    loraloader_43,
    cliptextencode_33,
    emptysd3latentimage_27,
    vaeloader_38,
):
    """Generate and save one image for a single JSON prompt file.

    Returns True if an image was generated, False if it was skipped (already exists).
    """
    output_image_path = os.path.join(
        os.path.dirname(json_file_path),
        os.path.splitext(os.path.basename(json_file_path))[0] + "_doc.png",
    )

    # Skip if image already exists
    if os.path.exists(output_image_path):
        return False

    # Read the existing JSON
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Generate a random seed and guidance value
    random_seed = random.randint(1, 2**64)
    random_guidance = round(random.uniform(1.7, 2.5), 1)

    # Persist seed + guidance back into the JSON so runs are reproducible
    data["seed"] = random_seed
    data["guidance"] = random_guidance
    with open(json_file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    caption = data["prompt"].strip()

    # Encode positive prompt
    cliptextencode_6 = cliptextencode.encode(text=caption, clip=get_value_at_index(loraloader_43, 1))

    fluxguidance_35 = fluxguidance.append(
        guidance=random_guidance, conditioning=get_value_at_index(cliptextencode_6, 0)
    )

    # Run diffusion
    ksampler_31 = ksampler.sample(
        seed=random_seed,
        steps=20,
        cfg=1,
        sampler_name="dpmpp_2m",
        scheduler="beta",
        denoise=1,
        model=get_value_at_index(loraloader_43, 0),
        positive=get_value_at_index(fluxguidance_35, 0),
        negative=get_value_at_index(cliptextencode_33, 0),
        latent_image=get_value_at_index(emptysd3latentimage_27, 0),
    )

    # Decode
    vaedecode_8 = vaedecode.decode(
        samples=get_value_at_index(ksampler_31, 0),
        vae=get_value_at_index(vaeloader_38, 0),
    )

    pil_img = ToPILImage()(get_value_at_index(vaedecode_8, 0).squeeze(0).permute(2, 0, 1))
    pil_img.save(output_image_path)
    return True


def main(dataset_dir, resolution=1024):
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory '{dataset_dir}' does not exist.")

    # face_detector = FaceCropper()

    import_custom_nodes()
    with torch.inference_mode():
        unetloader = NODE_CLASS_MAPPINGS["UNETLoader"]()
        unetloader_37 = unetloader.load_unet(unet_name="flux1-dev-fp8.safetensors", weight_dtype="fp8_e4m3fn_fast")

        dualcliploader = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
        dualcliploader_39 = dualcliploader.load_clip(
            clip_name1="clip_l.safetensors",
            clip_name2="t5xxl_fp16.safetensors",
            type="flux",
        )

        loraloader = NODE_CLASS_MAPPINGS["LoraLoader"]()
        loraloader_43 = loraloader.load_lora(
            lora_name="cfd-lora.safetensors",
            strength_model=1,
            strength_clip=1,
            model=get_value_at_index(unetloader_37, 0),
            clip=get_value_at_index(dualcliploader_39, 0),
        )

        cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
        # Encode negative prompt once (unused by FLUX but kept for API compatibility)
        cliptextencode_33 = cliptextencode.encode(text="", clip=get_value_at_index(loraloader_43, 1))

        emptysd3latentimage = NODE_CLASS_MAPPINGS["EmptySD3LatentImage"]()
        emptysd3latentimage_27 = emptysd3latentimage.generate(width=resolution, height=resolution, batch_size=1)

        vaeloader = NODE_CLASS_MAPPINGS["VAELoader"]()
        vaeloader_38 = vaeloader.load_vae(vae_name="ae.safetensors")

        fluxguidance = NODE_CLASS_MAPPINGS["FluxGuidance"]()
        ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
        vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()

        print(
            "WARNING: All outputs are now being captured, including errors. "
            "If issues arise, set capture_all_output(disable=True) for debugging."
        )

        # Collect all (folder, json_file) pairs upfront so the progress bar
        # reflects total images rather than total identity folders.
        all_jobs = []
        for folder in sorted(os.listdir(dataset_dir)):
            folder_path = os.path.join(dataset_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            json_files = sorted(f for f in os.listdir(folder_path) if f.endswith(".json"))
            if not json_files:
                print(f"Warning: no JSON files found in '{folder_path}', skipping.")
                continue
            for jf in json_files:
                all_jobs.append(os.path.join(folder_path, jf))

        print(f"Found {len(all_jobs)} prompt(s) across {len(os.listdir(dataset_dir))} identity folder(s).")

        for json_file_path in tqdm(all_jobs, desc="Generating images"):
            # NOTE: captures all output including errors; set disable=True to debug
            with capture_all_output(disable=False):
                generate_image_for_json(
                    json_file_path=json_file_path,
                    cliptextencode=cliptextencode,
                    fluxguidance=fluxguidance,
                    ksampler=ksampler,
                    vaedecode=vaedecode,
                    loraloader_43=loraloader_43,
                    cliptextencode_33=cliptextencode_33,
                    emptysd3latentimage_27=emptysd3latentimage_27,
                    vaeloader_38=vaeloader_38,
                )


if __name__ == "__main__":
    args, remaining_args = parse_args()

    # Merge back the remaining arguments into `sys.argv`
    sys.argv = [sys.argv[0]] + remaining_args  # Restore unknown args for ComfyUI

    add_comfyui_directory_to_sys_path()
    add_extra_model_paths()

    from ComfyUI.nodes import NODE_CLASS_MAPPINGS

    main(dataset_dir=args.dataset_dir, resolution=args.resolution)