import os
os.environ["HF_HOME"] = "./models/huggingface"

import json
import random
import torch
import copy
import inflect
from tqdm import tqdm
import argparse
import uuid
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch import inference_mode


def parse_args():
    parser = argparse.ArgumentParser(description="Generate prompts with specified attributes.")

    parser.add_argument("--dataset_dir", type=str, required=True, help="Directory where the dataset will be saved.")
    parser.add_argument("--number_of_prompts", type=int, required=True, help="Number of prompts to generate.")
    parser.add_argument(
        "--file_prob_txt",
        type=str,
        default="attributes/attributes/file_probabilities.json",
        help="Path to file probabilities JSON.",
    )
    parser.add_argument(
        "--attribute_clash_txt",
        type=str,
        default="attributes/attributes/attribute_clashes.json",
        help="Path to attribute clashes JSON.",
    )

    # Speed knobs
    parser.add_argument("--batch_size", type=int, default=64, help="LLM batch size (A100: try 64/128).")
    parser.add_argument("--max_new_tokens", type=int, default=192, help="Max new tokens for caption continuation.")
    parser.add_argument(
        "--shard_dirs",
        action="store_true",
        help="Shard output dirs (recommended) to avoid 10k folders in one directory.",
    )

    return parser.parse_args()


def contains_non_english(text: str) -> bool:
    """Detect if generated text contains non-English (non-ASCII) characters."""
    return bool(re.search(r"[^\x00-\x7F]", text))


def load_attributes_and_clashes(file_prob_txt, attribute_clash_txt):
    def validate_clash_rules(attributes, clash_rules):
        for rule in clash_rules:
            fixed_file = rule["fixed_file"]
            secondary_file = rule["secondary_file"]
            fixed_attributes = rule["fixed_attributes"]
            secondary_attributes = rule["secondary_attributes"]

            if fixed_file not in attributes:
                raise ValueError(f"Fixed file '{fixed_file}' does not exist in attributes.")
            if secondary_file not in attributes:
                raise ValueError(f"Secondary file '{secondary_file}' does not exist in attributes.")

            for attr in fixed_attributes:
                if attr not in attributes[fixed_file]["attribute_probabilities"]:
                    raise ValueError(f"Attribute '{attr}' in fixed file '{fixed_file}' does not exist.")

            for attr in secondary_attributes:
                if attr not in attributes[secondary_file]["attribute_probabilities"]:
                    raise ValueError(f"Attribute '{attr}' in secondary file '{secondary_file}' does not exist.")

    with open(file_prob_txt, "r") as f:
        file_probabilities = json.load(f)

    with open(attribute_clash_txt, "r") as f:
        clash_rules = json.load(f)

    attributes = {}
    base_dir = os.path.dirname(file_prob_txt)

    for filename, prob in file_probabilities.items():
        if prob > 0:
            filepath = os.path.join(base_dir, filename)
            with open(filepath, "r") as f:
                lines = f.readlines()
            attribute_probabilities = {}
            for line in lines:
                key, value = line.strip().split(",")
                attribute_probabilities[key] = float(value)
            attributes[filename] = {"prob_of_this_file": prob, "attribute_probabilities": attribute_probabilities}

    validate_clash_rules(attributes, clash_rules)
    return attributes, clash_rules


def select_attributes(attributes, clash_rules):
    """
    Randomly select one attribute from each file (based on probabilities),
    then resolve clashes using rules.
    """
    def normalize_probabilities(probabilities):
        total = sum(probabilities.values())
        return {k: v / total for k, v in probabilities.items() if v > 0}

    def select_current_attributes(attribute_probs):
        return random.choices(list(attribute_probs.keys()), weights=list(attribute_probs.values()), k=1)[0]

    def apply_clash_rules_bulk(selected_attributes, clash_rules, attributes_local):
        for rule in clash_rules:
            fixed_file = rule["fixed_file"]
            fixed_attributes = rule["fixed_attributes"]
            secondary_file = rule["secondary_file"]
            secondary_attributes = rule["secondary_attributes"]

            if fixed_file in selected_attributes and secondary_file in selected_attributes:
                fixed_selected = selected_attributes[fixed_file]
                if fixed_selected in fixed_attributes:
                    secondary_selected = selected_attributes[secondary_file]
                    if secondary_selected in secondary_attributes:
                        attribute_probs = attributes_local[secondary_file]["attribute_probabilities"]

                        for attr in secondary_attributes:
                            attribute_probs.pop(attr, None)

                        if not attribute_probs:
                            selected_attributes.pop(secondary_file, None)
                            attributes_local.pop(secondary_file, None)
                        else:
                            attribute_probs = normalize_probabilities(attribute_probs)
                            attributes_local[secondary_file]["attribute_probabilities"] = attribute_probs
                            selected_attributes[secondary_file] = select_current_attributes(attribute_probs)

    attributes_local = copy.deepcopy(attributes)

    filtered_attributes = {
        file: data
        for file, data in attributes_local.items()
        if random.uniform(0, 1) <= data["prob_of_this_file"]
    }

    selected_attributes = {
        file: select_current_attributes(data["attribute_probabilities"])
        for file, data in filtered_attributes.items()
    }

    while True:
        prev = selected_attributes.copy()
        apply_clash_rules_bulk(selected_attributes, clash_rules, attributes_local)
        if selected_attributes == prev:
            break

    return selected_attributes


# ---------------------------------------------------------------------------
# Head pose definitions  (strict 3-pose set)
# ---------------------------------------------------------------------------
# Keeping only hard, unambiguous angles so the diffusion model has no room
# to fall back to a near-frontal default.
#
#   front-facing   – 0° yaw, camera dead-on
#   profile left   – 90° yaw, true left side-on (only right ear/cheek visible)
#   profile right  – 90° yaw, true right side-on (only left ear/cheek visible)
#
# Equal weights → each pose appears ~33 % of the time.
# ---------------------------------------------------------------------------
HEAD_POSES = {
    "front-facing":   0.34,
    "profile left":   0.33,
    "profile right":  0.33,
}

# Flux.1 Dev optimised descriptions.
# Key principles for Flux:
#   1. Lead with the pose keyword immediately -- Flux weighs early tokens heavily.
#   2. Use dense natural prose, not tag lists.
#   3. State BOTH what IS visible AND what is NOT visible -- negative visibility
#      cues are the strongest way to force a true profile in Flux.
#   4. Repeat the core orientation word (profile / front-facing) twice.
HEAD_POSE_DESCRIPTIONS = {
    "front": (
        "Front-facing portrait. "
        "The subject looks straight into the camera in a perfectly symmetrical front-facing pose. "
        "Both eyes are fully visible, both ears are equally visible on each side, "
        "the nose points directly toward the lens, and the entire mouth is centred in frame. "
        "There is no yaw or tilt rotation whatsoever."
    ),
    "left_60": (
        "Strong side-profile portrait, head turned sharply to the left. "
        "The face is angled approximately 70-75 degrees away from the camera toward the left. "
        "The left ear is fully visible and prominent. "
        "The left cheek, left eye, and left side of the jaw dominate the frame. "
        "The right ear is completely hidden behind the head. "
        "The right eye is barely visible or not visible at all — only a sliver of the right cheek can be seen. "
        "The nose protrudes clearly to the left of the face, pointing away from the lens. "
        "The chin and jawline are seen sharply from the side. "
        "This is an aggressive turn — much closer to a full side profile than a standard three-quarter view."
    ),
    "right_60": (
        "Strong side-profile portrait, head turned sharply to the right. "
        "The face is angled approximately 70-75 degrees away from the camera toward the right. "
        "The right ear is fully visible and prominent. "
        "The right cheek, right eye, and right side of the jaw dominate the frame. "
        "The left ear is completely hidden behind the head. "
        "The left eye is barely visible or not visible at all — only a sliver of the left cheek can be seen. "
        "The nose protrudes clearly to the right of the face, pointing away from the lens. "
        "The chin and jawline are seen sharply from the side. "
        "This is an aggressive turn — much closer to a full side profile than a standard three-quarter view."
    ),
    "back": (
        "Rear portrait, subject facing completely away from the camera. "
        "The back of the head is fully visible: the occipital region, both ears from behind, "
        "and the nape of the neck are clearly in frame. "
        "No facial features are visible whatsoever -- no eyes, no nose, no mouth, no forehead. "
        "The camera looks directly at the back of the skull. "
        "Hair texture, head shape, and ear shape from behind are the primary visible features."
    ),
}

# Short filesystem-safe tags used in output filenames
HEAD_POSE_TAGS = {
    "front":    "front",
    "left_60":  "left_60",
    "right_60": "right_60",
    "back":     "back",
}

# Ordered list — every identity gets exactly these 4 poses, in this sequence
# Poses and their sampling weights.
# front=50%, left_60=25%, right_60=25% — gives natural distribution
# with meaningful side-face coverage for face recognition training.
POSE_WEIGHTS = {
    "front":    0.50,
    "left_60":  0.25,
    "right_60": 0.25,
}

def sample_pose() -> str:
    """Randomly sample one pose according to POSE_WEIGHTS."""
    return random.choices(list(POSE_WEIGHTS.keys()), weights=list(POSE_WEIGHTS.values()), k=1)[0]

# Short keyword injected at the very start of each Flux caption
POSE_LEAD = {
    "front":    "Front-facing portrait.",
    "left_60":  "Strong side profile portrait, head turned sharply to the left.",
    "right_60": "Strong side profile portrait, head turned sharply to the right.",
    "back":     "Rear portrait, subject facing away from camera.",
}


def construct_prompt(selected_attributes, pose: str = None):
    """Build the structured input + caption start for LLM continuation.

    Args:
        selected_attributes: dict of sampled attribute key->value pairs.
        pose: optional pose key from POSE_WEIGHTS. If None, one is sampled
              randomly according to POSE_WEIGHTS.
    """
    p = inflect.engine()
    prompt_parts = []
    gender = None

    for key, value in selected_attributes.items():
        attribute_name = key.replace(".txt", "")
        singular_name = p.singular_noun(attribute_name) or attribute_name
        formatted_name = singular_name.replace("_", " ").capitalize()

        # Normalize clothing keys so LLM sees a consistent field name
        if "clothing" in formatted_name.lower():
            formatted_name = "Clothing"

        # Capture gender but don't include as a parameter line
        if formatted_name.lower() in ["gender", "genders", "sex", "sexes"]:
            gender = value.lower()
            continue

        prompt_parts.append(f"{formatted_name}: {value}")

    # Sample a pose if not explicitly provided
    if pose is None:
        pose = sample_pose()

    # Do NOT pass head pose to the LLM at all.
    # Pose direction is injected programmatically after generation so the LLM
    # cannot hallucinate the wrong direction (left vs right confusion).

    # Decide subject
    if gender in ["male", "man"]:
        subject = "man"
    elif gender in ["female", "woman"]:
        subject = "woman"
    elif gender in ["non-binary"]:
        subject = "non-binary person"
    else:
        subject = "man" if random.choice([True, False]) else "woman"

    # Caption start — trigger token first, then pose keyword for side poses
    if pose == "front":
        caption = f"<s7ll2f4h:0.5>, Photograph of a {subject}. The subject has a neutral face expression."
    else:
        caption = f"<s7ll2f4h:0.5>, {POSE_LEAD[pose]} Photograph of a {subject}. The subject has a neutral face expression."

    formatted_parameters = "\n".join(prompt_parts)
    full_prompt = f"{formatted_parameters}\n\nPrompt: {caption}"
    return full_prompt, caption, subject, pose


def load_existing_uuids(dataset_dir):
    if not os.path.exists(dataset_dir):
        return set()
    return set(f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f)))


def generate_unique_uuid(existing_uuids):
    while True:
        unique_id = uuid.uuid4().hex[:12]
        if unique_id not in existing_uuids:
            existing_uuids.add(unique_id)
            return unique_id


def make_output_paths(dataset_dir: str, unique_uuid: str, subject: str, pose: str, shard_dirs: bool):
    """
    Build the output subfolder and JSON filename.
    Filename format: {uuid}_{gender}_{pose_tag}.json
    e.g. abc123def456_f_3q_left.json
    """
    if shard_dirs:
        subfolder_path = os.path.join(dataset_dir, unique_uuid[:2], unique_uuid[2:4], unique_uuid)
    else:
        subfolder_path = os.path.join(dataset_dir, unique_uuid)

    os.makedirs(subfolder_path, exist_ok=True)

    gender_tag = "m" if subject == "man" else "f" if subject == "woman" else "nb"
    pose_tag = HEAD_POSE_TAGS[pose]
    info_file = f"{unique_uuid}_{gender_tag}_{pose_tag}.json"
    info_file_path = os.path.join(subfolder_path, info_file)
    return subfolder_path, info_file_path


def enforce_ethnicity(generated_text: str, selected_attributes: dict, subject: str) -> str:
    """Inject ethnicity/origin into the caption if the LLM omitted it."""
    origin_val = selected_attributes.get("origins.txt")
    if not origin_val:
        return generated_text
    # Check if any word from the origin value appears in the text
    origin_words = [w.lower() for w in origin_val.split() if len(w) > 3]
    if any(w in generated_text.lower() for w in origin_words):
        return generated_text
    # Inject before "Photo captures"
    pronoun = "She is" if subject == "woman" else "He is" if subject == "man" else "They are"
    insert = f"{pronoun} of {origin_val} origin."
    idx = generated_text.find("Photo captures")
    if idx != -1:
        generated_text = generated_text[:idx].rstrip() + " " + insert + " " + generated_text[idx:].lstrip()
    else:
        generated_text = generated_text.rstrip() + " " + insert
    return generated_text


def enforce_single_clothing(generated_text: str, selected_attributes: dict, subject: str) -> str:
    """
    Prevent the LLM from inventing/stacking multiple outfits.
    Keep only the clothing that matches the sampled Clothing value.
    """
    clothing_val = selected_attributes.get("clothing_female.txt") or selected_attributes.get("clothing_male.txt")
    if not clothing_val:
        return generated_text

    # Separate the trigger/caption prefix from the rest so sentence splitting
    # never accidentally drops or mangles the <s7ll2f4h:0.5> opening.
    prefix = ""
    body = generated_text
    if generated_text.startswith("<"):
        # Find end of the second sentence (after "neutral face expression.")
        marker = "The subject has a neutral face expression."
        idx = generated_text.find(marker)
        if idx != -1:
            split_at = idx + len(marker)
            prefix = generated_text[:split_at].rstrip()
            body = generated_text[split_at:].lstrip()

    sentences = re.split(r'(?<=[.!?])\s+', body.strip())
    cleaned = []
    for s in sentences:
        sl = s.lower()
        if "wearing" in sl and clothing_val.lower() not in sl:
            continue
        cleaned.append(s)
    body = " ".join(cleaned)

    result = (prefix + " " + body).strip() if prefix else body

    if clothing_val.lower() not in result.lower():
        pronoun = "She" if subject == "woman" else "He" if subject == "man" else "They"
        insert = f"{pronoun} is wearing {clothing_val}."
        idx = result.find("Photo captures")
        if idx != -1:
            result = result[:idx].rstrip() + " " + insert + " " + result[idx:].lstrip()
        else:
            result = result.rstrip() + " " + insert

    return result


@inference_mode()
def generate_prompts(dataset_dir, number_of_prompts, file_prob_txt, attribute_clash_txt, batch_size=64, max_new_tokens=192, shard_dirs=True):
    os.makedirs(dataset_dir, exist_ok=True)

    existing_uuids = set() if shard_dirs else load_existing_uuids(dataset_dir)

    model_path = "./models/huggingface/Qwen2.5"

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()

    attributes, clash_rules = load_attributes_and_clashes(file_prob_txt, attribute_clash_txt)

    # Base negative prompt applied to every image
    negative_prompt_base = (
        "gray t-shirt, grey tshirt, prison uniform, inmate clothing, plain tee, undershirt, tank top, "
        "harsh flash, studio flash glare, ring light reflection, specular highlights, lens flare, light glare, "
        "shiny skin, oily skin, sweaty skin, overexposed face, blown highlights, hotspot lighting, glossy forehead, "
        "reflective glasses, mirror reflection, harsh shadows, low quality, blurry"
    )
    # Extra terms added only for side-pose images to push Flux away from its frontal default
    negative_prompt_side = (
        ", frontal face, facing camera, looking at viewer, symmetrical face, both eyes visible"
    )

    base_prompt = """
You will be provided with a structured list of input parameters describing details for a photo.

Rules for Output:
1) Combine all provided parameters into one descriptive prompt. Write dense natural prose (as Flux.1 Dev works best with prose, not tag lists). Explicitly include every single input parameter.
2) Make sure to use EVERY SINGLE parameter! DO NOT ignore anything.
3) DO NOT include unmentioned subjective parameters such as beauty, neatness, tidiness, etc.
4) The caption already starts with a pose keyword and subject line. Your job is to continue it by adding the remaining descriptions.
5) NEVER repeat the opening line. Start your continuation directly from where the prompt leaves off.
6) If any clothing parameter is present (e.g., "Clothing" or anything containing the word "clothing"), you MUST include it in the caption using the exact phrase: "wearing {clothing}".
7) Do NOT invent extra outfits. Mention EXACTLY ONE outfit using ONLY the provided Clothing value.
8) Do NOT mention head pose or head direction in your output at all. The pose will be handled separately.
9) The Origin/Ethnicity parameter MUST always be included. It is mandatory. Never skip it.
10) Output ONLY plain text. Do NOT wrap your output in quotes, XML tags, HTML attributes, markdown, or any formatting like content="...". Just output the raw continuation text directly.

Task: Using the given input parameters, complete (start directly where I left off) the given prompt for the diffusion model.
Do not include explanations, formatting templates, or additional examples — output only the resulting plain text prompt based on the provided data.

Input Parameters:
""".strip() + "\n"

    total = number_of_prompts
    pbar = tqdm(total=total, desc="Generating prompts")

    remaining = total
    while remaining > 0:
        cur_bs = min(batch_size, remaining)

        texts = []
        # tuples: (selected_attributes, caption_start, subject, pose, uuid, json_path)
        batch_meta = []

        # Build batch — one prompt per identity, pose sampled randomly
        for _ in range(cur_bs):
            selected_attributes = select_attributes(attributes, clash_rules)

            # Keep only one clothing file per identity
            g = selected_attributes.get("genders.txt", "").lower()
            if g in ["female", "woman"]:
                selected_attributes.pop("clothing_male.txt", None)
            elif g in ["male", "man"]:
                selected_attributes.pop("clothing_female.txt", None)
            else:
                # Non-binary: randomly keep one clothing file, drop the other
                if random.choice([True, False]):
                    selected_attributes.pop("clothing_male.txt", None)
                else:
                    selected_attributes.pop("clothing_female.txt", None)

            # pose is sampled inside construct_prompt when not specified
            dynamic_prompt, caption_start, subject, pose = construct_prompt(selected_attributes)
            final_prompt = base_prompt + dynamic_prompt

            messages = [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": final_prompt},
            ]
            chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            texts.append(chat_text)

            uid = generate_unique_uuid(existing_uuids)
            _, json_path = make_output_paths(dataset_dir, uid, subject, pose, shard_dirs)
            batch_meta.append((selected_attributes, caption_start, subject, pose, uid, json_path))

        # Tokenize batch once.
        # IMPORTANT: truncate from the LEFT so the "Prompt: <caption_start>" at the
        # END of each input is always preserved — this is what the LLM must see to
        # continue correctly. Losing the end causes the LLM to generate its own opening.
        tokenizer.truncation_side = "left"
        model_inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,  # Qwen2.5 supports 32k; 4096 is safe for our inputs
        ).to(model.device)

        # Generate batch once
        out_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)

        # Slice off the prompt portion for each item
        gen_only = []
        for inp_ids, out in zip(model_inputs.input_ids, out_ids):
            gen_only.append(out[len(inp_ids):])

        decoded = tokenizer.batch_decode(gen_only, skip_special_tokens=True)

        # Save each sample
        for (selected_attributes, caption_start, subject, pose, uid, json_path), gen in zip(batch_meta, decoded):
            generated_text = gen.strip()

            # ------------------------------------------------------------------
            # Strip chat-template artifacts that occasionally leak into the
            # decoded output, e.g. "assistant\n", "Assistant:", "<|im_start|>",
            # "<|im_end|>", or any leading role tokens left by the tokenizer.
            # Also strip content="..." wrapping that Qwen sometimes adds.
            # ------------------------------------------------------------------
            generated_text = re.sub(
                r"^(assistant\s*\n+|Assistant\s*:\s*|user\s*\n+|<\|im_start\|>\s*(assistant|user)\s*\n*|<\|im_end\|>\s*)+",
                "",
                generated_text,
                flags=re.IGNORECASE,
            ).strip()

            # Strip content="..." or content='...' wrapper that Qwen occasionally
            # outputs when it confuses the task with an HTML/XML attribute.
            if generated_text.startswith('content="') or generated_text.startswith("content='"):
                q = generated_text[8]
                end = generated_text.rfind(q)
                if end > 8:
                    generated_text = generated_text[9:end].strip()


            # Always prepend caption_start cleanly.
            # Strip ALL occurrences the LLM may have included to avoid duplication.
            while caption_start in generated_text:
                idx = generated_text.rfind(caption_start)
                generated_text = generated_text[idx + len(caption_start):].lstrip()
            generated_text = caption_start + " " + generated_text.lstrip()

            # For side poses: append the full pose description AFTER the face
            # feature text so Flux gets the strong angular signal without the
            # LLM spending tokens on it instead of face attributes.
            if pose != "front" and HEAD_POSE_DESCRIPTIONS[pose] not in generated_text:
                # Insert before "Photo captures" if present, else append
                idx = generated_text.find("Photo captures")
                pose_sentence = HEAD_POSE_DESCRIPTIONS[pose]
                if idx != -1:
                    generated_text = (
                        generated_text[:idx].rstrip()
                        + " " + pose_sentence
                        + " " + generated_text[idx:].lstrip()
                    )
                else:
                    generated_text = generated_text.rstrip() + " " + pose_sentence

            # Ensure framing/background suffix
            if "Photo captures" not in generated_text:
                generated_text += " Photo captures the subject from the chest up."
            if "White background" not in generated_text:
                generated_text += " White background."

            # Enforce clothing consistency
            generated_text = enforce_single_clothing(generated_text, selected_attributes, subject)
            # Enforce ethnicity is present
            generated_text = enforce_ethnicity(generated_text, selected_attributes, subject)

            # If non-English appears, do a single cheap retry
            if contains_non_english(generated_text):
                dynamic_prompt2, caption_start2, subject2, pose2 = construct_prompt(selected_attributes)
                final_prompt2 = base_prompt + dynamic_prompt2
                retry_chat = tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                        {"role": "user", "content": final_prompt2},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                retry_inputs = tokenizer([retry_chat], return_tensors="pt").to(model.device)
                retry_out = model.generate(**retry_inputs, max_new_tokens=max_new_tokens)
                retry_gen = retry_out[0][len(retry_inputs.input_ids[0]):]
                retry_text = tokenizer.decode(retry_gen, skip_special_tokens=True).strip()
                retry_text = re.sub(
                    r"^(assistant\s*\n+|Assistant\s*:\s*|<\|im_start\|>\s*assistant\s*\n*|<\|im_end\|>\s*)+",
                    "",
                    retry_text,
                    flags=re.IGNORECASE,
                ).strip()
                if retry_text.startswith('content="') or retry_text.startswith("content='"):
                    q = retry_text[8]
                    end = retry_text.rfind(q)
                    if end > 8:
                        retry_text = retry_text[9:end].strip()
                    retry_text = caption_start2 + (" " if retry_text and not retry_text.startswith(" ") else "") + retry_text
                if "Photo captures" not in retry_text:
                    retry_text += " Photo captures the subject from the chest up."
                if "White background" not in retry_text:
                    retry_text += " White background."
                retry_text = enforce_single_clothing(retry_text, selected_attributes, subject2)
                retry_text = enforce_ethnicity(retry_text, selected_attributes, subject2)

                if not contains_non_english(retry_text):
                    generated_text = retry_text
                    # Keep original pose — don't resample, pose is set at batch build time

            # Rebuild caption_start from pose to guarantee left/right is correct
            # (retry or other logic may have drifted caption_start out of sync)
            if pose == "front":
                caption_start = f"<s7ll2f4h:0.5>, Photograph of a {subject}. The subject has a neutral face expression."
            else:
                caption_start = f"<s7ll2f4h:0.5>, {POSE_LEAD[pose]} Photograph of a {subject}. The subject has a neutral face expression."

            # Final safety net — trigger token must always be present
            trigger = "<s7ll2f4h:0.5>"
            if trigger not in generated_text:
                generated_text = caption_start + " " + generated_text.lstrip()
            # Also fix wrong direction if present — strip and re-prepend
            elif not generated_text.startswith(caption_start):
                # Wrong caption_start was prepended — remove it and use correct one
                for wrong_lead in POSE_LEAD.values():
                    wrong_start = f"<s7ll2f4h:0.5>, {wrong_lead}"
                    if generated_text.startswith(wrong_start) and wrong_start != f"<s7ll2f4h:0.5>, {POSE_LEAD[pose]}":
                        generated_text = generated_text[len(wrong_start):].lstrip()
                        generated_text = caption_start + " " + generated_text.lstrip()
                        break

            # Write JSON — now includes head_pose fields
            # Front pose: use base negative only. Side poses: add anti-frontal terms.
            neg = negative_prompt_base + (negative_prompt_side if pose != "front" else "")

            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump(
                    {
                        "attributes": selected_attributes,
                        "head_pose": pose,
                        "head_pose_description": HEAD_POSE_DESCRIPTIONS[pose],
                        "prompt": generated_text,
                        "negative_prompt": neg,
                    },
                    jf,
                    indent=4,
                    ensure_ascii=False,
                )

            pbar.update(1)

        remaining -= cur_bs

    pbar.close()


if __name__ == "__main__":
    args = parse_args()
    generate_prompts(
        dataset_dir=args.dataset_dir,
        number_of_prompts=args.number_of_prompts,
        file_prob_txt=args.file_prob_txt,
        attribute_clash_txt=args.attribute_clash_txt,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        shard_dirs=args.shard_dirs,
    )