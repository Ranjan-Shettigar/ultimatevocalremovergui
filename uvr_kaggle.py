import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

MODEL_DIR = Path("/kaggle/working/uvr_models")
OUTPUT_DIR = Path("/kaggle/working/uvr_outputs")

# UVR/audio-separator model aliases.
# The resolver checks the live model catalogue instead of assuming every
# model has exactly the same filename.
MODEL_ALIASES = {
    "bs_roformer": [
        "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        "model_bs_roformer_ep_368_sdr_12.9628.ckpt",
        "bs_roformer",
    ],
    "kim_vocal_2": [
        "kim_vocal_2",
        "kim vocal 2",
        "kim_vocal",
    ],
    "inst_hq_3": [
        "uvr-mdx-net-inst_hq_3",
        "uvr_mdxnet_inst_hq_3",
        "inst_hq_3",
    ],
    "mdx23c_instvoc": [
        "mdx23c",
        "instvoc",
        "mdx23c-instvoc",
    ],
    "htdemucs": [
        "htdemucs.yaml",
        "htdemucs",
    ],
    "demucs_6stem": [
        "htdemucs_6s.yaml",
        "htdemucs_6s",
    ],
    "bip_reverb": [
        "uvr-bip-reverb-hq",
        "uvr_bip_reverb_hq",
        "bip_reverb",
    ],
    "deecho": [
        "uvr-deecho-dereverb.pth",
        "uvr_deecho_dereverb",
        "deecho-dereverb",
    ],
    "kara_2": [
        "uvr_mdxnet_kara_2.onnx",
        "uvr-mdxnet-kara-2",
        "kara_2",
    ],
}


# Tasks:
#   vocal       = BS-RoFormer vocal/instrumental separation
#   instrumental= instrumental-focused model
#   stems4      = HTDemucs vocals/drums/bass/other
#   stems6      = Demucs six-stem separation
#   cleanup     = de-reverb / echo cleanup on the generated vocal stem
#   karaoke     = KARA_2 on the generated instrumental stem
#   all         = run all major separation tasks
TASKS = {
    "vocal",
    "instrumental",
    "stems4",
    "stems6",
    "cleanup",
    "karaoke",
    "all",
}


def run_command(command, capture_output=False):
    print("\n$", " ".join(map(str, command)))

    result = subprocess.run(
        [str(x) for x in command],
        text=True,
        capture_output=capture_output,
    )

    if result.returncode != 0:
        if capture_output:
            print(result.stdout)
            print(result.stderr, file=sys.stderr)
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: "
            f"{' '.join(map(str, command))}"
        )

    return result.stdout if capture_output else None


def get_model_catalogue():
    """
    Query audio-separator's current supported-model catalogue.
    The catalogue format can change, so this extracts filenames recursively.
    """
    command = [
        "audio-separator",
        "--list_models",
        "--list_format",
        "json",
    ]

    try:
        raw = run_command(command, capture_output=True)
        data = json.loads(raw)
    except Exception as exc:
        print(f"Could not read live model catalogue: {exc}")
        return []

    found = []

    def walk(value):
        if isinstance(value, str):
            text = value.strip()
            if (
                text.lower().endswith(
                    (".ckpt", ".onnx", ".pth", ".yaml", ".th", ".pt")
                )
                or "roformer" in text.lower()
                or "demucs" in text.lower()
                or "vocal" in text.lower()
                or "kara" in text.lower()
            ):
                found.append(text)

        elif isinstance(value, dict):
            for item in value.values():
                walk(item)

        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(data)

    # Preserve order while removing duplicates.
    return list(dict.fromkeys(found))


def normalize_name(value):
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def resolve_model(alias):
    """
    Resolve a friendly alias to an actual model filename.
    """
    catalogue = get_model_catalogue()
    candidates = MODEL_ALIASES[alias]

    # First try exact filename/name matches.
    for candidate in candidates:
        candidate_norm = normalize_name(candidate)

        for available in catalogue:
            available_norm = normalize_name(available)

            if available_norm == candidate_norm:
                return available

    # Then try partial/fuzzy matches.
    for candidate in candidates:
        candidate_norm = normalize_name(candidate)

        if len(candidate_norm) < 5:
            continue

        matches = [
            available
            for available in catalogue
            if candidate_norm in normalize_name(available)
        ]

        if matches:
            # Prefer a file with a recognized model extension.
            matches.sort(
                key=lambda x: (
                    not x.lower().endswith(
                        (".ckpt", ".onnx", ".pth", ".yaml", ".th", ".pt")
                    ),
                    len(x),
                )
            )
            return matches[0]

    print("\nAvailable model names containing useful keywords:")
    for model in catalogue:
        low = model.lower()
        if any(
            key in low
            for key in [
                "roformer",
                "vocal",
                "inst",
                "demucs",
                "kara",
                "reverb",
                "echo",
                "hq",
            ]
        ):
            print("  ", model)

    raise RuntimeError(
        f"Could not resolve model alias '{alias}'. "
        "Run: audio-separator --list_models"
    )


def separate(
    input_file,
    model_alias,
    output_dir,
    single_stem=None,
    extra_args=None,
):
    """
    Run one model through audio-separator.
    """
    model_filename = resolve_model(model_alias)

    job_dir = output_dir / model_alias
    job_dir.mkdir(parents=True, exist_ok=True)

    command = [
        "audio-separator",
        str(input_file),
        "--model_filename",
        model_filename,
        "--model_file_dir",
        str(MODEL_DIR),
        "--output_dir",
        str(job_dir),
        "--output_format",
        "WAV",
        "--sample_rate",
        "44100",
        "--normalization",
        "0.9",
        "--use_soundfile",
    ]

    if single_stem:
        command.extend(["--single_stem", single_stem])

    if extra_args:
        command.extend(extra_args)

    run_command(command)

    print(f"\nCompleted: {model_alias}")
    print(f"Output directory: {job_dir}")

    return job_dir


def find_stem(directory, keywords):
    """
    Find a generated stem by filename keywords.
    """
    candidates = list(directory.rglob("*.wav"))

    for path in candidates:
        name = path.name.lower()
        if all(keyword.lower() in name for keyword in keywords):
            return path

    return None


def copy_if_found(source, destination):
    if source and source.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        print(f"Copied: {source} -> {destination}")
        return destination

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Headless UVR model runner for Kaggle."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input audio file, for example /kaggle/input/song/song.mp3",
    )

    parser.add_argument(
        "--task",
        default="vocal",
        choices=sorted(TASKS),
        help=(
            "vocal, instrumental, stems4, stems6, cleanup, karaoke, or all. "
            "Default: vocal"
        ),
    )

    parser.add_argument(
        "--output",
        default=str(OUTPUT_DIR),
        help="Output directory.",
    )

    parser.add_argument(
        "--use_fp16",
        action="store_true",
        help="Enable autocast to reduce GPU memory usage.",
    )

    args = parser.parse_args()

    input_file = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Input:", input_file)
    print("Task:", args.task)
    print("Models:", MODEL_DIR)
    print("Outputs:", output_dir)

    autocast_args = ["--use_autocast"] if args.use_fp16 else []

    # ---------------------------------------------------------------
    # 1. Best overall vocal and instrumental separation
    # ---------------------------------------------------------------
    if args.task in {"vocal", "all"}:
        separate(
            input_file=input_file,
            model_alias="bs_roformer",
            output_dir=output_dir,
            extra_args=autocast_args,
        )

    # ---------------------------------------------------------------
    # 2. Kim Vocal 2
    # ---------------------------------------------------------------
    if args.task == "all":
        separate(
            input_file=input_file,
            model_alias="kim_vocal_2",
            output_dir=output_dir,
            single_stem="Vocals",
            extra_args=autocast_args,
        )

    # ---------------------------------------------------------------
    # 3. Clean instrumental
    # ---------------------------------------------------------------
    if args.task in {"instrumental", "all"}:
        separate(
            input_file=input_file,
            model_alias="inst_hq_3",
            output_dir=output_dir,
            single_stem="Instrumental",
            extra_args=autocast_args,
        )

    # ---------------------------------------------------------------
    # 4. MDX23C instrumental/vocal model
    # ---------------------------------------------------------------
    if args.task == "all":
        separate(
            input_file=input_file,
            model_alias="mdx23c_instvoc",
            output_dir=output_dir,
            extra_args=[
                "--mdxc_segment_size",
                "256",
                "--mdxc_overlap",
                "8",
                *autocast_args,
            ],
        )

    # ---------------------------------------------------------------
    # 5. Four stems: vocals, drums, bass, other
    # ---------------------------------------------------------------
    if args.task in {"stems4", "all"}:
        separate(
            input_file=input_file,
            model_alias="htdemucs",
            output_dir=output_dir,
            extra_args=[
                "--demucs_shifts",
                "1",
                "--demucs_overlap",
                "0.25",
                "--demucs_segments_enabled",
                *autocast_args,
            ],
        )

    # ---------------------------------------------------------------
    # 6. Six stems: vocals, drums, bass, other, guitar, piano
    # ---------------------------------------------------------------
    if args.task in {"stems6", "all"}:
        separate(
            input_file=input_file,
            model_alias="demucs_6stem",
            output_dir=output_dir,
            extra_args=[
                "--demucs_shifts",
                "1",
                "--demucs_overlap",
                "0.25",
                "--demucs_segments_enabled",
                *autocast_args,
            ],
        )

    # ---------------------------------------------------------------
    # 7. Cleanup generated BS-RoFormer vocals
    # ---------------------------------------------------------------
    if args.task in {"cleanup", "all"}:
        bs_dir = output_dir / "bs_roformer"
        vocal_file = find_stem(bs_dir, ["vocal"])

        if vocal_file is None:
            print(
                "No BS-RoFormer vocal file found. "
                "Run --task vocal first or use --task all."
            )
        else:
            separate(
                input_file=vocal_file,
                model_alias="bip_reverb",
                output_dir=output_dir,
                single_stem="Vocals",
                extra_args=autocast_args,
            )

            separate(
                input_file=vocal_file,
                model_alias="deecho",
                output_dir=output_dir,
                single_stem="Vocals",
                extra_args=autocast_args,
            )

    # ---------------------------------------------------------------
    # 8. Remove backing vocals/harmonies from generated instrumental
    # ---------------------------------------------------------------
    if args.task in {"karaoke", "all"}:
        bs_dir = output_dir / "bs_roformer"
        instrumental_file = find_stem(bs_dir, ["instrumental"])

        if instrumental_file is None:
            print(
                "No BS-RoFormer instrumental file found. "
                "Run --task vocal first or use --task all."
            )
        else:
            separate(
                input_file=instrumental_file,
                model_alias="kara_2",
                output_dir=output_dir,
                single_stem="Instrumental",
                extra_args=autocast_args,
            )

    print("\nFinished.")
    print("All results are in:", output_dir)


if __name__ == "__main__":
    main()