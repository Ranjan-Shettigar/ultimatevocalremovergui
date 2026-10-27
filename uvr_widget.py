"""
Interactive ipywidgets UI for running UVR5 in Kaggle / Jupyter Notebooks.
"""

import os
import shutil
from pathlib import Path
import ipywidgets as widgets
from IPython.display import display, Audio, HTML, clear_output

from uvr_kaggle import (
    separate,
    find_stem,
    MODEL_DIR,
    OUTPUT_DIR,
    TASKS,
)


AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac", ".wma"}


def scan_audio_files(search_paths=None):
    """Scan common directories for audio files."""
    if search_paths is None:
        search_paths = [
            Path("/kaggle/input"),
            Path("/kaggle/working"),
            Path("."),
        ]

    found = []
    for base in search_paths:
        if base.exists():
            for ext in AUDIO_EXTENSIONS:
                for file_path in base.rglob(f"*{ext}"):
                    # Skip temporary files or output files
                    if ".ipynb_checkpoints" in str(file_path):
                        continue
                    found.append(str(file_path.resolve()))

    return sorted(list(dict.fromkeys(found)))


def create_uvr_ui():
    """Build and display the ipywidgets UVR interface."""

    # 1. Header
    header = widgets.HTML(
        """
        <div style="background: linear-gradient(135deg, #1e293b, #0f172a); padding: 16px 20px; border-radius: 10px; margin-bottom: 12px; color: #f8fafc;">
            <h2 style="margin: 0 0 6px 0; color: #38bdf8;">🎵 UVR5 Headless Separator</h2>
            <p style="margin: 0; font-size: 13px; color: #94a3b8;">
                High-quality vocal, instrumental, and multi-stem audio separation powered by BS-RoFormer & Demucs.
            </p>
        </div>
        """
    )

    # 2. File Input Mode
    scanned_files = scan_audio_files()
    def get_file_options():
        files = scan_audio_files()
        return [("Select a discovered file...", "")] + [
            (f"{Path(f).name} ({Path(f).parent.name})", f) for f in files
        ] + [("Enter custom path below...", "__CUSTOM__")]

    dropdown_file = widgets.Dropdown(
        options=get_file_options(),
        value="" if not scanned_files else get_file_options()[1][1],
        description="Found Files:",
        style={"description_width": "110px"},
        layout=widgets.Layout(width="75%"),
    )

    refresh_btn = widgets.Button(
        description="Scan",
        tooltip="Rescan for audio files in /kaggle/input and working directory",
        icon="refresh",
        layout=widgets.Layout(width="18%"),
    )

    file_picker_row = widgets.HBox([dropdown_file, refresh_btn])

    custom_input_text = widgets.Text(
        value="/kaggle/input/datasets/lokotwist/singari/singari.mp3",
        placeholder="/path/to/song.mp3",
        description="Audio Path:",
        style={"description_width": "110px"},
        layout=widgets.Layout(width="95%"),
    )

    def on_refresh_clicked(_):
        dropdown_file.options = get_file_options()
        if len(dropdown_file.options) > 1 and dropdown_file.options[1][1] != "__CUSTOM__":
            dropdown_file.value = dropdown_file.options[1][1]
            custom_input_text.value = dropdown_file.value

    refresh_btn.on_click(on_refresh_clicked)

    def on_dropdown_change(change):
        if change["new"] and change["new"] != "__CUSTOM__":
            custom_input_text.value = change["new"]

    dropdown_file.observe(on_dropdown_change, names="value")

    # 3. Task Selection
    task_options = [
        ("🎤 Vocals & Instrumental (BS-RoFormer - Best SDR)", "vocal"),
        ("🎹 Clean Instrumental (UVR-MDX-Net Inst HQ 3)", "instrumental"),
        ("🥁 4 Stems: Vocals, Drums, Bass, Other (HTDemucs)", "stems4"),
        ("🎸 6 Stems: Vocals, Drums, Bass, Other, Guitar, Piano (Demucs 6s)", "stems6"),
        ("🧹 Vocal Cleanup (De-Echo & Dereverb)", "cleanup"),
        ("🎤 Backing Vocals / Harmonies Filter (Kara 2)", "karaoke"),
        ("🌟 Full Pipeline / All Models", "all"),
    ]

    task_dropdown = widgets.Dropdown(
        options=task_options,
        value="vocal",
        description="Task / Model:",
        style={"description_width": "110px"},
        layout=widgets.Layout(width="95%"),
    )

    # 4. Settings
    format_dropdown = widgets.Dropdown(
        options=["WAV", "FLAC", "MP3"],
        value="WAV",
        description="Format:",
        style={"description_width": "80px"},
        layout=widgets.Layout(width="180px"),
    )

    fp16_checkbox = widgets.Checkbox(
        value=True,
        description="FP16 Autocast (Faster GPU inference & lower VRAM)",
        style={"description_width": "initial"},
    )

    output_dir_text = widgets.Text(
        value=str(OUTPUT_DIR),
        description="Output Dir:",
        style={"description_width": "110px"},
        layout=widgets.Layout(width="95%"),
    )

    # 5. Separation Button
    separate_btn = widgets.Button(
        description=" Start Separation",
        button_style="primary",
        icon="play",
        layout=widgets.Layout(width="220px", height="42px", margin="12px 0"),
    )

    # 6. Output log area & audio players
    log_output = widgets.Output(
        layout=widgets.Layout(
            border="1px solid #334155",
            padding="10px",
            max_height="320px",
            overflow="auto",
            margin="10px 0",
        )
    )

    players_output = widgets.Output()

    def run_separation_ui(_):
        input_path = Path(custom_input_text.value.strip())
        out_base_dir = Path(output_dir_text.value.strip())
        task = task_dropdown.value
        fmt = format_dropdown.value
        use_fp16 = fp16_checkbox.value

        players_output.clear_output()
        log_output.clear_output()

        if not input_path.exists():
            with log_output:
                print(f"❌ Error: Input file does not exist: {input_path}")
            return

        separate_btn.disabled = True
        separate_btn.description = " Separating..."
        separate_btn.button_style = "warning"
        separate_btn.icon = "hourglass-half"

        out_base_dir.mkdir(parents=True, exist_ok=True)
        autocast_args = ["--use_autocast"] if use_fp16 else []

        with log_output:
            print(f"▶ Input Audio: {input_path}")
            print(f"▶ Task: {task}")
            print(f"▶ Output Format: {fmt}")
            print(f"▶ Autocast (FP16): {use_fp16}")
            print(f"▶ Output Dir: {out_base_dir}")
            print("-" * 50)

        def do_sep(inp, alias, stem=None, extra=None):
            return separate(
                input_file=inp,
                model_alias=alias,
                output_dir=out_base_dir,
                single_stem=stem,
                extra_args=extra,
                output_format=fmt,
                use_soundfile=False,
            )

        try:
            with log_output:
                if task in {"vocal", "all"}:
                    do_sep(input_path, "bs_roformer", extra=autocast_args)

                if task == "all":
                    do_sep(input_path, "kim_vocal_2", stem="Vocals", extra=autocast_args)

                if task in {"instrumental", "all"}:
                    do_sep(input_path, "inst_hq_3", stem="Instrumental", extra=autocast_args)

                if task == "all":
                    do_sep(
                        input_path,
                        "mdx23c_instvoc",
                        extra=[
                            "--mdxc_segment_size",
                            "256",
                            "--mdxc_overlap",
                            "8",
                            *autocast_args,
                        ],
                    )

                if task in {"stems4", "all"}:
                    do_sep(
                        input_path,
                        "htdemucs",
                        extra=[
                            "--demucs_shifts",
                            "1",
                            "--demucs_overlap",
                            "0.25",
                            "--demucs_segments_enabled",
                            "True",
                            *autocast_args,
                        ],
                    )

                if task in {"stems6", "all"}:
                    do_sep(
                        input_path,
                        "demucs_6stem",
                        extra=[
                            "--demucs_shifts",
                            "1",
                            "--demucs_overlap",
                            "0.25",
                            "--demucs_segments_enabled",
                            "True",
                            *autocast_args,
                        ],
                    )

                if task in {"cleanup", "all"}:
                    bs_dir = out_base_dir / "bs_roformer"
                    vocal_file = find_stem(bs_dir, ["vocal"])
                    if vocal_file:
                        do_sep(vocal_file, "bip_reverb", stem="Vocals", extra=autocast_args)
                        do_sep(vocal_file, "deecho", stem="Vocals", extra=autocast_args)
                    else:
                        print("⚠ No BS-RoFormer vocal found to clean.")

                if task in {"karaoke", "all"}:
                    bs_dir = out_base_dir / "bs_roformer"
                    inst_file = find_stem(bs_dir, ["instrumental"])
                    if inst_file:
                        do_sep(inst_file, "kara_2", stem="Instrumental", extra=autocast_args)
                    else:
                        print("⚠ No BS-RoFormer instrumental found for karaoke processing.")

                print("\n✅ Separation successfully completed!")

            # 7. Render Audio Players for outputs
            with players_output:
                stems = [
                    p for p in out_base_dir.rglob("*")
                    if p.is_file() and p.suffix.lower() in {".wav", ".flac", ".mp3", ".ogg"}
                ]
                if stems:
                    display(
                        HTML(
                            """
                            <h3 style="color: #22c55e; margin-top: 15px;">🎧 Separated Stems:</h3>
                            """
                        )
                    )
                    for stem in sorted(stems):
                        size_mb = stem.stat().st_size / (1024 * 1024)
                        display(
                            HTML(
                                f"""
                                <div style="margin-top: 8px;">
                                    <b>{stem.name}</b> <span style="color:#64748b; font-size:12px;">({size_mb:.2f} MB)</span>
                                </div>
                                """
                            )
                        )
                        display(Audio(str(stem)))

        except Exception as exc:
            with log_output:
                print(f"\n❌ Error during separation: {exc}")
        finally:
            separate_btn.disabled = False
            separate_btn.description = " Start Separation"
            separate_btn.button_style = "primary"
            separate_btn.icon = "play"

    separate_btn.on_click(run_separation_ui)

    # Layout assembly
    options_box = widgets.HBox([format_dropdown, fp16_checkbox])

    main_box = widgets.VBox(
        [
            header,
            file_picker_row,
            custom_input_text,
            task_dropdown,
            options_box,
            output_dir_text,
            separate_btn,
            widgets.HTML("<b>Console Logs:</b>"),
            log_output,
            players_output,
        ],
        layout=widgets.Layout(
            max_width="780px",
            padding="12px",
            border="1px solid #cbd5e1",
            border_radius="10px",
        ),
    )

    display(main_box)


def render_ui():
    """Convenience alias."""
    create_uvr_ui()


if __name__ == "__main__":
    create_uvr_ui()
