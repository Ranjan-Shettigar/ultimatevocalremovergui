"""
Minimalistic ipywidgets UI for UVR5 audio separation in Kaggle / Jupyter.
"""

import subprocess
from pathlib import Path

import ipywidgets as widgets
from IPython.display import display, Audio, HTML, FileLink

from uvr_kaggle import (
    separate,
    find_stem,
    MODEL_DIR,
    OUTPUT_DIR,
    TASKS,
)

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac", ".wma"}
_STEM_EXTS = {".wav", ".flac", ".mp3", ".ogg"}


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _scan(paths=None):
    """Return sorted list of audio file paths found under *paths*."""
    paths = paths or [Path("/kaggle/input"), Path("/kaggle/working"), Path(".")]
    found = []
    for base in paths:
        if base.exists():
            for ext in AUDIO_EXTS:
                found.extend(
                    str(p.resolve())
                    for p in base.rglob(f"*{ext}")
                    if ".ipynb_checkpoints" not in str(p)
                )
    return sorted(dict.fromkeys(found))


def _preview(path_str):
    """Return a path suitable for in-notebook playback (≤10 MB)."""
    p = Path(path_str)
    if not p.exists():
        return None
    if p.stat().st_size <= 10 * 1024 * 1024 or p.suffix.lower() == ".mp3":
        return str(p)
    prev = Path(f"/tmp/{p.stem}_preview.mp3")
    if not prev.exists():
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(p), "-b:a", "192k", "-loglevel", "error", str(prev)],
                check=True,
            )
        except Exception:
            return str(p)
    return str(prev)


def _download_link(path_str):
    """Return a FileLink for Kaggle/Jupyter download, or a fallback copy instruction."""
    p = Path(path_str)
    if not p.exists():
        return None

    # FileLink works when the file is under the Jupyter server root.
    # On Kaggle that's /kaggle/working/. If the file is elsewhere, show the path.
    working = Path("/kaggle/working")
    try:
        rel = p.resolve().relative_to(working.resolve())
        return FileLink(str(working / rel), result_html_prefix="⬇ Download: ")
    except ValueError:
        pass

    # Fallback: show copy-paste path
    return HTML(
        f'<span style="font-size:13px;color:#555;">📁 File at: <code>{p}</code></span>'
    )


def _collect_stems(dirs, fallback_dir):
    """Gather output stem files from *dirs*, falling back to *fallback_dir*."""
    stems = []
    for d in dirs:
        if d.exists():
            stems.extend(
                p for p in d.rglob("*")
                if p.is_file() and p.suffix.lower() in _STEM_EXTS
                and "_preview" not in p.name
            )
    if not stems:
        stems = [
            p for p in fallback_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in _STEM_EXTS
            and "_preview" not in p.name
        ]
    return sorted(dict.fromkeys(stems))


# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------

def create_uvr_ui():
    """Build and display the widget UI."""

    # --- Header ---
    header = widgets.HTML(
        '<h3 style="margin:0 0 10px 0;">🎵 UVR5 Audio Separator</h3>'
    )

    # --- Input file ---
    scanned = _scan()

    def _opts():
        fs = _scan()
        return [("— select —", "")] + [
            (Path(f).name, f) for f in fs
        ] + [("custom path ↓", "__CUSTOM__")]

    file_dd = widgets.Dropdown(
        options=_opts(),
        value=_opts()[1][1] if scanned else "",
        description="Input:",
        layout=widgets.Layout(width="80%"),
    )
    scan_btn = widgets.Button(description="⟳", tooltip="Rescan", layout=widgets.Layout(width="15%"))
    input_text = widgets.Text(
        value=scanned[0] if scanned else "",
        placeholder="/path/to/audio.mp3",
        description="Path:",
        layout=widgets.Layout(width="96%"),
    )

    def _on_scan(_):
        file_dd.options = _opts()
        if len(file_dd.options) > 1 and file_dd.options[1][1] != "__CUSTOM__":
            file_dd.value = file_dd.options[1][1]
            input_text.value = file_dd.value
    scan_btn.on_click(_on_scan)

    def _on_dd(change):
        if change["new"] and change["new"] != "__CUSTOM__":
            input_text.value = change["new"]
    file_dd.observe(_on_dd, names="value")

    # --- Task ---
    task_dd = widgets.Dropdown(
        options=[
            ("Vocals + Instrumental  (BS-RoFormer)", "vocal"),
            ("Instrumental only  (MDX-Net HQ3)", "instrumental"),
            ("4 stems  (HTDemucs)", "stems4"),
            ("6 stems  (Demucs 6s)", "stems6"),
            ("Vocal cleanup  (De-echo/reverb)", "cleanup"),
            ("Karaoke  (Remove harmonies)", "karaoke"),
            ("Full pipeline  (all models)", "all"),
        ],
        value="vocal",
        description="Task:",
        layout=widgets.Layout(width="96%"),
    )

    # --- Options row ---
    fmt_dd = widgets.Dropdown(
        options=["WAV", "FLAC", "MP3"], value="WAV",
        description="Format:", layout=widgets.Layout(width="160px"),
    )
    fp16_cb = widgets.Checkbox(value=True, description="FP16", indent=False)
    out_text = widgets.Text(
        value=str(OUTPUT_DIR), description="Output:",
        layout=widgets.Layout(width="96%"),
    )

    # --- Run button ---
    run_btn = widgets.Button(
        description="Separate", button_style="primary", icon="play",
        layout=widgets.Layout(width="140px", height="36px"),
    )

    # --- Output areas ---
    progress = widgets.IntProgress(
        value=0, min=0, max=1,
        bar_style="info",
        layout=widgets.Layout(width="96%", visibility="hidden"),
    )
    progress_label = widgets.HTML(
        value="",
        layout=widgets.Layout(visibility="hidden"),
    )
    log_out = widgets.Output(layout=widgets.Layout(
        max_height="240px", overflow="auto",
        border="1px solid #ddd", padding="8px", margin="8px 0",
    ))
    result_out = widgets.Output()

    # --- Run logic ---
    def _build_steps(task, ac):
        """Return list of (label, callable) steps for the chosen task."""
        steps = []
        if task in {"vocal", "all"}:
            steps.append(("BS-RoFormer (vocals)", lambda inp, s=ac: _sep(inp, "bs_roformer", extra=s)))
        if task == "all":
            steps.append(("Kim Vocal 2", lambda inp, s=ac: _sep(inp, "kim_vocal_2", stem="Vocals", extra=s)))
        if task in {"instrumental", "all"}:
            steps.append(("MDX-Net Inst HQ3", lambda inp, s=ac: _sep(inp, "inst_hq_3", stem="Instrumental", extra=s)))
        if task == "all":
            steps.append(("MDX23C", lambda inp, s=ac: _sep(
                inp, "mdx23c_instvoc",
                extra=["--mdxc_segment_size", "256", "--mdxc_overlap", "8", *s])))
        if task in {"stems4", "all"}:
            steps.append(("HTDemucs 4-stem", lambda inp, s=ac: _sep(
                inp, "htdemucs",
                extra=["--demucs_shifts", "1", "--demucs_overlap", "0.25",
                       "--demucs_segments_enabled", "True", *s])))
        if task in {"stems6", "all"}:
            steps.append(("Demucs 6-stem", lambda inp, s=ac: _sep(
                inp, "demucs_6stem",
                extra=["--demucs_shifts", "1", "--demucs_overlap", "0.25",
                       "--demucs_segments_enabled", "True", *s])))
        if task in {"cleanup", "all"}:
            steps.append(("__cleanup__", None))
        if task in {"karaoke", "all"}:
            steps.append(("__karaoke__", None))
        return steps

    def _run(_):
        inp = Path(input_text.value.strip())
        out_dir = Path(out_text.value.strip())
        task = task_dd.value
        fmt = fmt_dd.value
        fp16 = fp16_cb.value

        log_out.clear_output()
        result_out.clear_output()

        if not inp.exists():
            with log_out:
                print(f"❌ File not found: {inp}")
            return

        run_btn.disabled = True
        run_btn.description = "Running…"
        run_btn.button_style = "warning"

        out_dir.mkdir(parents=True, exist_ok=True)
        ac = ["--use_autocast"] if fp16 else []
        created = []

        def _sep(f, alias, stem=None, extra=None):
            d = separate(
                input_file=f, model_alias=alias, output_dir=out_dir,
                single_stem=stem, extra_args=extra,
                output_format=fmt, use_soundfile=False,
            )
            if d:
                created.append(Path(d))
            return d

        steps = _build_steps(task, ac)
        total = len(steps)
        progress.max = total
        progress.value = 0
        progress.bar_style = "info"
        progress.layout.visibility = "visible"
        progress_label.layout.visibility = "visible"

        try:
            with log_out:
                print(f"Input: {inp.name}  |  Task: {task}  |  Format: {fmt}")
                print("-" * 40)

                for i, (label, fn) in enumerate(steps):
                    progress_label.value = (
                        f'<span style="font-size:13px;">'
                        f'Step {i+1}/{total}: <b>{label}</b></span>'
                    )

                    if label == "__cleanup__":
                        vf = find_stem(out_dir / "bs_roformer", ["vocal"])
                        if vf:
                            _sep(vf, "bip_reverb", stem="Vocals", extra=ac)
                            _sep(vf, "deecho", stem="Vocals", extra=ac)
                        else:
                            print("⚠ No vocal stem found for cleanup.")
                    elif label == "__karaoke__":
                        inf = find_stem(out_dir / "bs_roformer", ["instrumental"])
                        if inf:
                            _sep(inf, "kara_2", stem="Instrumental", extra=ac)
                        else:
                            print("⚠ No instrumental stem found for karaoke.")
                    else:
                        fn(inp)

                    progress.value = i + 1

                print("\n✅ Done.")

            progress.bar_style = "success"
            progress_label.value = (
                f'<span style="font-size:13px;color:green;">'
                f'✅ Completed {total} step(s)</span>'
            )

            # --- Results ---
            stems = _collect_stems(created, out_dir)
            if not stems:
                return

            with result_out:
                # File list
                rows = "".join(
                    f"<tr><td style='padding:4px 8px;'>{s.name}</td>"
                    f"<td style='padding:4px 8px;color:#666;'>{s.stat().st_size/(1024*1024):.1f} MB</td></tr>"
                    for s in stems
                )
                display(HTML(
                    f"<details open style='margin:8px 0;'>"
                    f"<summary style='cursor:pointer;font-weight:600;'>Output files ({len(stems)})</summary>"
                    f"<table style='font-size:13px;margin-top:4px;'>{rows}</table></details>"
                ))

                # Player + download
                opts = [(p.name, str(p)) for p in stems]
                stem_dd = widgets.Dropdown(
                    options=opts, value=opts[0][1],
                    description="Stem:", layout=widgets.Layout(width="70%"),
                )
                player = widgets.Output()
                dl_area = widgets.Output()

                def _update_player(path_str):
                    player.clear_output()
                    dl_area.clear_output()
                    if path_str:
                        pv = _preview(path_str)
                        if pv:
                            with player:
                                display(Audio(pv))
                        link = _download_link(path_str)
                        if link:
                            with dl_area:
                                display(link)

                def _on_stem(change):
                    _update_player(change["new"])
                stem_dd.observe(_on_stem, names="value")

                _update_player(opts[0][1])

                display(widgets.VBox([stem_dd, player, dl_area]))

        except Exception as exc:
            with log_out:
                print(f"\n❌ {exc}")
            progress.bar_style = "danger"
            progress_label.value = (
                f'<span style="font-size:13px;color:red;">❌ Failed</span>'
            )
        finally:
            run_btn.disabled = False
            run_btn.description = "Separate"
            run_btn.button_style = "primary"

    run_btn.on_click(_run)

    # --- Layout ---
    display(widgets.VBox([
        header,
        widgets.HBox([file_dd, scan_btn]),
        input_text,
        task_dd,
        widgets.HBox([fmt_dd, fp16_cb]),
        out_text,
        run_btn,
        log_out,
        result_out,
    ], layout=widgets.Layout(max_width="700px", padding="10px")))


def render_ui():
    """Convenience alias."""
    create_uvr_ui()


if __name__ == "__main__":
    create_uvr_ui()
