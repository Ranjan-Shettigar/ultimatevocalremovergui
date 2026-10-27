"""
Minimalistic ipywidgets UI for UVR5 audio separation in Kaggle / Jupyter.
"""

import re
import shutil
import subprocess
import zipfile
from pathlib import Path

import ipywidgets as widgets
from IPython.display import display, Audio, HTML

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
    """
    Generate a working download link for Kaggle/Jupyter.
    Links the file directly into the notebook directory with relative href
    to avoid Kaggle proxy 404 errors, and provides an optional ZIP link.
    """
    p = Path(path_str)
    if not p.exists():
        return None

    # Link or copy file into current notebook directory so relative URL resolves cleanly
    local_file = Path(p.name)
    if not local_file.exists() or local_file.resolve() != p.resolve():
        try:
            local_file.unlink(missing_ok=True)
            local_file.symlink_to(p)
        except Exception:
            try:
                shutil.copy2(p, local_file)
            except Exception:
                pass

    # Also prepare a zip file in the current directory for 100% reliable Kaggle downloading
    zip_file = Path(f"{p.stem}.zip")
    if not zip_file.exists():
        try:
            with zipfile.ZipFile(zip_file, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.write(p, arcname=p.name)
        except Exception:
            pass

    size_mb = p.stat().st_size / (1024 * 1024)
    links_html = f"""
    <div style="margin-top:8px;font-size:13px;line-height:1.8;">
        <span>⬇ <b>Download:</b> </span>
        <a href="{local_file.name}" download="{local_file.name}" target="_blank" style="color:#2563eb;text-decoration:underline;font-weight:500;margin-right:10px;">
            {local_file.name} ({size_mb:.1f} MB)
        </a>
    """
    if zip_file.exists():
        zip_mb = zip_file.stat().st_size / (1024 * 1024)
        links_html += f"""
        <span style="color:#94a3b8;">|</span>
        <a href="{zip_file.name}" download="{zip_file.name}" target="_blank" style="color:#2563eb;text-decoration:underline;margin-left:10px;">
            📦 Download as ZIP ({zip_mb:.1f} MB)
        </a>
        """

    links_html += f"""
        <div style="color:#64748b;font-size:11px;margin-top:4px;">
            💡 You can also download from Kaggle's right sidebar under <b>Data ➔ Output (/kaggle/working)</b>.
        </div>
    </div>
    """
    return HTML(links_html)


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

    # --- Run button & progress bar ---
    run_btn = widgets.Button(
        description="Separate", button_style="primary", icon="play",
        layout=widgets.Layout(width="140px", height="36px"),
    )
    progress_bar = widgets.IntProgress(
        value=0,
        min=0,
        max=100,
        bar_style="info",
        orientation="horizontal",
        layout=widgets.Layout(width="96%", height="16px", margin="8px 0 2px 0", display="none"),
    )
    progress_status = widgets.HTML(
        "",
        layout=widgets.Layout(margin="0 0 4px 0"),
    )

    # --- Output areas ---
    log_out = widgets.Output(layout=widgets.Layout(
        max_height="240px", overflow="auto",
        border="1px solid #ddd", padding="8px", margin="8px 0",
    ))
    result_out = widgets.Output()

    # --- Run logic ---
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

        # Determine total steps
        if task in {"vocal", "instrumental", "stems4", "stems6", "karaoke"}:
            total_steps = 1
        elif task == "cleanup":
            total_steps = 2
        elif task == "all":
            total_steps = 8
        else:
            total_steps = 1

        step_state = {
            "current": 1,
            "total": total_steps,
            "model": "",
        }

        progress_bar.layout.display = "block"
        progress_bar.bar_style = "info"
        progress_bar.value = 0
        progress_status.value = "<div style='font-size:12px;color:#2563eb;'>⏳ Starting separation...</div>"

        out_dir.mkdir(parents=True, exist_ok=True)
        ac = ["--use_autocast"] if fp16 else []
        created = []

        def _on_progress(line):
            m = re.search(r"(\d+)%", line)
            if m:
                pct = int(m.group(1))
                cur = step_state["current"]
                tot = step_state["total"]
                step_frac = 100.0 / max(1, tot)
                overall = int((cur - 1) * step_frac + (pct * step_frac / 100.0))
                overall = min(100, max(0, overall))

                progress_bar.value = overall
                m_name = step_state["model"]
                if tot > 1:
                    progress_status.value = (
                        f"<div style='font-size:12px;color:#2563eb;'>"
                        f"Step {cur}/{tot} ({m_name}): <b>{pct}%</b> "
                        f"<span style='color:#666;'>(Overall: {overall}%)</span></div>"
                    )
                else:
                    progress_status.value = (
                        f"<div style='font-size:12px;color:#2563eb;'>"
                        f"Separating ({m_name}): <b>{pct}%</b></div>"
                    )
            else:
                low = line.lower()
                if "loading model" in low or "downloading" in low:
                    progress_status.value = f"<div style='font-size:12px;color:#2563eb;'>⏳ {line}</div>"
                    print(line)
                elif "saving" in low or "writing output" in low or "using pydub" in low:
                    progress_status.value = f"<div style='font-size:12px;color:#2563eb;'>💾 Saving audio output...</div>"
                    print(line)
                elif any(k in low for k in ["duration", "complete", "error", "failed"]):
                    print(line)
                elif "|" not in line and not re.search(r"\d+/\d+", line):
                    print(line)

        def _sep(f, alias, stem=None, extra=None):
            step_state["model"] = alias
            d = separate(
                input_file=f, model_alias=alias, output_dir=out_dir,
                single_stem=stem, extra_args=extra,
                output_format=fmt, use_soundfile=False,
                on_progress=_on_progress,
            )
            if d:
                created.append(Path(d))
            step_state["current"] += 1
            return d

        try:
            with log_out:
                print(f"Input: {inp.name}  |  Task: {task}  |  Format: {fmt}")
                print("-" * 40)

                if task in {"vocal", "all"}:
                    _sep(inp, "bs_roformer", extra=ac)
                if task == "all":
                    _sep(inp, "kim_vocal_2", stem="Vocals", extra=ac)
                if task in {"instrumental", "all"}:
                    _sep(inp, "inst_hq_3", stem="Instrumental", extra=ac)
                if task == "all":
                    _sep(inp, "mdx23c_instvoc",
                         extra=["--mdxc_segment_size", "256", "--mdxc_overlap", "8", *ac])
                if task in {"stems4", "all"}:
                    _sep(inp, "htdemucs",
                         extra=["--demucs_shifts", "1", "--demucs_overlap", "0.25",
                                "--demucs_segments_enabled", "True", *ac])
                if task in {"stems6", "all"}:
                    _sep(inp, "demucs_6stem",
                         extra=["--demucs_shifts", "1", "--demucs_overlap", "0.25",
                                "--demucs_segments_enabled", "True", *ac])
                if task in {"cleanup", "all"}:
                    vf = find_stem(out_dir / "bs_roformer", ["vocal"])
                    if vf:
                        _sep(vf, "bip_reverb", stem="Vocals", extra=ac)
                        _sep(vf, "deecho", stem="Vocals", extra=ac)
                    else:
                        print("⚠ No vocal stem found for cleanup.")
                if task in {"karaoke", "all"}:
                    inf = find_stem(out_dir / "bs_roformer", ["instrumental"])
                    if inf:
                        _sep(inf, "kara_2", stem="Instrumental", extra=ac)
                    else:
                        print("⚠ No instrumental stem found for karaoke.")

                print("\n✅ Done.")

            progress_bar.value = 100
            progress_bar.bar_style = "success"
            progress_status.value = "<div style='font-size:12px;color:#16a34a;font-weight:600;'>✅ Complete (100%)</div>"

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
                all_zip_html = ""
                if len(stems) > 1:
                    all_zip = Path("all_stems.zip")
                    try:
                        with zipfile.ZipFile(all_zip, "w", zipfile.ZIP_DEFLATED) as zf:
                            for s in stems:
                                zf.write(s, arcname=s.name)
                        zip_mb = all_zip.stat().st_size / (1024 * 1024)
                        all_zip_html = (
                            f"<div style='margin: 8px 0 10px 0;'>"
                            f"<a href='{all_zip.name}' download='{all_zip.name}' target='_blank' "
                            f"style='display:inline-block;padding:6px 14px;background:#2563eb;color:#fff;"
                            f"border-radius:6px;text-decoration:none;font-size:12px;font-weight:500;'>"
                            f"📦 Download All Stems as ZIP ({zip_mb:.1f} MB)</a></div>"
                        )
                    except Exception:
                        pass

                display(HTML(
                    f"<details open style='margin:8px 0;'>"
                    f"<summary style='cursor:pointer;font-weight:600;'>Output files ({len(stems)})</summary>"
                    f"<table style='font-size:13px;margin-top:4px;'>{rows}</table></details>"
                    f"{all_zip_html}"
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
            progress_bar.bar_style = "danger"
            progress_status.value = f"<div style='font-size:12px;color:#dc2626;font-weight:600;'>❌ Failed: {exc}</div>"
            with log_out:
                print(f"\n❌ {exc}")
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
        progress_bar,
        progress_status,
        log_out,
        result_out,
    ], layout=widgets.Layout(max_width="700px", padding="10px")))


def render_ui():
    """Convenience alias."""
    create_uvr_ui()


if __name__ == "__main__":
    create_uvr_ui()
