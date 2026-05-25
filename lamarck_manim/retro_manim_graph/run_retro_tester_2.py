#!/usr/bin/env python3
"""
Python script to run retro_tester_2.py scene with version management.

This script replicates the bash command functionality:
- Increments version number
- Runs manim with --disable_caching
- Opens the output image if it exists

Usage (from project root):
    python retro_manim_graph/run_retro_tester_2.py
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

try:
    from rich.console import Console
    from rich.live import Live
    from rich.text import Text
    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False

# Stable filenames for "current" and "previous" viewers (used to find existing viewer processes)
CURRENT_FILENAME = "MyRetro3D_current.png"
PREVIOUS_FILENAME = "MyRetro3D_previous.png"

if _HAS_RICH:
    console = Console()
else:
    def _strip_rich(s) -> str:
        return re.sub(r"\[/?[^\]]+\]", "", str(s))
    class _PlainConsole:
        @staticmethod
        def print(*args, **kwargs):
            stripped = tuple(_strip_rich(a) for a in args) if args else args
            print(*stripped, **kwargs)
    console = _PlainConsole()


def _status_text(msg: str):
    """Orange-styled for Rich, plain string fallback."""
    if _HAS_RICH:
        return Text(msg, style="orange1")
    return msg


def _live_context(initial_msg):
    """Rich Live or no-op context manager."""
    if _HAS_RICH:
        return Live(_status_text(initial_msg), refresh_per_second=4, console=console)
    class _NoOpLive:
        def __enter__(self):
            self.live = type("Live", (), {"update": lambda s, x: None})()
            return self.live
        def __exit__(self, *args):
            pass
    return _NoOpLive()


def _venv_has_manim(venv_python: str, project_root: Path) -> bool:
    """Return True if the venv's python can run manim."""
    try:
        r = subprocess.run(
            [venv_python, "-m", "manim", "--version"],
            capture_output=True,
            timeout=5,
            cwd=project_root,
        )
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _find_venv_python(project_root: Path) -> str | None:
    """
    Find Python that can run manim. On Linux/WSL, prefer Linux-style venv (bin/python)
    to avoid path/regex issues when Windows Python in a cross-platform venv sees
    \\wsl.localhost\\Ubuntu paths. Fallback: python3 -m manim if manim is installed.
    """
    # On Linux, Windows-style venv (Scripts/) causes re.error when paths contain \\Ubuntu
    is_linux = sys.platform in ("linux", "darwin")
    for venv_name in ("manim.venv", "manim-linux.venv", ".venv", "venv"):
        venv = project_root / venv_name
        venv_python = venv / "bin" / "python"
        if venv_python.exists():
            if _venv_has_manim(str(venv_python), project_root):
                return str(venv_python)
            continue  # Venv exists but no manim - try next or fall through to _ensure_linux_venv
        # On Linux, avoid Windows venv - it breaks with WSL paths
        if not is_linux and (venv / "Scripts" / "python.exe").exists():
            win_python = str(venv / "Scripts" / "python.exe")
            if _venv_has_manim(win_python, project_root):
                return win_python
    # On Linux, try system python3 (user may have pip install manim)
    if is_linux:
        try:
            r = subprocess.run(
                [sys.executable, "-m", "manim", "--version"],
                capture_output=True,
                timeout=5,
                cwd=project_root,
            )
            if r.returncode == 0:
                return sys.executable
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    return None


def _ensure_linux_venv(project_root: Path) -> str | None:
    """On Linux, create manim-linux.venv if we have no working Linux venv. Returns path to python or None."""
    if sys.platform not in ("linux", "darwin"):
        return None
    venv_path = project_root / "manim-linux.venv"
    venv_python = venv_path / "bin" / "python"
    if venv_python.exists():
        # Verify venv has manim (venv may exist but be broken if ensurepip failed)
        try:
            r = subprocess.run(
                [str(venv_python), "-m", "manim", "--version"],
                capture_output=True,
                timeout=5,
                cwd=project_root,
            )
            if r.returncode == 0:
                return str(venv_python)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    # Create venv and install manim (remove broken venv if it exists)
    req_file = project_root / "requirements.txt"
    if venv_path.exists():
        console.print("[yellow]Removing incomplete manim-linux.venv and recreating...[/yellow]")
        shutil.rmtree(venv_path, ignore_errors=True)
    console.print("[yellow]Creating manim-linux.venv for WSL/Linux (avoids path issues). This may take a minute...[/yellow]")
    try:
        r = subprocess.run(
            [sys.executable, "-m", "venv", str(venv_path)],
            cwd=project_root,
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            err = (r.stderr or r.stdout or "").strip()
            if "ensurepip" in err.lower() or "python3-venv" in err.lower():
                console.print("[red]Install python3-venv first: sudo apt install python3.12-venv[/red]")
            else:
                console.print(f"[red]venv failed: {err}[/red]")
            return None
        pip_cmd = [str(venv_python), "-m", "pip", "install", "manim"]
        if req_file.exists():
            pip_cmd = [str(venv_python), "-m", "pip", "install", "-r", str(req_file)]
        subprocess.run(pip_cmd, cwd=project_root, check=True, capture_output=True)
        console.print(f"[green]Created {venv_path}[/green]")
        return str(venv_python)
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Failed to install manim in venv: {e}[/red]")
        return None


def _path_for_manim(path: Path) -> str:
    """Normalize path to forward slashes for manim. Avoids re.error on WSL when path contains \\Ubuntu etc."""
    return str(path.resolve()).replace("\\", "/")


def _collapse_log_line(parts: list[str]) -> str:
    """Turn accumulated log parts into one line. Merges single-char 'words' and strips spaces inside quoted paths (manim wraps when stdout is a pipe)."""
    tokens = " ".join(parts).split()
    if not tokens:
        return ""
    result = [tokens[0]]
    for t in tokens[1:]:
        if len(t) == 1 and result and len(result[-1]) == 1:
            result[-1] += t
        else:
            result.append(t)
    line = " ".join(result)
    # Remove spaces inside quoted paths: manim wraps paths and produces '/home /cody/ ...'
    def _despace_quoted_path(m: re.Match) -> str:
        quote, inner = m.group(1), m.group(2)
        if "/" in inner or "\\" in inner:
            inner = inner.replace(" ", "")
        return quote + inner + quote
    line = re.sub(r"(['\"])([^'\"]*)\1", _despace_quoted_path, line)
    return line


def _run_manim_and_stream_stdout(manim_cmd: list[str], cwd: Path, env: dict | None) -> int:
    """Run manim, stream stdout/stderr, collapse multi-line indented log lines into single lines."""
    proc = subprocess.Popen(
        manim_cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    parts: list[str] = []
    for line in proc.stdout:
        if not line.endswith("\n"):
            line = line + "\n"
        raw = line
        stripped = line.rstrip("\n").strip()
        if not stripped:
            if parts:
                console.print(_collapse_log_line(parts))
                parts = []
            continue
        if raw.startswith((" ", "\t")):
            parts.append(stripped)
        else:
            if parts:
                console.print(_collapse_log_line(parts))
            parts = [stripped]
    if parts:
        console.print(_collapse_log_line(parts))
    return proc.wait()




def is_viewer_open_for_file(filepath):
    """
    Return True if a viewer process appears to have this file open.
    Uses the filename in process command lines so we can reuse existing viewers
    instead of opening new ones (e.g. across script restarts).
    """
    name = Path(filepath).name
    try:
        if sys.platform in ("linux", "darwin"):
            r = subprocess.run(
                ["pgrep", "-f", re.escape(name)],
                capture_output=True,
                timeout=2,
            )
            return r.returncode == 0
        if sys.platform == "win32":
            r = subprocess.run(
                ["tasklist", "/V", "/FO", "CSV"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if r.returncode != 0:
                return False
            return name in (r.stdout or "")
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        return False


def detect_color_scheme(scene_file_path):
    """
    Detect color scheme preset from scene file by searching for color_scheme patterns.
    
    Returns:
        str or None: Detected color scheme name (or None if not found)
    """
    try:
        # Import color scheme presets and aliases
        sys.path.insert(0, str(scene_file_path.parent))
        from retro_configuration import COLOR_SCHEME_PRESETS, COLOR_SCHEME_ALIASES
        
        # Read scene file content
        with open(scene_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Search for color_scheme patterns (handles both 'color_scheme' and 'COLOR_SCHEME')
        # Look for patterns like: 'color_scheme': 'bw' or "color_scheme": "wb" etc.
        patterns = [
            r"['\"]color_scheme['\"]\s*:\s*['\"]([^'\"]+)['\"]",
            r"['\"]COLOR_SCHEME['\"]\s*:\s*['\"]([^'\"]+)['\"]",
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content)
            if matches:
                scheme = matches[0].strip()
                # Resolve alias if needed
                resolved_scheme = COLOR_SCHEME_ALIASES.get(scheme, scheme)
                if resolved_scheme in COLOR_SCHEME_PRESETS:
                    return resolved_scheme
        
        return None
    except Exception as e:
        # If detection fails, just return None (don't break the script)
        return None


def main():
    # Get the package directory and project root (parent of retro_manim_graph)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    os.chdir(project_root)

    # Version file path (in package directory)
    version_file = script_dir / ".version"

    # Read and increment version
    if version_file.exists():
        with open(version_file, 'r') as f:
            version = int(f.read().strip()) + 1
    else:
        version = 1

    # Write new version
    with open(version_file, 'w') as f:
        f.write(str(version))

    # Persistent status line at bottom; console.print scrolls above it (no timestamp/indent)
    with _live_context("Starting...") as live:
        console.print(f"Version: v{version}")

        # Scene file and class name (in package)
        scene_file = "retro_manim_graph/retro_tester_2.py"
        scene_file_path = project_root / scene_file
        scene_class = "SimpleRetroScene"
        output_name = f"MyRetro3D_v{version}"

        # Detect and print color scheme if present
        color_scheme = detect_color_scheme(scene_file_path)
        if color_scheme:
            console.print(f"Color scheme: {color_scheme}")
        else:
            console.print("Color scheme: default (not specified)")

        # Run manim: use venv python -m manim (avoids PATH), pass path with forward slashes (avoids re.error on WSL)
        venv_python = _find_venv_python(project_root)
        if venv_python is None and sys.platform in ("linux", "darwin"):
            venv_python = _ensure_linux_venv(project_root)
        if venv_python is None and sys.platform in ("linux", "darwin"):
            console.print(
                "[red]No manim Python found. On WSL/Linux:[/red]\n"
                "  1. Install venv: [bold]sudo apt install python3.12-venv[/bold]\n"
                "  2. Create venv: [bold]python3 -m venv manim-linux.venv[/bold]\n"
                "  3. Install deps: [bold]manim-linux.venv/bin/pip install -r requirements.txt[/bold]"
            )
            sys.exit(1)
        scene_file_for_manim = _path_for_manim(scene_file_path)
        if venv_python:
            manim_cmd = [
                venv_python,
                "-m", "manim",
                "-s",
                "--disable_caching",
                scene_file_for_manim,
                scene_class,
                "-o", output_name,
            ]
        else:
            manim_cmd = [
                "manim",
                "-s",
                "--disable_caching",
                scene_file_for_manim,
                scene_class,
                "-o", output_name,
            ]

        console.print(f"Running: {' '.join(manim_cmd)}")
        live.update(_status_text("Status: running manim..."))
        # Pass real terminal size so manim (and its libs) don't see a tiny width from Rich's Live
        try:
            cols, lines = shutil.get_terminal_size()
            manim_env = {**os.environ, "COLUMNS": str(cols), "LINES": str(lines)}
        except OSError:
            manim_env = None
        returncode = _run_manim_and_stream_stdout(manim_cmd, project_root, manim_env)

        if returncode != 0:
            console.print(f"Error: Manim command failed with exit code {returncode}")
            live.update(_status_text("Status: manim failed"))
            sys.exit(1)

        # Manim writes to media/images/<scene_file_stem>/ (scene file name without path)
        images_dir = project_root / "media" / "images" / scene_file_path.stem
        output_path = images_dir / f"{output_name}.png"
        current_path = images_dir / CURRENT_FILENAME
        previous_path = images_dir / PREVIOUS_FILENAME

        def open_image(path):
            """Open path in default viewer (background, non-blocking)."""
            kwargs = {}
            if sys.platform != "win32":
                kwargs["start_new_session"] = True
            if sys.platform == "linux":
                subprocess.Popen(["xdg-open", str(path)], **kwargs)
            elif sys.platform == "win32":
                subprocess.Popen(
                    ["start", "", str(path)],
                    shell=True,
                    creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
                )
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)], **kwargs)

        if output_path.exists():
            live.update(_status_text("Status: updating images..."))
            # Roll current -> previous, then new output -> current (so both viewers refresh)
            if current_path.exists():
                shutil.copy2(current_path, previous_path)
            shutil.copy2(output_path, current_path)

            # Reuse existing viewer processes if they're already showing these files
            current_viewer_already_open = is_viewer_open_for_file(current_path)
            previous_viewer_already_open = (
                previous_path.exists() and is_viewer_open_for_file(previous_path)
            )

            if not current_viewer_already_open:
                console.print(f"Opening current (v{version}): {current_path}")
                open_image(current_path)
            else:
                console.print(f"Updated current (v{version}): {current_path} (existing viewer should refresh)")

            if previous_path.exists():
                prev_version = version - 1
                if not previous_viewer_already_open:
                    console.print(f"Opening previous (v{prev_version}): {previous_path}")
                    open_image(previous_path)
                else:
                    console.print(f"Updated previous (v{prev_version}): {previous_path} (existing viewer should refresh)")
        else:
            console.print(f"Warning: Output image not found at {output_path}")

        live.update(_status_text("Status: ready"))

    response = input("\nRun again? [y/N]: ").strip().lower()
    return response in ('y', 'yes')


if __name__ == "__main__":
    while main():
        pass
