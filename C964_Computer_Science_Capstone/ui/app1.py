"""
Simple browser UI for running visualization experiments and viewing artifacts.

Run:
    python ui/app.py

Then open:
    http://127.0.0.1:8765
"""

from __future__ import annotations

import html
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
import threading
from urllib.parse import parse_qs, quote, unquote, urlparse


HOST = "127.0.0.1"
PORT = 8765
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MEDIA_ROOT = PROJECT_ROOT / "optimizationlab" / "_Experimental_Media"
VIS_SCRIPT = PROJECT_ROOT / "optimizationlab" / "visualize_experiment.py"


@dataclass
class RunArtifacts:
    experiment: str
    run_started_at: str
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    experiment_id: Optional[str]
    main_image: Optional[Path]
    bar_chart: Optional[Path]
    trend_svgs: list[Path]
    evaluation_report: Optional[Path]


_LAST_RUN: Optional[RunArtifacts] = None
_RUN_LOCK = threading.Lock()


@dataclass
class LiveRun:
    experiment: str
    run_started_at: str
    command: list[str]
    running: bool
    output_lines: list[str]
    returncode: Optional[int] = None


_CURRENT_RUN: Optional[LiveRun] = None


def _safe_rel_path(path: Path) -> str:
    rel = path.resolve().relative_to(PROJECT_ROOT.resolve())
    return str(rel).replace("\\", "/")


def _find_latest_file(directory: Path, pattern: str) -> Optional[Path]:
    files = list(directory.glob(pattern))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def _extract_experiment_id_from_bar_chart(bar_path: Optional[Path]) -> Optional[str]:
    if bar_path is None:
        return None
    suffix = "_best_candidates_bar"
    stem = bar_path.stem
    return stem[:-len(suffix)] if stem.endswith(suffix) else None


def _collect_artifacts_for_experiment(exp: str) -> tuple[Optional[str], Optional[Path], Optional[Path], list[Path], Optional[Path]]:
    images_dir = MEDIA_ROOT / "images"
    bars_dir = MEDIA_ROOT / "bar_charts"
    trends_dir = MEDIA_ROOT / "meta_trends"
    reports_dir = MEDIA_ROOT / "evaluation_reports"

    bar = _find_latest_file(bars_dir, f"{exp}_*_best_candidates_bar.svg")
    experiment_id = _extract_experiment_id_from_bar_chart(bar)

    main_image = None
    trend_svgs: list[Path] = []
    report = None
    if experiment_id:
        main_image = _find_latest_file(images_dir, f"{exp}_viz_*_{experiment_id}_*.png")
        trend_svgs = sorted(trends_dir.glob(f"{experiment_id}_meta_trend_seed_*.svg"))
        report = _find_latest_file(reports_dir, f"{experiment_id}_evaluation_report.txt")
    else:
        # Fallback: latest generic outputs for experiment prefix
        main_image = _find_latest_file(images_dir, f"{exp}_viz_*")
        report = _find_latest_file(reports_dir, f"{exp}_*_evaluation_report.txt")

    return experiment_id, main_image, bar, trend_svgs, report


def _build_command(form: dict[str, str]) -> tuple[str, list[str]]:
    exp = form.get("experiment", "md2").strip().lower()
    command = [sys.executable, str(VIS_SCRIPT), "--experiment", exp]

    seed = form.get("seed", "").strip()
    seeds = form.get("seeds", "").strip()
    calls = form.get("calls", "").strip()
    if seed:
        command += ["--seed", seed]
    if seeds:
        command += ["--seeds", seeds]
    if calls:
        command += ["--calls", calls]
    if form.get("quiet") == "on":
        command.append("--quiet")
    if form.get("meta_verbose") == "on":
        command.append("--meta-verbose")
    return exp, command


def _run_visualization(form: dict[str, str]) -> RunArtifacts:
    exp, command = _build_command(form)

    run_started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    proc = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    experiment_id, main_image, bar, trends, report = _collect_artifacts_for_experiment(exp)

    return RunArtifacts(
        experiment=exp,
        run_started_at=run_started_at,
        command=command,
        returncode=proc.returncode,
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
        experiment_id=experiment_id,
        main_image=main_image,
        bar_chart=bar,
        trend_svgs=trends,
        evaluation_report=report,
    )


def _run_visualization_background(form: dict[str, str]) -> None:
    global _LAST_RUN, _CURRENT_RUN

    exp, command = _build_command(form)
    run_started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    live = LiveRun(
        experiment=exp,
        run_started_at=run_started_at,
        command=command,
        running=True,
        output_lines=[],
    )
    with _RUN_LOCK:
        _CURRENT_RUN = live

    proc = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    lines: list[str] = []
    if proc.stdout is not None:
        for raw in proc.stdout:
            line = raw.rstrip("\n")
            lines.append(line)
            with _RUN_LOCK:
                if _CURRENT_RUN is not None:
                    _CURRENT_RUN.output_lines.append(line)
                    # Keep recent output bounded for UI responsiveness.
                    if len(_CURRENT_RUN.output_lines) > 1200:
                        _CURRENT_RUN.output_lines = _CURRENT_RUN.output_lines[-1200:]

    returncode = proc.wait()
    experiment_id, main_image, bar, trends, report = _collect_artifacts_for_experiment(exp)
    final = RunArtifacts(
        experiment=exp,
        run_started_at=run_started_at,
        command=command,
        returncode=returncode,
        stdout="\n".join(lines),
        stderr="",
        experiment_id=experiment_id,
        main_image=main_image,
        bar_chart=bar,
        trend_svgs=trends,
        evaluation_report=report,
    )
    with _RUN_LOCK:
        _LAST_RUN = final
        if _CURRENT_RUN is not None:
            _CURRENT_RUN.running = False
            _CURRENT_RUN.returncode = returncode
            _CURRENT_RUN.output_lines = lines[-1200:]


def _render_page() -> str:
    global _LAST_RUN, _CURRENT_RUN
    with _RUN_LOCK:
        last = _LAST_RUN
        current = _CURRENT_RUN

    status_html = "<p>No run yet. Submit the form to generate artifacts.</p>"
    artifacts_html = ""
    logs_html = ""
    auto_refresh_meta = ""

    if current is not None and current.running:
        auto_refresh_meta = "<meta http-equiv='refresh' content='2'>"
        status_html = (
            f"<p><strong>Current run:</strong> {html.escape(current.run_started_at)} | "
            f"<strong>Experiment:</strong> {html.escape(current.experiment.upper())} | "
            "<strong>Status:</strong> RUNNING</p>"
            f"<p><strong>Command:</strong> <code>{html.escape(' '.join(current.command))}</code></p>"
        )
        current_output = "\n".join(current.output_lines[-400:])
        logs_html = (
            "<h3>Live Script Output</h3>"
            "<p>Auto-refreshing every 2 seconds while running.</p>"
            f"<pre style='border:1px solid var(--border); padding:10px; overflow:auto; max-height:420px;'>{html.escape(current_output)}</pre>"
        )

    if last is not None:
        status = "SUCCESS" if last.returncode == 0 else f"FAILED ({last.returncode})"
        status_html = (
            f"<p><strong>Last run:</strong> {html.escape(last.run_started_at)} | "
            f"<strong>Experiment:</strong> {html.escape(last.experiment.upper())} | "
            f"<strong>Status:</strong> {html.escape(status)} | "
            f"<strong>Experiment ID:</strong> {html.escape(last.experiment_id or 'not found')}</p>"
            f"<p><strong>Command:</strong> <code>{html.escape(' '.join(last.command))}</code></p>"
        )

        def img_or_missing(path: Optional[Path], label: str) -> str:
            if path is None:
                return f"<div><h3>{label}</h3><p>Not found for last run.</p></div>"
            href = f"/artifact?path={quote(_safe_rel_path(path))}"
            return (
                f"<div><h3>{label}</h3>"
                f"<p><code>{html.escape(_safe_rel_path(path))}</code></p>"
                f"<img src='{href}' style='max-width:100%; border:1px solid var(--border); padding:4px;'/>"
                f"</div>"
            )

        parts = [
            img_or_missing(last.main_image, "1) Main Visualization Image"),
            img_or_missing(last.bar_chart, "2) Best-Candidate Bar Chart"),
        ]
        if last.trend_svgs:
            for i, trend in enumerate(last.trend_svgs, start=1):
                parts.append(img_or_missing(trend, f"3) Meta Trend SVG #{i}"))
        else:
            parts.append("<div><h3>3) Meta Trend SVG(s)</h3><p>Not found for last run.</p></div>")

        if last.evaluation_report is not None and last.evaluation_report.is_file():
            report_text = last.evaluation_report.read_text(encoding="utf-8")
            parts.append(
                "<div><h3>Evaluation Report</h3>"
                f"<p><code>{html.escape(_safe_rel_path(last.evaluation_report))}</code></p>"
                f"<pre style='border:1px solid var(--border); padding:10px; overflow:auto;'>{html.escape(report_text)}</pre>"
                "</div>"
            )

        artifacts_html = "\n".join(parts)
        # Show completed logs only when no run is currently active.
        if not (current is not None and current.running):
            logs_html = (
                "<h3>Run Logs</h3>"
                f"<details open><summary>stdout</summary><pre>{html.escape(last.stdout)}</pre></details>"
                f"<details><summary>stderr</summary><pre>{html.escape(last.stderr)}</pre></details>"
            )

    return f"""<!doctype html>
<html data-theme="dark">
<head>
  <meta charset="utf-8" />
  {auto_refresh_meta}
  <title>Capstone Visualization UI</title>
  <style>
    :root {{
      --bg: #0f1115;
      --fg: #e8edf2;
      --muted: #9aa7b4;
      --panel: #161b22;
      --border: #2a3440;
      --code-bg: #1f2630;
      --button-bg: #2a3440;
      --button-fg: #e8edf2;
      --link: #8ec5ff;
    }}
    html[data-theme="light"] {{
      --bg: #ffffff;
      --fg: #111111;
      --muted: #4f5b66;
      --panel: #f7f9fb;
      --border: #d0d7de;
      --code-bg: #eef2f6;
      --button-bg: #e6ebf1;
      --button-fg: #111111;
      --link: #0969da;
    }}
    body {{
      font-family: Arial, sans-serif;
      margin: 20px;
      background: var(--bg);
      color: var(--fg);
    }}
    .grid {{ display:grid; grid-template-columns: 1fr; gap: 20px; }}
    input, select {{
      padding: 6px;
      margin: 4px 0;
      background: var(--panel);
      color: var(--fg);
      border: 1px solid var(--border);
    }}
    button {{
      padding: 8px 14px;
      background: var(--button-bg);
      color: var(--button-fg);
      border: 1px solid var(--border);
      cursor: pointer;
    }}
    code {{ background: var(--code-bg); padding:2px 4px; }}
    pre {{
      background: var(--panel);
      border: 1px solid var(--border);
      padding: 10px;
      overflow: auto;
    }}
    .topbar {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
    }}
    .muted {{ color: var(--muted); }}
    img {{ background: #fff; }}
    a {{ color: var(--link); }}
  </style>
</head>
<body>
  <div class="topbar">
    <h1>Visualization Experiment UI</h1>
    <button type="button" id="theme-toggle" aria-label="Toggle dark mode">Switch to light mode</button>
  </div>
  <p>Run <code>optimizationlab/visualize_experiment.py</code> with interactive inputs and inspect generated artifacts.</p>

  <h2>Run Experiment</h2>
  <form method="post" action="/run">
    <label>Experiment:
      <select name="experiment">
        <option value="md2">md2</option>
        <option value="up1">up1</option>
        <option value="mdsingle">mdsingle</option>
        <option value="ld4">ld4</option>
      </select>
    </label><br/>
    <label>Seed (single): <input name="seed" placeholder="e.g., 7"/></label><br/>
    <label>Seeds (comma list): <input name="seeds" placeholder="e.g., 7,27,107"/></label><br/>
    <label>Call budget: <input name="calls" placeholder="e.g., 300"/></label><br/>
    <label><input type="checkbox" name="quiet"/> quiet</label>
    <label><input type="checkbox" name="meta_verbose"/> meta_verbose</label><br/>
    <button type="submit">Run and Refresh Artifacts</button>
  </form>

  <h2>Status</h2>
  {status_html}

  <h2>Artifacts</h2>
  <div class="grid">
    {artifacts_html}
  </div>
  {logs_html}
  <script>
    (function () {{
      const key = "capstoneTheme";
      const root = document.documentElement;
      const btn = document.getElementById("theme-toggle");
      const saved = localStorage.getItem(key);
      const theme = saved === "light" || saved === "dark" ? saved : "dark";
      root.setAttribute("data-theme", theme);

      function updateLabel() {{
        const current = root.getAttribute("data-theme");
        btn.textContent = current === "dark" ? "Switch to light mode" : "Switch to dark mode";
      }}
      updateLabel();

      btn.addEventListener("click", function () {{
        const current = root.getAttribute("data-theme");
        const next = current === "dark" ? "light" : "dark";
        root.setAttribute("data-theme", next);
        localStorage.setItem(key, next);
        updateLabel();
      }});
    }})();
  </script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            body = _render_page().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if parsed.path == "/artifact":
            qs = parse_qs(parsed.query)
            rel = unquote((qs.get("path") or [""])[0])
            if not rel:
                self.send_error(400, "Missing path query param")
                return
            p = (PROJECT_ROOT / rel).resolve()
            try:
                p.relative_to(PROJECT_ROOT.resolve())
            except ValueError:
                self.send_error(403, "Path outside project")
                return
            if not p.is_file():
                self.send_error(404, "Artifact not found")
                return
            data = p.read_bytes()
            ctype = "application/octet-stream"
            suffix = p.suffix.lower()
            if suffix == ".png":
                ctype = "image/png"
            elif suffix == ".svg":
                ctype = "image/svg+xml"
            elif suffix == ".txt":
                ctype = "text/plain; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        self.send_error(404, "Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/run":
            self.send_error(404, "Not found")
            return

        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8", errors="replace")
        raw = parse_qs(body)
        form = {k: (v[0] if v else "") for k, v in raw.items()}

        global _CURRENT_RUN
        with _RUN_LOCK:
            running_now = _CURRENT_RUN is not None and _CURRENT_RUN.running
        if not running_now:
            t = threading.Thread(target=_run_visualization_background, args=(form,), daemon=True)
            t.start()

        payload = b""
        self.send_response(303)
        self.send_header("Location", "/")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        # Keep terminal output concise while running local UI.
        return


def main() -> None:
    print(f"Starting UI at http://{HOST}:{PORT}")
    print("Press Ctrl+C to stop.")
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        print("UI server stopped.")


if __name__ == "__main__":
    main()

