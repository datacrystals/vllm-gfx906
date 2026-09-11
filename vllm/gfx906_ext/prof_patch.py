# SPDX-License-Identifier: Apache-2.0
# On-demand torch.profiler harness for the gfx906 production fork.
#
# Vendored as vllm/gfx906_ext/prof_patch.py (supersedes
# /data/vllm-gfx906-dsv4/patches/gdn/prof_patch.py, provenance only).
# Imported by the VLLM_GFX906_PROF_DIR anchors in hy_v3.py /
# glm5next/__init__.py, or from gdn_gfx906_fallback.py install path,
# iff VLLM_GFX906_PROF_DIR is set. Wraps Worker.execute_model and profiles per
# worker process, driven by a trigger file so no server routes are needed.
#
# Env:
#   VLLM_GFX906_PROF_DIR   directory for traces + trigger file (required)
#   VLLM_GFX906_PROF_MAX_WINDOWS  safety cap on windows (default 8)
#
# Trigger protocol:
#   write {"$N"} i.e. a bare integer window id N into $DIR/TRIGGER.
#   At the next execute_model call every worker starts torch.profiler,
#   captures VLLM_GFX906_PROF_STEPS_BASE + optional per-window steps, stops,
#   and dumps trace_w{N}_p{pid}.json.gz in $DIR.
#   Steps per window: put "N:STEPS" (e.g. "2:40") to override steps for
#   that window; default 32.
#
# Safety: every exception is swallowed; a window left open at shutdown is
# harmless; profiling only ever STARTS from the trigger file, so with no
# trigger file the wrapper is one stat() per scheduler step.

import json
import os
import time

_INSTALLED = False


def _log(msg):
    print(f"[PROF] {msg}", flush=True)


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    prof_dir = os.environ.get("VLLM_GFX906_PROF_DIR", "")
    if not prof_dir:
        return
    try:
        os.makedirs(prof_dir, exist_ok=True)
    except Exception as e:
        _log(f"cannot make dir {prof_dir}: {e}")
        return
    try:
        from vllm.v1.worker.gpu_worker import Worker
    except Exception as e:
        _log(f"cannot import gpu_worker.Worker: {e}")
        return

    trigger_path = os.path.join(prof_dir, "TRIGGER")
    max_windows = int(os.environ.get("VLLM_GFX906_PROF_MAX_WINDOWS", "8"))
    pid = os.getpid()
    rank = os.environ.get("RANK", "")
    state = {
        "seen": 0,          # last consumed window id
        "done": 0,          # windows completed (cap)
        "prof": None,       # active profiler or None
        "steps_left": 0,
        "win": 0,
        "last_check": 0.0,
        "cached_mtime": 0.0,
    }

    orig_execute_model = Worker.execute_model

    def _read_trigger():
        try:
            with open(trigger_path) as f:
                txt = f.read().strip()
        except Exception:
            return None, None
        if ":" in txt:
            w, s = txt.split(":", 1)
            return int(w), int(s)
        return int(txt), 32

    def _maybe_trigger():
        # throttle stat/read to ~2 Hz; execute_model can run at 20+/s
        now = time.monotonic()
        if now - state["last_check"] < 0.5:
            return
        state["last_check"] = now
        try:
            mtime = os.stat(trigger_path).st_mtime
        except Exception:
            return
        if mtime == state["cached_mtime"]:
            return
        state["cached_mtime"] = mtime
        if state["prof"] is not None:
            return  # busy; will pick up next trigger afterwards
        if state["done"] >= max_windows:
            return
        try:
            win, steps = _read_trigger()
        except Exception as e:
            _log(f"bad trigger content at p{pid}: {e}")
            return
        if win is None or win <= state["seen"]:
            return
        state["seen"] = win
        try:
            import torch
            prof = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA],
                record_shapes=os.environ.get(
                    "VLLM_GFX906_PROF_SHAPES", "0") == "1",
                with_stack=False,
                profile_memory=False,
            )
            prof.start()
            state["prof"] = prof
            state["steps_left"] = max(1, steps)
            state["win"] = win
            _log(f"p{pid} rank={rank} window {win}: profiling {steps} steps")
        except Exception as e:
            _log(f"p{pid} profiler start failed: {e}")
            state["prof"] = None

    def _finish_step():
        state["steps_left"] -= 1
        if state["steps_left"] > 0:
            return
        prof = state["prof"]
        state["prof"] = None
        win = state["win"]
        try:
            prof.stop()
            out = os.path.join(prof_dir, f"trace_w{win}_p{pid}.json.gz")
            prof.export_chrome_trace(out)
            state["done"] += 1
            _log(f"p{pid} rank={rank} window {win}: wrote {out}")
        except Exception as e:
            _log(f"p{pid} profiler stop/export failed: {e}")

    def wrapped(self, *args, **kwargs):
        try:
            _maybe_trigger()
        except Exception:
            pass
        out = orig_execute_model(self, *args, **kwargs)
        if state["prof"] is not None:
            try:
                _finish_step()
            except Exception:
                state["prof"] = None
        return out

    Worker.execute_model = wrapped
    _log(f"installed on Worker.execute_model in p{pid} rank={rank} dir={prof_dir}")


install()
