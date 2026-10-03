#!/usr/bin/env python3
"""mimo_perf_bench.py - acceptance performance bench for a freshly booted
mimo-v2.6-flash server (OpenAI-compatible, /v1/completions).

Request conventions follow needle_probe.py in this directory: POST
/v1/completions with temperature 0 via urllib against 127.0.0.1:<port>.
Every subcommand takes --port (default 9700), --model (default
mimo-v2.6-flash) and --dry-run.

Subcommands:
  decode              steady-state decode tok/s at 1 user on a ~4000-token
                      prose prompt (max_tokens 256): 3 reps with different
                      variants, per-rep + median, compact table
  prefill             prefill tok/s at 20000 and 60000 tokens (configurable
                      via --sizes), cold each time via variant, max_tokens 8
  concurrency         N parallel long requests (default 3) whose prompts
                      target 256000 tokens (default --target-tokens 200000;
                      full 256k x N is a memory-liveness test, not a speed
                      test): liveness, per-request prefill estimate and
                      decode tok/s, aggregate throughput, slowest request
  decode-concurrency  N=3 parallel decode streams at 4k prompt / 256 tokens,
                      per-stream tok/s and total

Each measurement prints one machine-readable line
  RESULT <name> key=val key=val ...
plus a human-readable table, so a driver script can grep the RESULT lines.
Exit 0 on success, non-zero if any request errors. Stdlib only (urllib,
json, threading, time, argparse, sys).

Cold runs: prompts are built from prose FILLERS plus a variant stamp (same
trick as needle_probe.py) so every request carries fresh token blocks and
prefix caching cannot silently shorten a timing. --variant-base pins the
variant sequence; the default is clock-derived so each invocation is cold.
Layout: measure requests use base+i, prefill-calibration requests
base+1000+i, the --skip-first warmup base+2000.

Prefill/decode split: a decode measurement comes from one request whose wall
time contains both prefill and decode. A calibration request (max_tokens=1,
its own variant, same prompt size) measures prefill wall time; steady-state
decode rate is (output_tokens-1)/(total_time - prefill_estimate), reported
alongside raw total time and output token count.
"""
import argparse, json, sys, threading, time, urllib.request

CALIB_VARIANT_OFFSET = 1000
WARMUP_VARIANT_OFFSET = 2000
FULL_CONTEXT_TOKENS = 256000

# Realistic prose fillers. Variant picks / stamps the filler so every run is
# cold (see module docstring). Token estimate uses chars/3.6 like needle_probe.
FILLERS = [
    "The meadow stretches quietly beyond the old stone wall, and the wind "
    "moves through it without hurry. Grass leans in long waves that catch the "
    "afternoon light, then settles again as if nothing had passed. A narrow "
    "path crosses the field from the gate to the line of alders at the far "
    "edge, and each spring the path grows a little fainter under the "
    "returning green. Bees work the clover in patient circuits, indifferent "
    "to the heat and to the hours. From the hill above, the whole meadow "
    "looks like a cloth laid out for some quiet occasion, and the farmhouses "
    "beyond it seem to be waiting for a bell that never rings. Evening comes "
    "slowly here, and the shadows of the alders lengthen across the grass "
    "until they reach the wall and stop.",
    "Along the harbor the boats lean against their ropes while the tide turns "
    "under them, and the water slaps the stone quay with the sound of a slow "
    "clock. Fishermen mend nets on the warm steps, speaking little, working "
    "with the ease of long practice. Gulls argue over the remains of the "
    "morning catch and then rise together at some signal only they can hear. "
    "The warehouses along the front keep their tall doors open to the breeze, "
    "and the smell of tar and salt drifts out across the empty square. By "
    "late afternoon the light lies flat on the water, and every mast in the "
    "harbor casts a thin shadow toward the town.",
    "Snow gathers on the porch rails overnight, and by morning the garden "
    "path has vanished under a smooth white cover. The mill wheel freezes "
    "mid-turn, and the stream beneath it keeps working at the ice until it "
    "finds a way through. Smoke rises from the chimney in a straight line, "
    "then breaks apart against the low grey sky. Indoors the stove ticks as "
    "it warms, and the windows fog along their lower edges while the children "
    "map the frost with their fingers. Winter holds the valley in a long "
    "silence that no one disturbs, not even the crows that gather in the bare "
    "trees at noon and leave before dusk.",
    "The reading room keeps its own hours, quieter than the street outside "
    "and warmer than the empty corridors that lead to it. Lamps stand at each "
    "table with green glass shades, and the light they throw falls in circles "
    "on the open pages. Somewhere in the stacks a cart waits with books that "
    "have traveled from other rooms and other winters. The catalog drawers "
    "slide with a wooden sound, and the cards inside record a century of "
    "small arguments about where a thing belongs. A clock at the far end "
    "measures nothing important, and no one looks up when it strikes.",
    "The train leaves the platform at a walking pace and gathers speed only "
    "after the river, where the fields open on both sides and the telegraph "
    "poles begin their steady count. Passengers settle into the rhythm of the "
    "carriage, arranging coats and papers, watching the country revise itself "
    "outside the glass. A conductor moves down the aisle with the patience of "
    "someone who has made this trip in every kind of weather. Stations arrive "
    "and depart like punctuation, brief and unremarked. Toward evening the "
    "light in the window turns from silver to copper, and the hills ahead "
    "take on the color of old maps.",
]


def build_prose(target_tokens, variant):
    """~target_tokens of prose; variant selects/stamps fresh token blocks."""
    filler = FILLERS[variant % len(FILLERS)]
    if variant >= len(FILLERS):
        filler = f"[pass {variant}] " + filler
    approx_tok_per_rep = len(filler) / 3.6
    n_reps = max(1, int(target_tokens / approx_tok_per_rep))
    return filler * n_reps


def request_completion(port, model, prompt, max_tokens, timeout):
    body = json.dumps({"model": model, "prompt": prompt,
                       "max_tokens": max_tokens, "temperature": 0}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions", data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    wall = time.time() - t0
    text = d["choices"][0]["text"]
    usage = d.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens")
    if prompt_tokens is None:
        prompt_tokens = int(len(prompt) / 3.6)
    out_tokens = usage.get("completion_tokens")
    if out_tokens is None:
        out_tokens = max(1, int(len(text) / 3.6))
    return {"prompt_tokens": int(prompt_tokens), "output_tokens": int(out_tokens),
            "wall_s": wall, "text": text}


def run_parallel(port, model, prompts, max_tokens, timeout):
    """Launch len(prompts) requests on threads released by a barrier."""
    n = len(prompts)
    barrier = threading.Barrier(n)
    slots = [None] * n

    def worker(i):
        barrier.wait()
        t0 = time.time()
        try:
            r = request_completion(port, model, prompts[i], max_tokens, timeout)
            slots[i] = {"t0": t0, "t1": time.time(), "r": r, "err": None}
        except Exception as e:
            slots[i] = {"t0": t0, "t1": time.time(), "r": None, "err": e}

    threads = [threading.Thread(target=worker, args=(i,), daemon=True)
               for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return slots


def calib_prefill_rate(port, model, target_tokens, variant, timeout):
    """Serial max_tokens=1 request; returns (prefill_tok_per_s, record)."""
    r = request_completion(port, model, build_prose(target_tokens, variant),
                           1, timeout)
    pt = r["prompt_tokens"] or target_tokens
    return pt / max(r["wall_s"], 1e-9), r


def fmt(v):
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def emit_result(name, **kv):
    print("RESULT " + name + " " +
          " ".join(f"{k}={fmt(v)}" for k, v in kv.items()))


def print_table(headers, rows):
    if not rows:
        print("(no measurements)")
        return
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    fmt_row = "  ".join("{:>%d}" % w for w in widths)
    print(fmt_row.format(*headers))
    for row in rows:
        print(fmt_row.format(*[str(c) for c in row]))


def median(xs):
    ys = sorted(xs)
    n = len(ys)
    if n == 0:
        return 0.0
    mid = n // 2
    return ys[mid] if n % 2 else (ys[mid - 1] + ys[mid]) / 2.0


def variant_base(a):
    return a.variant_base if a.variant_base is not None else int(time.time()) % 1000000


def cmd_decode(a):
    vbase = variant_base(a)
    print(f"# decode model={a.model} port={a.port} prompt_tokens~{a.prompt_tokens} "
          f"max_tokens={a.max_tokens} temperature=0 reps={a.reps} variant_base={vbase}")
    if a.dry_run:
        print(f"DRY-RUN decode endpoint=/v1/completions model={a.model} port={a.port} network=off")
        for i in range(a.reps):
            for role, var, mt in (("measure", vbase + i, a.max_tokens),
                                  ("prefill-calib", vbase + CALIB_VARIANT_OFFSET + i, 1)):
                prompt = build_prose(a.prompt_tokens, var)
                print(f"DRY-RUN decode rep={i} role={role} variant={var} "
                      f"prompt_target_tokens={a.prompt_tokens} "
                      f"expected_prompt_tokens={int(len(prompt) / 3.6)} "
                      f"prompt_chars={len(prompt)} max_tokens={mt} "
                      f"expected_output_tokens={mt} temperature=0")
        print("DRY-RUN decode rate_formula=(output_tokens-1)/(total_s-prefill_est_s) "
              "prefill_est=calib:max_tokens=1")
        return 0
    rows, rates, totals, outs = [], [], [], []
    errors = 0
    for i in range(a.reps):
        var = vbase + i
        cvar = vbase + CALIB_VARIANT_OFFSET + i
        try:
            rate_s, c = calib_prefill_rate(a.port, a.model, a.prompt_tokens,
                                           cvar, a.timeout)
            prefill_est = c["wall_s"]
        except Exception as e:
            print(f"ERROR prefill-calib rep={i} variant={cvar}: {e}")
            errors += 1
            prefill_est = 0.0
        try:
            d = request_completion(a.port, a.model,
                                   build_prose(a.prompt_tokens, var),
                                   a.max_tokens, a.timeout)
        except Exception as e:
            print(f"ERROR decode rep={i} variant={var}: {e}")
            errors += 1
            continue
        denom = max(d["wall_s"] - prefill_est, 1e-9)
        rate = max(d["output_tokens"] - 1, 0) / denom
        rows.append([str(i), str(var), str(d["prompt_tokens"]),
                     str(d["output_tokens"]), f"{d['wall_s']:.3f}",
                     f"{prefill_est:.3f}", f"{rate:.3f}"])
        rates.append(rate)
        totals.append(d["wall_s"])
        outs.append(d["output_tokens"])
        emit_result("decode_rep", rep=i, variant=var,
                    prompt_tokens=d["prompt_tokens"],
                    output_tokens=d["output_tokens"], total_s=d["wall_s"],
                    prefill_est_s=prefill_est, decode_tok_s=rate)
    print()
    print_table(["rep", "variant", "prompt_tok", "out_tok", "total_s",
                 "prefill_est_s", "decode_tok_s"], rows)
    if rates:
        med_rate, med_tot, med_out = median(rates), median(totals), median(outs)
        print(f"median of {len(rates)} reps: decode_tok_s={med_rate:.3f} "
              f"total_s={med_tot:.3f} output_tokens={med_out:.0f}")
        emit_result("decode_median", reps=len(rates), decode_tok_s=med_rate,
                    total_s=med_tot, output_tokens=int(med_out))
    return 1 if errors else 0


def cmd_prefill(a):
    vbase = variant_base(a)
    sizes = list(a.sizes)
    print(f"# prefill model={a.model} port={a.port} sizes={sizes} "
          f"max_tokens={a.max_tokens} temperature=0 variant_base={vbase}")
    print("note: the first request after boot may include warmup/compile time; "
          "pass --skip-first to run and discard a warmup call")
    wvar = vbase + WARMUP_VARIANT_OFFSET
    if a.dry_run:
        print(f"DRY-RUN prefill endpoint=/v1/completions model={a.model} port={a.port} network=off")
        if a.skip_first:
            prompt = build_prose(sizes[0], wvar)
            print(f"DRY-RUN prefill role=warmup(discarded) size={sizes[0]} variant={wvar} "
                  f"prompt_target_tokens={sizes[0]} "
                  f"expected_prompt_tokens={int(len(prompt) / 3.6)} "
                  f"prompt_chars={len(prompt)} max_tokens={a.max_tokens} "
                  f"expected_output_tokens={a.max_tokens} temperature=0")
        for i, size in enumerate(sizes):
            var = vbase + i
            prompt = build_prose(size, var)
            print(f"DRY-RUN prefill role=measure size={size} variant={var} "
                  f"prompt_target_tokens={size} "
                  f"expected_prompt_tokens={int(len(prompt) / 3.6)} "
                  f"prompt_chars={len(prompt)} max_tokens={a.max_tokens} "
                  f"expected_output_tokens={a.max_tokens} temperature=0")
        return 0
    errors = 0
    if a.skip_first:
        try:
            w = request_completion(a.port, a.model,
                                   build_prose(sizes[0], wvar),
                                   a.max_tokens, a.timeout)
            print(f"warmup discarded: size={sizes[0]} variant={wvar} "
                  f"wall_s={w['wall_s']:.3f} (not reported in results)")
        except Exception as e:
            print(f"ERROR warmup variant={wvar}: {e}")
            errors += 1
    rows = []
    for i, size in enumerate(sizes):
        var = vbase + i
        try:
            r = request_completion(a.port, a.model, build_prose(size, var),
                                   a.max_tokens, a.timeout)
        except Exception as e:
            print(f"ERROR prefill size={size} variant={var}: {e}")
            errors += 1
            continue
        pt = r["prompt_tokens"] or int(size)
        tok_s = pt / max(r["wall_s"], 1e-9)
        rows.append([str(size), str(var), str(pt), f"{r['wall_s']:.3f}",
                     f"{tok_s:.3f}", str(r["output_tokens"])])
        emit_result("prefill", size=size, variant=var, prompt_tokens=pt,
                    wall_s=r["wall_s"], prompt_tok_s=tok_s,
                    output_tokens=r["output_tokens"])
    print()
    print_table(["size", "variant", "prompt_tok", "wall_s", "prompt_tok_s",
                 "out_tok"], rows)
    print(f"note: wall_s includes the short max_tokens={a.max_tokens} decode; "
          "prompt_tok_s = prompt_tokens / wall_s")
    return 1 if errors else 0


def _calib_or_none(a, target, cvar):
    try:
        rate, r = calib_prefill_rate(a.port, a.model, target, cvar, a.timeout)
        print(f"calib: prompt_tokens={r['prompt_tokens']} wall_s={r['wall_s']:.3f} "
              f"prefill_rate={rate:.3f} tok/s (serial, max_tokens=1, variant={cvar})")
        return rate
    except Exception as e:
        print(f"ERROR prefill-calib variant={cvar}: {e}")
        return None


def cmd_concurrency(a):
    vbase = variant_base(a)
    n, target = a.n, a.target_tokens
    print(f"# concurrency model={a.model} port={a.port} n={n} "
          f"target_tokens={target} max_tokens={a.max_tokens} "
          f"temperature=0 variant_base={vbase}")
    print(f"WARNING: full {FULL_CONTEXT_TOKENS} x {n} is a memory-liveness "
          f"test, not a speed test; this run uses --target-tokens {target}.")
    cvar = vbase + CALIB_VARIANT_OFFSET
    if a.dry_run:
        print(f"DRY-RUN concurrency endpoint=/v1/completions model={a.model} port={a.port} network=off")
        print(f"DRY-RUN concurrency role=prefill-calib variant={cvar} "
              f"prompt_target_tokens={target} max_tokens=1 "
              f"expected_output_tokens=1 temperature=0")
        for i in range(n):
            var = vbase + i
            prompt = build_prose(target, var)
            print(f"DRY-RUN concurrency req={i} role=measure variant={var} "
                  f"prompt_target_tokens={target} "
                  f"expected_prompt_tokens={int(len(prompt) / 3.6)} "
                  f"prompt_chars={len(prompt)} max_tokens={a.max_tokens} "
                  f"expected_output_tokens={a.max_tokens} temperature=0")
        print(f"DRY-RUN concurrency start=barrier(threads={n}) "
              "liveness=all_requests_finished")
        return 0
    errors = 0
    calib_rate = _calib_or_none(a, target, cvar)
    if calib_rate is None:
        errors += 1
    prompts = [build_prose(target, vbase + i) for i in range(n)]
    slots = run_parallel(a.port, a.model, prompts, a.max_tokens, a.timeout)
    rows, slow = [], []
    sum_prompt = sum_out = 0
    for i, s in enumerate(slots):
        if s is None:
            continue
        if s["err"] is not None:
            print(f"ERROR concurrency req={i} variant={vbase + i}: {s['err']}")
            errors += 1
            emit_result("concurrency_req", req=i, variant=vbase + i, status="error")
            continue
        r = s["r"]
        total = s["t1"] - s["t0"]
        pest = (r["prompt_tokens"] or target) / calib_rate if calib_rate else 0.0
        rate = max(r["output_tokens"] - 1, 0) / max(total - pest, 1e-9)
        sum_prompt += r["prompt_tokens"]
        sum_out += r["output_tokens"]
        slow.append((total, i, rate))
        rows.append([str(i), str(vbase + i), str(r["prompt_tokens"]),
                     str(r["output_tokens"]), f"{total:.3f}", f"{pest:.3f}",
                     f"{rate:.3f}"])
        emit_result("concurrency_req", req=i, variant=vbase + i, status="ok",
                    prompt_tokens=r["prompt_tokens"],
                    output_tokens=r["output_tokens"], total_s=total,
                    prefill_est_s=pest, decode_tok_s=rate)
    finished = sum(1 for s in slots if s and s["err"] is None and s["r"] is not None)
    liveness = "PASS" if finished == n else "FAIL"
    ok = [s for s in slots if s and s["r"] is not None]
    burst_wall = (max(s["t1"] for s in ok) - min(s["t0"] for s in ok)) if ok else 0.0
    print()
    print_table(["req", "variant", "prompt_tok", "out_tok", "total_s",
                 "prefill_est_s", "decode_tok_s"], rows)
    print(f"liveness: {liveness} ({finished}/{n} requests finished)")
    if slow:
        slow.sort()
        t, i, rate = slow[-1]
        print(f"slowest req={i} total_s={t:.3f} decode_tok_s={rate:.3f}")
        emit_result("concurrency_slowest", req=i, total_s=t, decode_tok_s=rate)
        agg_out = sum_out / max(burst_wall, 1e-9)
        agg_prompt = sum_prompt / max(burst_wall, 1e-9)
        agg_all = (sum_prompt + sum_out) / max(burst_wall, 1e-9)
        print(f"aggregate over burst wall {burst_wall:.3f}s: agg_tok_s={agg_all:.3f} "
              f"(prompt {sum_prompt} + output {sum_out} tokens)")
        emit_result("concurrency_agg", n=n, finished=finished, liveness=liveness,
                    target_tokens=target, wall_s=burst_wall,
                    agg_tok_s=agg_all, agg_prompt_tok_s=agg_prompt,
                    agg_out_tok_s=agg_out)
    else:
        emit_result("concurrency_agg", n=n, finished=finished, liveness=liveness,
                    target_tokens=target, wall_s=burst_wall)
    return 1 if errors else 0


def cmd_decode_concurrency(a):
    vbase = variant_base(a)
    n = a.n
    print(f"# decode-concurrency model={a.model} port={a.port} n={n} "
          f"prompt_tokens~{a.prompt_tokens} max_tokens={a.max_tokens} "
          f"temperature=0 variant_base={vbase}")
    cvar = vbase + CALIB_VARIANT_OFFSET
    if a.dry_run:
        print(f"DRY-RUN decode-concurrency endpoint=/v1/completions model={a.model} port={a.port} network=off")
        print(f"DRY-RUN decode-concurrency role=prefill-calib variant={cvar} "
              f"prompt_target_tokens={a.prompt_tokens} max_tokens=1 "
              f"expected_output_tokens=1 temperature=0")
        for i in range(n):
            var = vbase + i
            prompt = build_prose(a.prompt_tokens, var)
            print(f"DRY-RUN decode-concurrency stream={i} role=measure variant={var} "
                  f"prompt_target_tokens={a.prompt_tokens} "
                  f"expected_prompt_tokens={int(len(prompt) / 3.6)} "
                  f"prompt_chars={len(prompt)} max_tokens={a.max_tokens} "
                  f"expected_output_tokens={a.max_tokens} temperature=0")
        print(f"DRY-RUN decode-concurrency start=barrier(threads={n})")
        return 0
    errors = 0
    calib_rate = _calib_or_none(a, a.prompt_tokens, cvar)
    if calib_rate is None:
        errors += 1
    prompts = [build_prose(a.prompt_tokens, vbase + i) for i in range(n)]
    slots = run_parallel(a.port, a.model, prompts, a.max_tokens, a.timeout)
    rows, rates = [], []
    sum_out = 0
    for i, s in enumerate(slots):
        if s is None:
            continue
        if s["err"] is not None:
            print(f"ERROR decode-concurrency stream={i} variant={vbase + i}: {s['err']}")
            errors += 1
            emit_result("decode_conc_stream", stream=i, variant=vbase + i,
                        status="error")
            continue
        r = s["r"]
        total = s["t1"] - s["t0"]
        pest = (r["prompt_tokens"] or a.prompt_tokens) / calib_rate if calib_rate else 0.0
        rate = max(r["output_tokens"] - 1, 0) / max(total - pest, 1e-9)
        raw = max(r["output_tokens"] - 1, 0) / max(total, 1e-9)
        sum_out += r["output_tokens"]
        rates.append(rate)
        rows.append([str(i), str(vbase + i), str(r["prompt_tokens"]),
                     str(r["output_tokens"]), f"{total:.3f}", f"{pest:.3f}",
                     f"{rate:.3f}", f"{raw:.3f}"])
        emit_result("decode_conc_stream", stream=i, variant=vbase + i,
                    status="ok", prompt_tokens=r["prompt_tokens"],
                    output_tokens=r["output_tokens"], total_s=total,
                    prefill_est_s=pest, decode_tok_s=rate, raw_tok_s=raw)
    finished = sum(1 for s in slots if s and s["err"] is None and s["r"] is not None)
    ok = [s for s in slots if s and s["r"] is not None]
    burst_wall = (max(s["t1"] for s in ok) - min(s["t0"] for s in ok)) if ok else 0.0
    liveness = "PASS" if finished == n else "FAIL"
    print()
    print_table(["stream", "variant", "prompt_tok", "out_tok", "total_s",
                 "prefill_est_s", "decode_tok_s", "raw_tok_s"], rows)
    sum_rate = sum(rates)
    agg_out = sum_out / max(burst_wall, 1e-9)
    print(f"total: streams={finished}/{n} liveness={liveness} "
          f"sum_stream_tok_s={sum_rate:.3f} agg_out_tok_s={agg_out:.3f} "
          f"burst_wall_s={burst_wall:.3f}")
    emit_result("decode_conc_total", n=n, finished=finished, liveness=liveness,
                wall_s=burst_wall, sum_stream_tok_s=sum_rate,
                agg_out_tok_s=agg_out)
    return 1 if errors else 0


def add_common(p):
    p.add_argument("--port", default="9700",
                   help="server port (default 9700)")
    p.add_argument("--model", default="mimo-v2.6-flash",
                   help="model name (default mimo-v2.6-flash)")
    p.add_argument("--dry-run", action="store_true",
                   help="print the requests that would be sent; no network")
    p.add_argument("--timeout", type=float, default=5400.0,
                   help="per-request timeout in seconds (default 5400)")
    p.add_argument("--variant-base", type=int, default=None,
                   help="first variant id (fresh prompt text = cold run); "
                        "default: clock-derived so runs are cold")


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="mimo_perf_bench.py",
        description="Acceptance performance bench for mimo-v2.6-flash "
                    "(decode / prefill / concurrency / decode-concurrency).")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("decode", help="decode tok/s at 1 user, 3 reps + median")
    add_common(d)
    d.add_argument("--reps", type=int, default=3)
    d.add_argument("--prompt-tokens", type=int, default=4000)
    d.add_argument("--max-tokens", type=int, default=256)
    d.set_defaults(func=cmd_decode)

    pf = sub.add_parser("prefill", help="prefill tok/s at target prompt sizes")
    add_common(pf)
    pf.add_argument("--sizes", type=int, nargs="+", default=[20000, 60000],
                    help="prompt target sizes (default 20000 60000)")
    pf.add_argument("--max-tokens", type=int, default=8)
    pf.add_argument("--skip-first", action="store_true",
                    help="run and discard one warmup call first")
    pf.set_defaults(func=cmd_prefill)

    cc = sub.add_parser("concurrency",
                        help="N parallel long requests: liveness + throughput")
    add_common(cc)
    cc.add_argument("--n", type=int, default=3)
    cc.add_argument("--target-tokens", type=int, default=200000,
                    help="prompt target tokens per request (default 200000; "
                         "256000 x N is a memory-liveness test, not a speed test)")
    cc.add_argument("--max-tokens", type=int, default=256)
    cc.set_defaults(func=cmd_concurrency)

    dc = sub.add_parser("decode-concurrency",
                        help="N parallel decode streams at 4k prompt")
    add_common(dc)
    dc.add_argument("--n", type=int, default=3)
    dc.add_argument("--prompt-tokens", type=int, default=4000)
    dc.add_argument("--max-tokens", type=int, default=256)
    dc.set_defaults(func=cmd_decode_concurrency)

    a = p.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
