# Crash forensics (2026-10-03 night session) — three distinct failure modes

## Boot map (journalctl --list-boots)
-5  06:53 -> 13:44   power reset #1 (teardown wedge)
-4  13:45 -> 15:21   power reset #2 (teardown wedge)
-3  15:22 -> 17:06   power reset #3 (teardown wedge, 1031 D-state kfd threads)
-2  17:07 -> 19:18   **MACHINE FREEZE** (the user-visible crash)
-1  19:19 -> 20:01   agent-20 deliberate power reset (H1 test)
 0  20:02 -> now    H1 prefix-cache A/B boot

## Mode 1: whole-machine freeze (boot -2, 19:18:11)
Journal just STOPS mid-stream: no oops, no panic, no shutdown, no BMC/SEL
event (SEL is stale-full since 07/26), no EDAC/MCE memory errors (EDAC
skx_edac is loaded and reporting clean).

LAST KERNEL LINE (19:16:33, 2 min before freeze):
    workqueue: svm_range_restore_work [amdgpu] hogged CPU for >10000us 64 times,
    consider switching to WQ_UNBOUND

Same family seen on the clean 19:53 teardown:
    workqueue: kfd_process_wq_release [amdgpu] hogged CPU for >10000us

MECHANISM (evidence-based): amdgpu SVM/KFD memory-management workqueues
(svm_range_restore_work on range restore, kfd_process_wq_release on process
teardown) go pathological under large pinned-VRAM churn (33 GB resident,
multi-GB KV), hog CPUs, and the box hard-freezes. Matches all 4 events:
each happened at process teardown or heavy KV churn with pinned VRAM.

NOTE: kernel is tainted by an unsigned out-of-tree amdgpu KCL module
("amdkcl: module verification failed ... tainting kernel") — custom driver
stack. /opt holds rocm-6.2.4, 6.4.2.disabled, 6.4.3 side by side.

MITIGATION CANDIDATES: WQ_UNBOUND for those workqueues (the kernel itself
suggests it); keep the 3.5 GiB KV pool (not 5.6) to leave >=2 GB free;
never SIGTERM under VRAM pressure (power-cycle instead, rule learned 3x);
consider ROCm version alignment (3 installs present).

## Mode 2: silent engine-parent death at post-load (the "flaky boot segfault")
NOT a segfault: zero segfault lines in any dmesg. Signature: weights load
fine (486s), then WorkerProc "BrokenPipeError" — engine parent already dead
at the worker-ready handshake. The post-load transition is where the
GLM53-MIMO-AUDIT checksum block ran (24 tensors x .float() temps x 8 workers
= multi-GB spike at exactly that moment). Gated behind VLLM_MIMO_AUDIT=1 in
commit b0f29e91fc.
NEGATIVE RESULT (2026-10-03 21:32 boot, audit OFF): the early boot death
PERSISTS with the audit code gated off -- died before weight loading
(0 safetensors lines, same _start/Py_BytesMain stack). The audit-spam
theory is weakened: it was a stressor at the post-load transition but is
NOT the sole trigger. The ~50% early-boot death has another cause (suspects:
TP8 worker spawn race, ROCm init under the KCL-tainted driver, or the
custom compilation-config path). Retry loop remains the mitigation.

## Mode 3: "crashes" that were not crashes
19:53 event = graceful SIGTERM + restart by an orphaned scheduled chain from
a previously-killed agent session (remote sleep;cmd chains survive agent
death). Swept clean afterwards.

## Evidence-handling lesson
/tmp boot logs are wiped by machine reboots (tmpfiles) — the freeze boot's
stdout log was lost that way. Kernel journal (/var/log/journal) survives.
Future boot logs should go to /data/tmp.

## SUSPECT CONFIG: expandable_segments <-> svm_range_restore_work (2026-10-03)
The launcher env sets PYTORCH_ALLOC_CONF="expandable_segments:True". ROCm's
expandable segments are SVM-backed (shared virtual memory ranges), and
svm_range_restore_work is the amdgpu workqueue that restores those ranges on
eviction/resume. Our config may be feeding the exact CPU-hog storm seen
before the 19:18 freeze (64x >10ms in 2 min) and during teardowns.
MITIGATION TO A/B at next convenient restart: drop expandable_segments
(plain caching allocator) -> less SVM range churn -> fewer restore storms.
Also candidate: the silent post-load parent death (Mode 2) coincides with
allocator pressure at the load->ready transition; the plain allocator
changes that pressure profile too. One env line, cheap to A/B, both failure
modes benefit. Not applied yet (avoid launcher edits while quality agent
owns restarts).

### Mode 2 sub-pattern (2026-10-03 21:32 boot attempts 1-3)
attempt 1: died BEFORE weight loading (0 shards) -- flaky early death.
attempt 2: loaded at 7.5 s/shard (2.5x slower than normal ~3 s) and died
at the post-load transition. attempt 3: loading at 3.3 s/shard, healthy.
CORRELATION: the abnormally-slow-loading boot is the one that dies. Slow
weight streaming may share a root cause with the death (host-RAM pressure,
PCIe retry storms not visible in lspci link state, or KFD context setup
degrading under whatever makes loads slow). Candidate check: compare
dmesg/xid during a slow load vs a fast one; the slowness is observable
in real time as an early warning that the boot will die -- the retry loop
could watch shard-rate and abort+retry early instead of waiting for death.
