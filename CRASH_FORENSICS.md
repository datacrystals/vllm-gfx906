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
commit b0f29e91fc. Boot-death rate after gating: TBD (watch next boots).

## Mode 3: "crashes" that were not crashes
19:53 event = graceful SIGTERM + restart by an orphaned scheduled chain from
a previously-killed agent session (remote sleep;cmd chains survive agent
death). Swept clean afterwards.

## Evidence-handling lesson
/tmp boot logs are wiped by machine reboots (tmpfiles) — the freeze boot's
stdout log was lost that way. Kernel journal (/var/log/journal) survives.
Future boot logs should go to /data/tmp.
