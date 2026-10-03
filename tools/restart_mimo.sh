#!/bin/bash
# Graceful MiMo server restart: SIGTERM by exact PID -> drain /dev/kfd -> boot
# with retry (early-boot segfault is flaky ~50% on this rig).
set -u
LOG=/tmp/mimo_boot_$(date +%H%M%S).log

PAT=$(printf 'bin/%s serve' 'vllm')
PID=$(pgrep -f "$PAT" | head -1)
if [ -n "${PID:-}" ]; then
  # WEDGE LESSON (2026-10-03, two power resets): SIGTERM while requests are
  # mid-flight (open profile window, or a 256k prefill) leaves the process
  # in D-state with VRAM pinned and 1000+ kfd_process_wq threads stuck.
  # Drain the queue first; only then signal.
  if curl -s --max-time 3 http://127.0.0.1:9700/metrics >/dev/null 2>&1; then
    echo "draining request queue before SIGTERM (max 120s)"
    for i in $(seq 1 24); do
      R=$(curl -s --max-time 3 http://127.0.0.1:9700/metrics 2>/dev/null | \
          grep -E "^vllm:num_requests_(running|waiting)" | \
          awk '{s+=$2} END {print s+0}')
      [ "$R" = "0" ] && { echo "queue empty"; break; }
      [ "$i" = "24" ] && echo "queue still busy after 120s - proceeding anyway"
      sleep 5
    done
  fi
  echo "SIGTERM pid $PID"
  kill "$PID"
  for i in $(seq 1 60); do
    kill -0 "$PID" 2>/dev/null || break
    sleep 1
  done
  if kill -0 "$PID" 2>/dev/null; then
    ST=$(grep "^State" "/proc/$PID/status" 2>/dev/null | awk '{print $2}')
    echo "STILL ALIVE after 60s (state=$ST) - refusing to SIGKILL (driver risk)."
    if [ "$ST" = "D" ]; then
      echo "WEDGE: D-state teardown; if VRAM stays pinned after ~3 min of"
      echo "kfd drain: sudo ipmitool chassis power reset"
    fi
    exit 1
  fi
  echo "server exited"
else
  echo "no server running"
fi

# ROOT CAUSE c8c152ef1e: orphaned VLLM::Worker_TP keep 25.7GB pinned per
# GPU after parent death -> next boot dies at WorkerProc.init_device. Sweep
# them by explicit PID before booting.
sleep 3
for _try in 1 2 3 4 5; do
    Z=$(ps -eo pid,comm | awk '$2 ~ /^VLLM::/ {print $1}')
    [ -z "$Z" ] && break
    echo "zombie KFD workers: $Z -- SIGTERM"
    for p in $Z; do kill "$p" 2>/dev/null; done
    sleep 4
done
# workers sometimes linger a beat after the parent exits
sleep 3
WRK=$(pgrep -c '^VLLM::' 2>/dev/null || true)
echo "lingering VLLM:: workers: ${WRK:-0}"

if [ -x /data/vllm-gfx906-dsv4/tools/gpu_drain_wait.sh ]; then
  echo "--- gpu drain ---"
  bash /data/vllm-gfx906-dsv4/tools/gpu_drain_wait.sh || echo "drain script returned nonzero (continuing)"
fi

for attempt in 1 2 3 4; do
  echo "=== boot attempt $attempt ==="
  T0=$(date +%s)
  setsid nohup bash /data/vllm-gfx906-dsv4/run_mimo_v2_6_omni.sh 9700 > "$LOG" 2>&1 < /dev/null &
  ok=0
  # port-ownership poll: never trust setsid $! (it exits immediately; the
  # old BGPID check false-positived and double-booted 2026-10-03).
  # READY = /v1/models 200 AND :9700 owned by a pid started after T0.
  for i in $(seq 1 300); do   # up to 10 min to first ready (cold cache)
    sleep 2
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:9700/v1/models 2>/dev/null || echo 000)
    if [ "$code" = "200" ]; then
      SPID=$(ss -tlnp 2>/dev/null | awk '/:9700 /{print}' | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)
      SSTART=$(stat -c %Y "/proc/${SPID:-0}" 2>/dev/null || echo 0)
      if [ -n "${SPID:-}" ] && [ "$SSTART" -ge "$T0" ]; then
        echo "READY on attempt $attempt after ~$((i*2))s (server pid=$SPID)"
        ok=1
        break
      fi
      echo "port 9700 answers but owned by stale pid ${SPID:-?} (predates T0) - waiting"
    fi
    if [ "$i" -ge 40 ] && ! pgrep -f "$PAT" >/dev/null 2>&1; then
      echo "boot process gone with no server after $((i*2))s (see $LOG tail below)"
      tail -15 "$LOG"
      break
    fi
  done
  if [ "$ok" = "1" ]; then
    echo "BOOT OK log=$LOG"
    exit 0
  fi
  echo "attempt $attempt failed; cleaning up before retry"
  P2=$(pgrep -f "$PAT" | head -1)
  [ -n "${P2:-}" ] && kill "$P2" 2>/dev/null
  for j in $(seq 1 30); do kill -0 "${P2:-0}" 2>/dev/null || break; sleep 1; done
  sleep 5
done

echo "ALL BOOT ATTEMPTS FAILED — machine may need attention"
exit 1
