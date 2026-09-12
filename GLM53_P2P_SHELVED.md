# PCIe P2P on the 8× MI50 (gfx906) rig — investigation record (SHELVED)

Status 2026-09-12: shelved — requires BIOS access (user has none atm).

## Measured facts
- `rocm-smi --showtopoaccess`: P2P access = False for ALL GPU pairs. KFD has zero
  GPU↔GPU links (so RCCL falls back to the SHM ring transport).
- PCI topology: 8 MI50s behind PLX 8747 Gen3 switches (17 bridges observed),
  dual Xeon, iommu=pt in kernel cmdline; large BARs map fine.
- Kernel has all the right bits: CONFIG_PCI_P2PDMA=y, CONFIG_HSA_AMD=y,
  CONFIG_HSA_AMD_P2P=y, CONFIG_HSA_AMD_SVM=y (6.8.0-139-generic).
- RCCL consequence: ~92 small allreduces per decode step ≈ 42 ms of an 86 ms
  step (MIB ~49%). Measured 456 µs/call ≈ 14 SHM ring hops × 32 µs.

## What was tried and failed (all parity, measured)
- `HSA_FORCE_FINE_GRAIN_PCIE=1` (would force P2P over PCIe): 11.57→11.58 tok/s.
- `NCCL_ALGO=Tree` + `NCCL_MIN_CTAS=1/MAX=2/NTHREADS=128`: 11.54–11.60 tok/s.
- Conclusion: fnv1a it. NCCL knob space exhausted at SHM-ring transport.

## The blocker (PCIe ACS on PLX downstream ports)
ACS (Access Control Services) on the PLX switch ports routes GPU-destined TLPs
to the root complex = P2P is architecturally impossible until ACS is relaxed.
The 4029GP-TRT's GT/s-level serving: no ACS override in Ubuntu's stock kernel
(acs_override lives in special patch sets, not mainline); a rebuild+reboot with
it is tracked as a possible path here (do not bother on this machine unless
BIOS can toggle).

## Options ranked
1. **BIOS** route (preferred, needs console/BMC): hunt Advanced ▸ Chipset ▸
   North Bridge/IIO slots for "ACS", "ACS Control", "PLX ACS", or "PCIe P2P"
   toggles; disable ACS.
2. **Custom kernel** `pci=acs_override=downstream,multifunction` — messy because
   mainline lacks it; kills APTERG for this box; parked.
3. **xGMI bridges** (MI50 era): different link entirely — separate doc
   (see agent-30 xgmi findings; MI50 xGMI taps are physical edge-fingers,
   connectors scarce; also untracked).

## If it ever lights up (verify before trusting)
rocm-smi --showtopoaccess flips True for at least intra-switch pairs →
rccl bench (bench_rccl_smallmsg.py is done for that) BEFORE letting any vLLM
upgrade hit it (RCCL has (had) a MI50-on-P2P crash bug on 6.3.3/7.x builds —
we're on 6.4.3: bench anyway). Expected: NCCL 42ms chunk → ~12–18 ms → GLM
~19–20 t/s, Qwen likewise.

## References in repo
- glm53-gfx906 branch: GLM53_RCCL_PLAN.md (agent-25), EXPERIMENTS.md bench numbers.
- RIG topology: /proc/bus/pci reads + rocm-smi —run after any BIOS change.
