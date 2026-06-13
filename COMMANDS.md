# moe-epmi Command Log

All commands run in or for this project, in order.

---

## 2026-06-12

### Create project folder
```bash
mkdir /u/yxu30/moe-epmi
```

### Write SLURM batch script
Created `/u/yxu30/moe-epmi/run_olmoe_test.sh`
- Requests 1x A100 GPU, 1 hour, on `gpuA100x4` partition under account `bexq-delta-gpu`
- Uses conda env at `/projects/bexq/yxu30/conda/envs/olmoe` (torch 2.9.1, transformers 4.57.3)
- Caches HuggingFace models to `/work/hdd/bexq/yxu30/hf/hub`
- Runs `/u/yxu30/olmoe_expert_probe.py` against `allenai/OLMoE-1B-7B-0924`

### Submit job
```bash
sbatch /u/yxu30/moe-epmi/run_olmoe_test.sh
# Job ID: 19140518
# Output: /u/yxu30/moe-epmi/olmoe_test_19140518.out
# Error:  /u/yxu30/moe-epmi/olmoe_test_19140518.err
```

### Monitor job
```bash
squeue -u yxu30
tail -f /u/yxu30/moe-epmi/olmoe_test_19140518.out
```

### Write EPMI diagnostic pipeline
Created five files in `/u/yxu30/moe-epmi/`:
- `config.py` — EPMIConfig dataclass (model, datasets, metric knobs)
- `data_loader.py` — load_texts() with streaming/non-streaming support
- `routing_extractor.py` — extract_routing_stats() via output_router_logits=True
- `metrics.py` — compute_red(), compute_ead(), compute_epmi(); RLC stub
- `run_diagnostics.py` — orchestration, direction check, stability check
- `run_epmi.sh` — SLURM job script for A100 node

### Submit EPMI diagnostic job
```bash
sbatch run_epmi.sh
# Monitor: tail -f epmi_<jobid>.out
```
