# Runtime Analysis (`sample_15.jsonl`)

## Dataset Snapshot
- Source: `datasets/RTL-Coder_small/sample_15.jsonl`
- Size: 15 instructions (one JSON record per line), each already accompanied by a reference `output`.

## Latest Run (see `run.log`)
- Command inputs: `VERIREASON_INPUT_FILE=.../sample_15.jsonl`, `VERIREASON_OUTPUT_DIR=.../runs/rtl_small`
- Run summary (`run.log`:123-135):
  - Processed: 15/15 entries
  - Successful simulations: 10
  - Deterministic implementations: 0

## `processed_entries.jsonl`
- Current total lines: 39 (file contains results from multiple past runs and keeps appending).
- Unique entries accumulated over time: 26 (`entry_1`, `entry_2`, ..., `entry_28` excluding a few numbers).
- The last 15 JSON objects correspond to the most recent run (IDs in order: `entry_2`, `entry_3`, `entry_4`, `entry_5`, `entry_1`, `entry_1`, `entry_2`, `entry_4`, `entry_5`, `entry_7`, `entry_8`, `entry_10`, `entry_13`, `entry_14`, `entry_15`).
  - Among these, 14 have a filled `tb_result` (successful TB generation + simulation).
  - `entry_1` appears twice; the first attempt failed to produce a testbench (null `tb`, `tb_result`), while the second succeeded.

## Examples
- Successful case: `runs/rtl_small/processed_entries.jsonl:1` (`entry_4`, module `twos_complement`) contains reasoning, Verilog implementation, a valid testbench, and 100 test vectors.
- Failed TB generation example: `runs/rtl_small/processed_entries.jsonl:22` (`entry_1` first attempt) has code but `tb`/`tb_result` are null after repeated retries.

## Tips for Further Runs
1. To keep statistics clear per run, delete `runs/rtl_small/processed_entries.jsonl` before launching a new experiment, or snapshot it elsewhere.
2. Alternatively, analyze only the tail section (e.g., `tail -n 15 processed_entries.jsonl`) to focus on the most recent batch when the file already contains historical data.
