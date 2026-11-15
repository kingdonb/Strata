"""Qwen3.6-35B-A3B on a Mac: the sizes this fork's Metal engine path runs (docs/QWEN36_METAL.md).

ONE table, read by metal/setup_mac_q36.py (the setup) and tools/strata_mcp.py (the model list the app and an
assistant see), so both name the same sizes with the same numbers.

Why the sizes are here and not in setup.py's OWN_PATH_FAMILIES["qwen36"]: that table is the family as upstream's CUDA /
HIP engine path runs it (UD-Q4_K_XL, UD-Q6_K_XL, Q8_0).  A size added there is offered, and for a PC with little RAM
recommended, on every platform.  UD-Q2_K_XL was run on the Metal engine only, so it is a Mac size and nothing else.

Standard library only; importing this file reads nothing and starts nothing.
"""
from __future__ import annotations

FAMILY = "qwen36"

# size -> what setup and the model list say about it on a Mac.
#   file / size / sha256: the GGUF in the family's repository (setup.py OWN_PATH_FAMILIES["qwen36"]["hf"], at its pinned
#     revision).  The SHA-256 is the repository's own (the file's LFS id); the copy on the M2 this was run on has it.
#   ram_gb: unified memory the size was run in (an M2 with 24 GB).  arena_gb: the experts' share of the file.
MAC_SIZES = {
    "UD-Q2_K_XL": {"about": "2-bit (Unsloth Dynamic), the smallest: the size run on a 24 GB Mac",
                   "file": "Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf", "size": 12574128416,
                   "sha256": "ed7cda7e38985b4fcff76475865135039641d2bfbac3c169df15ca770f37fb0c",
                   "download_gb": 12.6, "ram_gb": 16, "arena_gb": 11.7},
}
DEFAULT = "UD-Q2_K_XL"

# The names earlier setups of this fork gave the run config of a size (strata-<id>.json): a model set up then is
# still shown as installed, under the id its config has.
EARLIER_IDS = {"UD-Q2_K_XL": ("q36_q2", "q36")}
