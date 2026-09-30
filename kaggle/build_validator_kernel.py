"""Faithful bench: run the public 0.947 notebook's own VALIDATOR on chosen training crops and dump the RAW
graphs (post-ILP, pre-repair) of its real detection (two seeds + TTA + threshold 0.965 + bidirectional fusion).

With those graphs, `filter_output_graph` (locally, the notebook's own module) reproduces the deployed graph
repair with any set of constants, without a GPU, so the pipeline's constants can be measured on the real chain.

Injections into the public notebook (passed with --base-notebook; third-party, not redistributed):
  code cell 0:  BIOHUB_VALIDATOR_ENABLE "0" -> "1" (the fork disables the validator)
  code cell 1:  BIOHUB_HOCT_VETO forced to "0" (the HOCT veto is not needed for raw graphs)
  code cell 3:  optional BIOHUB_* overrides (--env) before the constants are read
  code cell 6:  log line before write_test_submission("base") (the test submission is kept: a guard
                cell requires /kaggle/working/submission.csv)
  code cell 9:  after `print(val_stems)`: val_stems = VAL19 (19 training crops, organizer's seed-0 split)
                or the crops given with --dids
  code cell 11: after "VALIDATOR: cached ...": dump /kaggle/working/raw/<stem>.json
                {"nodes": [[id, t, z, y, x], ...], "edges": [[source, target, prob], ...]}
                (key names kept for compatibility with bench/faithful_bench.py)

Usage:
  python kaggle/build_validator_kernel.py --base-notebook <public.ipynb> --base-metadata <kernel-metadata.json> \
      --owner <your-kaggle-username> [--dids a,b,c] [--env BIOHUB_DET_THRESHOLD=0.95]
"""
from __future__ import annotations

import argparse
import io
import json
import os

VAL19 = ['44b6_1574802b', '44b6_706092f0', '44b6_d5e7d891', '44b6_d754aa59', '44b6_e57ff5c6', '6bba_2312ac41',
         '6bba_268e1230', '6bba_283bf9f1', '6bba_3a1849c2', '6bba_3abfe10a', '6bba_5c824876', '6bba_61dd1e0d',
         '6bba_7af54fde', '6bba_7b5d3b2c', '6bba_aeee7805', '6bba_afb141ff', '6bba_c27cba08', '6bba_c328f2fd', '6bba_d1acb6ff']

INJ9 = '''
# ==== FAITHFUL BENCH: run the validator on the chosen crops ====
val_stems = [s for s in %(val19)r if (TRAIN_DIR / (s + ".zarr")).exists()]
print("VALIDATOR: forced crop list:", len(val_stems), "crops", flush=True)
# ==== end ====
'''

INJ11 = '''
# ==== FAITHFUL BENCH: dump raw graphs (post-ILP, pre-repair) ====
import json as _bf_json
import os as _bf_os
_bf_dir = "/kaggle/working/raw"
_bf_os.makedirs(_bf_dir, exist_ok=True)
for _bf_stem, (_bf_nodes, _bf_edges) in VAL_RAW_GRAPHS.items():
    _bf_out = {
        "nodes": [[int(k), int(v["t"]), float(v["z"]), float(v["y"]), float(v["x"])] for k, v in sorted(_bf_nodes.items())],
        "edges": [[int(e["source_id"]), int(e["target_id"]), (None if e.get("edge_prob") is None else float(e["edge_prob"]))]
                    for e in _bf_edges],
    }
    with open(_bf_os.path.join(_bf_dir, _bf_stem + ".json"), "w") as _bf_f:
        _bf_json.dump(_bf_out, _bf_f)
    print("FAITHFUL BENCH:", _bf_stem, len(_bf_out["nodes"]), "nodes", len(_bf_out["edges"]), "raw edges", flush=True)
# ==== end ====
'''


def _indent(block):
    return "".join("    " + l + "\n" if l.strip() else "\n" for l in block.strip("\n").splitlines())


def main():
    ap = argparse.ArgumentParser(description="Build a kernel that runs the public notebook's validator and dumps raw graphs.")
    ap.add_argument("--base-notebook", required=True, help="public 0.947 notebook (.ipynb), not redistributed")
    ap.add_argument("--base-metadata", required=True, help="kernel-metadata.json of the public notebook")
    ap.add_argument("--owner", default="<your-kaggle-username>", help="Kaggle username that owns the new kernel")
    ap.add_argument("--dids", default="", help="comma-separated training crop ids (default: VAL19)")
    ap.add_argument("--name", default="validador-val19", help="kernel name (prefixed with 'biohub-')")
    ap.add_argument("--env", default="",
                    help="comma-separated BIOHUB_* overrides set before the constants cell, e.g. BIOHUB_DET_THRESHOLD=0.95")
    ap.add_argument("--out-dir", default=None, help="output directory (default: kaggle_biohub_<name>)")
    a = ap.parse_args()
    stems = a.dids.split(",") if a.dids else VAL19

    nb = json.load(io.open(a.base_notebook, encoding="utf-8"))
    code = [i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"]
    c6, c9, c11 = code[6], code[9], code[11]
    s6 = "".join(nb["cells"][c6]["source"]); s9 = "".join(nb["cells"][c9]["source"]); s11 = "".join(nb["cells"][c11]["source"])

    # Cell 6: a guard cell requires /kaggle/working/submission.csv, so the test submission is kept (~40 min).
    a6 = '\nwrite_test_submission("base")\n'
    assert s6.count(a6) == 1
    s6 = s6.replace(a6, '\nprint("FAITHFUL BENCH: test submission kept for the guard cell; HOCT disabled", flush=True)' + a6, 1)

    # Cell 0: the fork disables the validator -> enable it.
    c0 = code[0]
    s0 = "".join(nb["cells"][c0]["source"])
    a0 = 'os.environ["BIOHUB_VALIDATOR_ENABLE"] = "0"'
    assert s0.count(a0) == 1, s0.count(a0)
    s0 = s0.replace(a0, 'os.environ["BIOHUB_VALIDATOR_ENABLE"] = "1"  # FAITHFUL BENCH: validator enabled', 1)
    compile(s0, "cell0", "exec")
    nb["cells"][c0]["source"] = s0

    # Cell 1: turn the HOCT veto off (~17 min, not needed for raw graphs).
    c1 = code[1]
    s1 = "".join(nb["cells"][c1]["source"])
    a1 = 'os.environ["BIOHUB_HOCT_VETO"] = "2"'
    assert s1.count(a1) == 1, s1.count(a1)
    s1 = s1.replace(a1, a1 + '\nos.environ["BIOHUB_HOCT_VETO"] = "0"  # FAITHFUL BENCH: no veto, raw graphs only', 1)
    compile(s1, "cell1", "exec")
    nb["cells"][c1]["source"] = s1

    # Cell 9: force the validation crop list.
    a9 = "\n    print(val_stems)\n"
    assert s9.count(a9) == 1, s9.count(a9)
    s9 = s9.replace(a9, a9 + _indent(INJ9 % dict(val19=stems)), 1)

    # Cell 11: dump the raw graphs.
    a11 = '\n    print(f"VALIDATOR: cached {len(VAL_RAW_GRAPHS)} raw prediction graphs + GT")\n'
    assert s11.count(a11) == 1, s11.count(a11)
    s11 = s11.replace(a11, a11 + _indent(INJ11), 1)

    # Cell 3 (optional): BIOHUB_* environment overrides before the constants are read.
    if a.env:
        c3 = code[3]
        s3 = "".join(nb["cells"][c3]["source"])
        NL = chr(10)
        ancla3 = NL + "import csv" + NL
        assert s3.count(ancla3) == 1
        lines = ["", "import os as _dv_os1"]
        for pair in a.env.split(","):
            k, v = pair.split("=", 1)
            assert k.startswith("BIOHUB_"), k
            lines.append("_dv_os1.environ[%r] = %r" % (k, v))
        lines += ["print('CONSTANTS FROM ENVIRONMENT:', %r, flush=True)" % a.env, "import csv", ""]
        s3 = s3.replace(ancla3, NL.join(lines), 1)
        compile(s3, "cell3", "exec")
        nb["cells"][c3]["source"] = s3

    for ci, s in ((c6, s6), (c9, s9), (c11, s11)):
        compile(s, f"cell{ci}", "exec")
        nb["cells"][ci]["source"] = s
    assert "\n".join("".join(c["source"]) for c in nb["cells"]).isascii(), "notebook is not pure ASCII"

    name = "biohub-" + a.name
    d = a.out_dir or "kaggle_" + name.replace("-", "_")
    os.makedirs(d, exist_ok=True)
    f = os.path.join(d, name.replace("-", "_") + ".ipynb")
    json.dump(nb, io.open(f, "w", encoding="utf-8"), ensure_ascii=False)
    m = json.load(io.open(a.base_metadata, encoding="utf-8"))
    m.update(id=a.owner + "/" + name, title=name, code_file=os.path.basename(f))
    io.open(os.path.join(d, "kernel-metadata.json"), "w", encoding="utf-8").write(json.dumps(m, indent=1))
    print(f, "| cells", c6, c9, c11)


if __name__ == "__main__":
    main()
