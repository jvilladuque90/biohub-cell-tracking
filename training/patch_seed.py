# -*- coding: utf-8 -*-
"""Seed control for the organizer's `train_unet_transformer.py`: train independent copies of the model
(small seed bank -> ensemble curve) and resume training across recycled sessions.

  S1 --seed N          torch/numpy/random seeds + DataLoader generator (train(seed=...)).
  S2 --init PATH       resume from a full model state (strict=False): Colab sessions are recycled, so
                       training runs in SEGMENTS and each segment starts from the previous one.
  S3 last per epoch    besides edge_predictor_best.pth (best acc*recall), save edge_predictor_last.pth
                       and last.json {epoch, score} after EVERY epoch; that is what is carried between
                       segments.

Every patch is an exact old-string replacement (assert count == 1) followed by compile().
Apply before training/patch_fp16.py and training/patch_score.py (the latter matches text inserted here).

Usage: python training/patch_seed.py <repo_dir>
"""
from __future__ import annotations

import io
import os
import sys

TRAIN = "scripts/train_unet_transformer.py"

P_TRAIN = [
    # S1/S2: CLI flags
    ('    parser.add_argument("--epochs", type=int, default=50)\n',
     '    parser.add_argument("--epochs", type=int, default=50)\n'
     '    parser.add_argument("--seed", type=int, default=None)\n'
     '    parser.add_argument("--init", type=str, default=None,\n'
     '                        help="full model state to resume from (strict=False)")\n'),
    # S1: pass the seed to train()
    ('            data_parallel=args.data_parallel,\n'
     '        )\n',
     '            data_parallel=args.data_parallel,\n'
     '            seed=args.seed,\n'
     '        )\n'),
    # S3: last checkpoint path (next to the best one)
    ('    best_score = 0.0\n'
     '    save_path = output_dir / "edge_predictor_best.pth"\n',
     '    best_score = 0.0\n'
     '    save_path = output_dir / "edge_predictor_best.pth"\n'
     '    last_path = output_dir / "edge_predictor_last.pth"\n'),
]


def apply_patches(path, patches):
    s = io.open(path, encoding="utf-8").read()
    for old, new in patches:
        n = s.count(old)
        assert n == 1, (path, n, old[:60])
        s = s.replace(old, new, 1)
    return s


def main(repo):
    t_path = os.path.join(repo, TRAIN)
    s = apply_patches(t_path, P_TRAIN)
    # S2: load the --init state right after the model is created.
    # NOTE: the inserted load/print lines are an old-string for training/patch_score.py (G9): keep them verbatim.
    anchor = '    ).to(device)\n'
    i = s.find('    model = UNetNodeTransformer(')
    j = s.find(anchor, i)
    assert i > 0 and j > i, "model creation not found"
    insert = (anchor +
              '    _init = globals().get("_INIT_STATE")\n'
              '    if _init:\n'
              '        _st = torch.load(_init, map_location="cpu", weights_only=True)\n'
              '        _mis, _un = model.load_state_dict(_st, strict=False)\n'
              '        print(f"  reanudo desde {_init}: {len(_mis)} missing, {len(_un)} unexpected", flush=True)\n')
    s = s[:j] + insert + s[j + len(anchor):]
    # S3: save the last state after every epoch, hooked just before the best-checkpoint check
    k = s.find('        is_best = score >= best_score\n')
    assert k > 0, "is_best not found"
    s = (s[:k] + '        torch.save(_estado_modelo(model), last_path)\n'
         '        (output_dir / "last.json").write_text(\n'
         '            __import__("json").dumps({"epoch": int(epoch) + 1, "score": float(score)}))\n' + s[k:])
    # same state dict as the organizer's best checkpoint (DataParallel prefix normalized)
    NL = chr(10)
    s = s.replace("def train(" + NL,
                  "def _estado_modelo(model):" + NL
                  + '    return {k.replace("unet.module.", "unet.", 1): v for k, v in model.state_dict().items()}' + NL
                  + NL + NL + "def train(" + NL, 1)
    # S1/S2: global seeds and --init at the start of main
    k2 = s.find('    args = parser.parse_args()')
    assert k2 > 0
    end2 = s.find('\n', k2) + 1
    s = (s[:end2] + '    globals()["_INIT_STATE"] = args.init\n'
         '    if args.seed is not None:\n'
         '        import random as _random\n'
         '        _random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)\n'
         '        torch.cuda.manual_seed_all(args.seed)\n' + s[end2:])
    compile(s, t_path, "exec")
    io.open(t_path, "w", encoding="utf-8").write(s)
    print("seed patch applied to", repo)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0 if len(sys.argv) == 2 else 2)
    main(sys.argv[1])
