# -*- coding: utf-8 -*-
"""Mixed precision (fp16) for the organizer's `train_unet_transformer.py`.

Measured in this project: fp16 autocast gives ~2x on a T4 with unchanged losses. The organizer's script
trains in fp32. Here autocast wraps ONLY the two heavy passes -- `model.encode` (3D UNet) and
`model.predict_edges` (transformer) -- in both training and evaluation; their outputs are cast back to
fp32 before the losses (the organizer's `F.binary_cross_entropy` on a softmax is not allowed under
autocast), and the optimizer step uses a GradScaler: scale the loss, unscale before gradient clipping,
and skip the step on inf/nan.
Enabled with --fp16 (off by default: without the flag the script behaves exactly as the original).

Apply after training/patch_seed.py and before training/patch_score.py (the latter matches text
inserted here, e.g. the GradScaler lines and `edge_logits = edge_logits.float()`).

Usage: python training/patch_fp16.py <repo_dir>
"""
from __future__ import annotations

import io
import os
import sys

TRAIN = "scripts/train_unet_transformer.py"
AC = '        with torch.autocast("cuda", dtype=torch.float16, enabled=_AMP):\n'

P = [
    # CLI flag
    ('    parser.add_argument("--epochs", type=int, default=50)\n',
     '    parser.add_argument("--epochs", type=int, default=50)\n'
     '    parser.add_argument("--fp16", action="store_true", help="mixed precision (fp16 autocast + GradScaler)")\n'),
    # training: encode under autocast, outputs back to fp32
    ('        # --- 1. Encode: UNet features + detection logits --------------------\n'
     '        unet_out, det_logits = model.encode(imgs)\n',
     '        # --- 1. Encode: UNet features + detection logits --------------------\n'
     + AC +
     '            unet_out, det_logits = model.encode(imgs)\n'
     '        det_logits = [d.float() for d in det_logits]\n'),
    # training: predict_edges under autocast (logits cast to fp32 in main())
    ('            edge_logits = model.predict_edges(\n',
     AC.replace('        with', '            with') +
     '              edge_logits = model.predict_edges(\n'),
    # presence check only (identity replacement)
    ('        edge_loss = sum(block_losses) / len(block_losses)\n',
     '        edge_loss = sum(block_losses) / len(block_losses)\n'),
    # optimizer step with GradScaler
    ('        optimizer.zero_grad()\n'
     '        loss.backward()\n'
     '        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)\n'
     '        optimizer.step()\n',
     '        optimizer.zero_grad()\n'
     '        _SCALER.scale(loss).backward()\n'
     '        _SCALER.unscale_(optimizer)\n'
     '        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)\n'
     '        _SCALER.step(optimizer)\n'
     '        _SCALER.update()\n'),
    # evaluation: encode under autocast
    ('        B, W = imgs.shape[:2]\n'
     '        unet_out, det_logits = model.encode(imgs)\n'
     '        frame_det: list[',
     '        B, W = imgs.shape[:2]\n'
     + AC +
     '            unet_out, det_logits = model.encode(imgs)\n'
     '        det_logits = [d.float() for d in det_logits]\n'
     '        frame_det: list['),
    # evaluation: predict_edges under autocast
    ('            pair_logits = model.predict_edges(\n',
     AC.replace('        with', '            with') +
     '              pair_logits = model.predict_edges(\n'),
]


def main(repo):
    p = os.path.join(repo, TRAIN)
    s = io.open(p, encoding="utf-8").read()
    for old, new in P:
        n = s.count(old)
        assert n == 1, (n, old[:70])
        s = s.replace(old, new, 1)
    # Edge logits back to fp32 right after each predict_edges call (train and eval):
    # the closing parenthesis is the first "            )" line after the call.
    for name in ("edge_logits", "pair_logits"):
        i = s.find(f"              {name} = model.predict_edges(")
        j = s.find("\n            )\n", i) + len("\n            )\n")
        s = s[:j] + f"            {name} = {name}.float()\n" + s[j:]
    # globals _AMP and _SCALER from the flag
    k = s.find('    args = parser.parse_args()')
    assert k > 0
    end = s.find('\n', k) + 1
    s = (s[:end] + '    globals()["_AMP"] = bool(args.fp16) and torch.cuda.is_available()\n'
         '    globals()["_SCALER"] = torch.amp.GradScaler("cuda", enabled=globals()["_AMP"])\n'
         '    print(f"  mixed precision fp16: {globals()[\'_AMP\']}", flush=True)\n' + s[end:])
    # defaults when train() is imported without running main()
    s = s.replace('def train_epoch(\n', '_AMP = False\n_SCALER = torch.amp.GradScaler("cuda", enabled=False)\n\n\ndef train_epoch(\n', 1)
    compile(s, p, "exec")
    io.open(p, "w", encoding="utf-8").write(s)
    print("fp16 patch applied to", repo)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0 if len(sys.argv) == 2 else 2)
    main(sys.argv[1])
