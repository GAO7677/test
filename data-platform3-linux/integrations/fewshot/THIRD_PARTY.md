# Source provenance

The executable files in `vendor/` are byte-preserved snapshots of the local sources used by the original experiments. `vendor/sources.json` contains their SHA256 hashes and upstream commit identities. No runtime dependency on the old workstation repositories is required.

| Component | Origin | Scope |
|---|---|---|
| BeyondObjects training | https://github.com/princetonvisualai/BeyondObjects, `8bc17e0537991f6bb5c0399339bddb3d1418d220` | Original training, transform, tokenizer and optimizer code; the external wrapper supplies folder captions and monitoring |
| Chimera training | Existing ChimeraLoRA checkout, `c42069d4ede36d17da17ebe102da552e1624e8f5` | Experiment snapshot includes shared-A verification and logging changes; not described as an unmodified upstream checkout |
| FLUX trainer | https://github.com/huggingface/diffusers, `71774d9bc84600eb4c60522dbf85c4b8cc317e20` | Official FLUX2 training script; external wrapper supplies datasets and records |

Source headers and attribution are retained; this package does not relicense the included third-party code, base weights, or datasets. Model access/download terms remain those of their providers. The original repositories did not all include a root LICENSE in the local snapshots.
