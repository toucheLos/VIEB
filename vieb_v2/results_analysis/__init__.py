"""Cross-method comparison of the v2 behavior-detection arms.

Four state-discovery algorithms have now been run on Luna, over two latents,
and their outputs are scattered across a dozen result directories in three
incompatible conventions. Nothing reads them together.

This package does three things:

`collect`      harvests every run's JSON into one normalized record per arm.
`discriminate` scores each arm on the *same* axis as the MoSeq positive
               control -- does its state set separate Context A after
               conditioning -- which is the only metric here that is about
               behavior rather than geometry.
`plots`        renders the comparison.

The design rule throughout: a geometric metric (state count, entropy, noise
fraction) describes a partition, it does not validate one. A decomposition that
is elegant and does not separate the post-shock condition is a negative result
(`scripts/moseq_control.py:11`). So every geometric number in here is reported
beside the discrimination score, never instead of it.
"""

__all__ = ["collect", "discriminate", "plots"]
