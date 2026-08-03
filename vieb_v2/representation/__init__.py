"""VIEB v2 representation backend.

Pipeline: egocentric alignment -> pooled PCA -> delay embedding -> HDBSCAN.

See ../docs/REPRESENTATION_MATH.md for the derivations this implements, and in
particular for the two results that constrain the code:

  * the alignment closed form (section 1.3), and
  * the rank structure of the aligned covariance (section 1.4b), which
    per-frame confidence weights partially destroy.
"""
