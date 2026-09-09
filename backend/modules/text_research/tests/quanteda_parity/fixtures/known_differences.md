# Quanteda parity — known differences

Production remains Python-native. Fixtures store Python reference outputs for
the research tokenizer/DFM path (regression parity). R is not required in CI.

Intentional differences vs R `quanteda` when compared externally:

* Unicode / punctuation class handling may differ from `tokens()`.
* Stopword lists come from research language profiles, not quanteda defaults.
* Stemming uses `snowballstemmer` and can diverge on edge cases.
* TF-IDF follows our weighting module / sklearn-compatible smooth IDF unless configured.
* Dictionary matching is hierarchical user-defined; quanteda wildcards are not fully mirrored.
* Compare floating-point weights with tolerances; bit-identical matrices are not required.
