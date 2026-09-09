#!/usr/bin/env Rscript
# Generate quanteda tokens/DFM JSON fixtures when quanteda is installed.
# Intended for optional CI parity checks; Python fixtures remain the default truth.

args <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args) >= 1) args[[1]] else "."
out_path <- file.path(out_dir, "quanteda_tiny_corpus.json")

if (!requireNamespace("quanteda", quietly = TRUE)) {
  message("quanteda is not installed; skipping fixture generation")
  quit(status = 0)
}

suppressPackageStartupMessages({
  library(quanteda)
  library(jsonlite)
})

corpus_texts <- c("alpha beta gamma", "beta delta")
corp <- corpus(corpus_texts)
toks <- tokens(corp, remove_punct = TRUE, remove_numbers = FALSE)
toks <- tokens_tolower(toks)
dfm_mat <- dfm(toks)

payload <- list(
  corpus = corpus_texts,
  tokens = lapply(toks, as.character),
  dfm = list(
    n_features = ncol(dfm_mat),
    vocabulary_sorted = sort(colnames(dfm_mat))
  ),
  generator = list(
    package = "quanteda",
    version = as.character(packageVersion("quanteda"))
  )
)

jsonlite::write_json(payload, out_path, auto_unbox = TRUE, pretty = TRUE)
message("Wrote ", out_path)
