#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) stop("Usage: run_analysis.R <manifest.json>")
script_arg <- commandArgs()[grep("^--file=", commandArgs())][1]
engine_root <- dirname(normalizePath(sub("^--file=", "", script_arg)))
if (requireNamespace("renv", quietly = TRUE)) renv::load(project = engine_root)
for (name in c("io.R", "schemas.R", "frequencies.R", "dfm.R", "kwic.R", "dictionary.R", "keyness.R", "cooccurrence.R", "dispatcher.R")) source(file.path(engine_root, "R", name))
manifest <- read_manifest(args[[1]])
validate_manifest(manifest)
inputs <- read_inputs(manifest, dirname(normalizePath(args[[1]])))
started <- proc.time()[["elapsed"]]
result <- dispatch_analysis(manifest, inputs)
result$timing <- list(elapsed_seconds = unname(proc.time()[["elapsed"]] - started))
write_result(result, file.path(dirname(normalizePath(args[[1]])), manifest$output$result))
