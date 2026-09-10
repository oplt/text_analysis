library(testthat)
engine_root <- normalizePath(file.path(test_path(), "..", ".."))
for (name in c("io.R", "schemas.R", "frequencies.R", "dfm.R", "kwic.R", "dictionary.R", "keyness.R", "cooccurrence.R", "dispatcher.R")) source(file.path(engine_root, "R", name))

test_manifest <- function(type = "frequencies", parameters = list()) list(
  schema_version = "1.0", analysis = list(type = type, parameters = parameters),
  identity = list(spec_hash = "spec", corpus_checksum = "corpus", pipeline_checksum = "pipeline", engine_name = "r", engine_version = "r-quanteda-1"),
  inputs = list(units = "units.parquet", tokens = "tokens.parquet", metadata = "metadata.parquet"),
  output = list(result = "result.json")
)

test_inputs <- function() list(
  units = data.frame(unit_id = c("u1", "u2")),
  tokens = data.frame(unit_id = c("u1", "u1", "u2", "u2"), token_position = c(0L, 1L, 0L, 1L), token = c("alpha", "beta", "beta", "gamma")),
  metadata = data.frame(unit_id = c("u1", "u2"))
)
