read_manifest <- function(path) {
  if (!is.character(path) || length(path) != 1L || !file.exists(path)) stop("Manifest file is missing", call. = FALSE)
  tryCatch(jsonlite::fromJSON(path, simplifyVector = FALSE), error = function(error) stop("Manifest is not valid JSON", call. = FALSE))
}

input_path <- function(root, value, label) {
  if (!is.character(value) || length(value) != 1L || !nzchar(value) || basename(value) != value) stop(paste("Invalid", label, "input path"), call. = FALSE)
  path <- file.path(root, value)
  if (!file.exists(path)) stop(paste("Missing", label, "input file"), call. = FALSE)
  path
}

read_inputs <- function(manifest, root) {
  inputs <- list(
  units = arrow::read_parquet(input_path(root, manifest$inputs$units, "units")),
  tokens = arrow::read_parquet(input_path(root, manifest$inputs$tokens, "tokens")),
  metadata = arrow::read_parquet(input_path(root, manifest$inputs$metadata, "metadata"))
  )
  if (!is.null(manifest$inputs$tokens_b)) {
    inputs$units_b <- arrow::read_parquet(input_path(root, manifest$inputs$units_b, "units_b"))
    inputs$tokens_b <- arrow::read_parquet(input_path(root, manifest$inputs$tokens_b, "tokens_b"))
  }
  inputs
}

write_result <- function(result, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  jsonlite::write_json(result, path, auto_unbox = TRUE, null = "null", pretty = FALSE)
}
