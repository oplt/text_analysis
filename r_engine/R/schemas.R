validate_manifest <- function(manifest) {
  if (!is.list(manifest)) stop("Invalid manifest", call. = FALSE)
  required <- c("schema_version", "analysis", "identity", "inputs", "output")
  if (!all(required %in% names(manifest))) stop("Invalid manifest", call. = FALSE)
  if (!identical(manifest$schema_version, "1.0")) stop("Unsupported manifest schema", call. = FALSE)
  if (!is.list(manifest$analysis) || !is.character(manifest$analysis$type) || length(manifest$analysis$type) != 1L) stop("Invalid analysis manifest", call. = FALSE)
  if (!manifest$analysis$type %in% c("frequencies", "dfm", "kwic", "dictionary", "keyness", "cooccurrence")) stop("Unsupported R analysis", call. = FALSE)
  if (!is.list(manifest$identity) || !all(c("spec_hash", "corpus_checksum", "pipeline_checksum", "engine_name", "engine_version") %in% names(manifest$identity))) stop("Invalid manifest identity", call. = FALSE)
  if (!is.list(manifest$inputs) || !all(c("units", "tokens", "metadata") %in% names(manifest$inputs))) stop("Invalid manifest inputs", call. = FALSE)
  if (!is.list(manifest$output) || !is.character(manifest$output$result) || length(manifest$output$result) != 1L || basename(manifest$output$result) != manifest$output$result) stop("Invalid manifest output", call. = FALSE)
  invisible(TRUE)
}

runtime_info <- function() {
  packages <- c("quanteda", "quanteda.textstats", "arrow", "jsonlite")
  versions <- lapply(packages, function(pkg) as.character(utils::packageVersion(pkg)))
  names(versions) <- packages
  list(
    engine = "r",
    implementation = "quanteda",
    runtime_version = R.version.string,
    package_versions = versions
  )
}

new_result <- function(manifest, results, warnings = list(), diagnostics = list(), artifacts = list()) list(
  schema_version = "1.0", analysis_type = manifest$analysis$type, runtime = runtime_info(),
  identity = manifest$identity, results = results, warnings = warnings, diagnostics = diagnostics,
  artifacts = artifacts, timing = list(elapsed_seconds = NULL)
)
