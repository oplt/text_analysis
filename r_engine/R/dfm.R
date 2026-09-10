`%||%` <- function(x, y) if (is.null(x)) y else x

.bounded_preview <- function(values, limit = 200L) {
  n <- length(values)
  if (n <= limit) return(as.list(values))
  as.list(values[seq_len(limit)])
}

run_dfm <- function(manifest, inputs) {
  units <- as.character(inputs$units$unit_id)
  token_rows <- inputs$tokens[order(inputs$tokens$unit_id, inputs$tokens$token_position), ]
  grouped <- split(as.character(token_rows$token), as.character(token_rows$unit_id))
  # Preserve canonical Python tokens exactly — never paste/re-tokenize.
  sequences <- lapply(units, function(id) as.character(grouped[[id]] %||% character()))
  names(sequences) <- units
  toks <- quanteda::as.tokens(sequences)
  matrix <- quanteda::dfm(toks)
  sparse <- methods::as(matrix, "dgTMatrix")
  feature_names <- quanteda::featnames(matrix)
  rows <- as.integer(sparse@i)
  cols <- as.integer(sparse@j)
  values <- as.numeric(sparse@x)
  artifacts_dir <- file.path(dirname(manifest$output$result), manifest$output$artifacts_directory %||% "artifacts")
  dir.create(artifacts_dir, recursive = TRUE, showWarnings = FALSE)
  coo_path <- file.path(artifacts_dir, "dfm_sparse_coo.parquet")
  features_path <- file.path(artifacts_dir, "dfm_features.parquet")
  units_path <- file.path(artifacts_dir, "dfm_units.parquet")
  arrow::write_parquet(data.frame(row = rows, col = cols, value = values), coo_path)
  arrow::write_parquet(data.frame(feature_index = seq_along(feature_names) - 1L, feature = feature_names), features_path)
  arrow::write_parquet(data.frame(unit_index = seq_along(units) - 1L, unit_id = units), units_path)
  preview_n <- min(length(values), 200L)
  cells <- list(
    rows = .bounded_preview(rows, preview_n),
    cols = .bounded_preview(cols, preview_n),
    values = .bounded_preview(values, preview_n),
    preview_only = length(values) > preview_n,
    preview_limit = preview_n
  )
  new_result(
    manifest,
    list(
      dimensions = list(documents = nrow(matrix), features = ncol(matrix)),
      feature_names = as.list(feature_names),
      summary = list(
        unit_count = nrow(matrix),
        feature_count = ncol(matrix),
        nnz = length(values)
      ),
      nnz = length(values),
      sparse_coo = cells,
      unit_ids = as.list(units)
    ),
    artifacts = list(
      list(
        kind = "dfm_sparse_coo",
        name = "dfm_sparse_coo.parquet",
        format = "parquet_coo",
        nnz = length(values),
        shape = list(nrow(matrix), ncol(matrix))
      ),
      list(kind = "dfm_features", name = "dfm_features.parquet", format = "parquet"),
      list(kind = "dfm_units", name = "dfm_units.parquet", format = "parquet")
    )
  )
}
