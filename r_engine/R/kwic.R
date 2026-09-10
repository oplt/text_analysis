`%||%` <- function(x, y) if (is.null(x)) y else x

run_kwic <- function(manifest, inputs) {
  params <- manifest$analysis$parameters; keyword <- as.character(params$keyword %||% "")
  if (nchar(keyword) == 0) stop("keyword is required")
  window <- as.integer(params$window_size %||% 5L)
  case_sensitive <- isTRUE(params$case_sensitive %||% FALSE)
  token_rows <- inputs$tokens[order(inputs$tokens$unit_id, inputs$tokens$token_position), ]
  grouped <- split(token_rows, as.character(token_rows$unit_id)); matches <- list()
  for (unit_id in names(grouped)) {
    values <- as.character(grouped[[unit_id]]$token)
    where <- if (case_sensitive) which(values == keyword) else which(tolower(values) == tolower(keyword))
    for (position in where) {
      left_values <- if (position <= 1L) character() else values[seq.int(max(1L, position - window), position - 1L)]
      right_values <- if (position >= length(values)) character() else values[seq.int(position + 1L, min(length(values), position + window))]
      matches[[length(matches) + 1L]] <- list(
        text_unit_id = unit_id,
        id = unit_id,
        left_context = paste(left_values, collapse = " "),
        keyword = values[[position]],
        right_context = paste(right_values, collapse = " "),
        token_start = as.integer(position - 1L),
        token_end = as.integer(position)
      )
    }
  }
  new_result(manifest, list(matches = matches, match_count = length(matches)))
}
