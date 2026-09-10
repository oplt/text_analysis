`%||%` <- function(x, y) if (is.null(x)) y else x

run_dfm <- function(manifest, inputs) {
  units <- as.character(inputs$units$unit_id)
  token_rows <- inputs$tokens[order(inputs$tokens$unit_id, inputs$tokens$token_position), ]
  grouped <- split(as.character(token_rows$token), as.character(token_rows$unit_id))
  texts <- vapply(units, function(id) paste(grouped[[id]] %||% character(), collapse = " "), FUN.VALUE = "")
  corpus <- quanteda::corpus(data.frame(doc_id = units, text = texts), text_field = "text")
  matrix <- quanteda::dfm(quanteda::tokens(corpus, what = "word", remove_punct = FALSE))
  new_result(manifest, list(dimensions = list(documents = nrow(matrix), features = ncol(matrix)), feature_names = as.list(quanteda::featnames(matrix)), summary = list(unit_count = nrow(matrix), feature_count = ncol(matrix), nnz = sum(matrix != 0)), unit_ids = as.list(units)))
}
