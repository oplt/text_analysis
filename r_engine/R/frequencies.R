run_frequencies <- function(manifest, inputs) {
  params <- manifest$analysis$parameters
  top_n <- if (is.null(params$top_n)) 50L else as.integer(params$top_n)
  rate_per <- if (is.null(params$rate_per)) 1000 else as.numeric(params$rate_per)
  tokens <- as.character(inputs$tokens$token)
  counts <- sort(table(tokens), decreasing = TRUE)
  terms <- names(counts); ord <- order(-as.integer(counts), terms); counts <- counts[ord]; terms <- terms[ord]
  keep <- seq_len(min(length(terms), top_n)); total <- length(tokens)
  rows <- lapply(keep, function(i) list(term = terms[[i]], count = as.integer(counts[[i]]), relative_frequency = if (total == 0) 0 else as.numeric(counts[[i]]) / total * rate_per, rank = as.integer(i)))
  new_result(manifest, list(frequencies = rows, metadata = list(unit_count = nrow(inputs$units), token_count = total, vocabulary_size = length(terms), terms_returned = length(rows), rate_per = rate_per)))
}
