`%||%` <- function(x, y) if (is.null(x)) y else x

cooccurrence_method <- function(value) {
  method <- tolower(value %||% "pmi")
  if (!method %in% c("count", "pmi", "npmi", "dice", "log_dice", "t_score")) stop("Unsupported co-occurrence method")
  method
}

pair_metrics <- function(count, freq_a, freq_b, n_tokens) {
  pmi <- if (count && freq_a && freq_b && n_tokens) log2((count * n_tokens) / (freq_a * freq_b)) else 0
  npmi <- if (count && n_tokens && count < n_tokens) pmi / -log2(count / n_tokens) else 0
  dice <- if (freq_a + freq_b) 2 * count / (freq_a + freq_b) else 0
  log_dice <- if (dice > 0) 14 + log2(dice) else 0
  t_score <- if (count && n_tokens) (count - (freq_a * freq_b / n_tokens)) / sqrt(count) else 0
  list(count = count, pmi = pmi, npmi = npmi, dice = dice, log_dice = log_dice, t_score = t_score)
}

run_cooccurrence <- function(manifest, inputs) {
  params <- manifest$analysis$parameters; method <- cooccurrence_method(params$association_method)
  window <- as.integer(params$window_size %||% 5L); top_n <- as.integer(params$top_n %||% 50L)
  min_frequency <- as.integer(params$min_frequency %||% 1L); min_count <- as.integer(params$min_count %||% 1L)
  directional <- isTRUE(params$directional %||% FALSE)
  if (window < 1L || top_n < 1L || min_frequency < 0L || min_count < 0L) stop("Invalid co-occurrence parameters")
  rows <- inputs$tokens[order(inputs$tokens$unit_id, inputs$tokens$token_position), ]
  grouped <- split(as.character(rows$token), as.character(rows$unit_id)); all_tokens <- as.character(rows$token)
  frequencies <- table(all_tokens); counts <- new.env(parent = emptyenv())
  for (tokens in grouped) if (length(tokens) > 1L) for (i in seq_len(length(tokens) - 1L)) for (j in seq.int(i + 1L, min(length(tokens), i + window))) {
    if (tokens[[i]] == tokens[[j]]) next
    pair <- if (directional) c(tokens[[i]], tokens[[j]]) else sort(c(tokens[[i]], tokens[[j]]))
    key <- paste(pair, collapse = "\u001f"); counts[[key]] <- (counts[[key]] %||% 0L) + 1L
  }
  keys <- sort(ls(counts)); output <- list(); n_tokens <- length(all_tokens)
  for (key in keys) {
    pair <- strsplit(key, "\u001f", fixed = TRUE)[[1]]; count <- counts[[key]]; freq_a <- as.integer(frequencies[[pair[[1]]]]); freq_b <- as.integer(frequencies[[pair[[2]]]])
    if (count < min_count || freq_a < min_frequency || freq_b < min_frequency) next
    metrics <- pair_metrics(count, freq_a, freq_b, n_tokens)
    output[[length(output) + 1L]] <- list(term_a = pair[[1]], term_b = pair[[2]], count = count, freq_a = freq_a, freq_b = freq_b, association_method = method, association_score = metrics[[method]], pmi = metrics$pmi, npmi = metrics$npmi, dice = metrics$dice, log_dice = metrics$log_dice, t_score = metrics$t_score, directional = directional, window = window)
  }
  output <- output[order(vapply(output, function(row) -row$association_score, numeric(1)), vapply(output, function(row) -row$count, numeric(1)), vapply(output, function(row) row$term_a, character(1)), vapply(output, function(row) row$term_b, character(1)))]
  output <- head(output, top_n)
  new_result(manifest, list(association_method = method, window = window, direction = if (directional) "directional" else "undirected", directional = directional, min_frequency = min_frequency, min_count = min_count, top_n = top_n, n_tokens = n_tokens, pairs_tested = length(keys), pairs_returned = length(output), pairs = output, available_methods = as.list(c("count", "pmi", "npmi", "dice", "log_dice", "t_score"))))
}
