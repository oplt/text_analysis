`%||%` <- function(x, y) if (is.null(x)) y else x

keyness_method <- function(value) {
  aliases <- c(g2 = "log_likelihood", ll = "log_likelihood", chi2 = "chi_square", fisher_exact = "fisher")
  method <- aliases[[tolower(value %||% "log_likelihood")]] %||% tolower(value %||% "log_likelihood")
  if (!method %in% c("log_likelihood", "chi_square", "fisher")) stop("Unsupported keyness method")
  method
}

g2_statistic <- function(a, b, total_a, total_b) {
  observed <- c(a, b, total_a - a, total_b - b)
  row_totals <- c(a + b, total_a + total_b - a - b)
  col_totals <- c(total_a, total_b); grand <- total_a + total_b
  if (!grand || any(row_totals == 0) || any(col_totals == 0)) return(0)
  expected <- c(row_totals[1] * col_totals[1], row_totals[1] * col_totals[2], row_totals[2] * col_totals[1], row_totals[2] * col_totals[2]) / grand
  2 * sum(ifelse(observed > 0 & expected > 0, observed * log(observed / expected), 0))
}

keyness_row <- function(term, a, b, total_a, total_b, method) {
  table <- matrix(c(a, total_a - a, b, total_b - b), nrow = 2, byrow = TRUE)
  g2 <- g2_statistic(a, b, total_a, total_b)
  g2_p <- if (g2 > 0) stats::pchisq(g2, df = 1, lower.tail = FALSE) else 1
  chi <- tryCatch(stats::chisq.test(table, correct = FALSE), error = function(...) NULL)
  chi_square <- if (is.null(chi)) 0 else unname(chi$statistic)
  chi_p <- if (is.null(chi)) 1 else chi$p.value
  fisher <- tryCatch(stats::fisher.test(table), error = function(...) NULL)
  fisher_odds <- if (is.null(fisher)) NULL else unname(fisher$estimate)
  fisher_p <- if (is.null(fisher)) 1 else fisher$p.value
  statistic <- switch(method, log_likelihood = g2, chi_square = chi_square, fisher = -log10(max(fisher_p, 1e-300)))
  p_value <- switch(method, log_likelihood = g2_p, chi_square = chi_p, fisher = fisher_p)
  rate_a <- if (total_a) a / total_a else 0; rate_b <- if (total_b) b / total_b else 0
  list(feature = term, freq_a = a, freq_b = b, rate_a = rate_a, rate_b = rate_b, keyness_statistic = statistic, g2 = g2, g2_p_value = g2_p, chi_square = chi_square, chi_square_p_value = chi_p, phi = if (total_a + total_b) sqrt(max(chi_square, 0) / (total_a + total_b)) else 0, fisher_odds_ratio = fisher_odds, fisher_p_value = fisher_p, p_value = p_value, effect_direction = if (rate_a > rate_b) "a" else if (rate_b > rate_a) "b" else "neutral", log_ratio = log2(((a + 0.5) / max(total_a, 1)) / ((b + 0.5) / max(total_b, 1))), odds_ratio = ((a + 0.5) / (total_a - a + 0.5)) / ((b + 0.5) / (total_b - b + 0.5)))
}

run_keyness <- function(manifest, inputs) {
  if (is.null(inputs$tokens_b)) stop("Keyness requires comparison corpus tokens")
  params <- manifest$analysis$parameters; method <- keyness_method(params$method)
  correction <- tolower(params$correction %||% "bh")
  if (!correction %in% c("bh", "fdr", "benjamini_hochberg", "none")) stop("Unsupported keyness correction")
  min_frequency <- as.integer(params$min_frequency %||% 1L); top_n <- as.integer(params$top_n %||% 50L)
  tokens_a <- as.character(inputs$tokens$token); tokens_b <- as.character(inputs$tokens_b$token)
  counts_a <- table(tokens_a); counts_b <- table(tokens_b); terms <- sort(union(names(counts_a), names(counts_b)))
  rows <- lapply(terms, function(term) keyness_row(term, as.integer(counts_a[[term]] %||% 0L), as.integer(counts_b[[term]] %||% 0L), length(tokens_a), length(tokens_b), method))
  rows <- Filter(function(row) row$freq_a + row$freq_b >= min_frequency, rows)
  p_values <- vapply(rows, function(row) row$p_value, numeric(1))
  adjusted <- if (correction == "none") p_values else stats::p.adjust(p_values, method = "BH")
  for (index in seq_along(rows)) rows[[index]]$p_value_adjusted <- adjusted[[index]]
  rows <- rows[order(vapply(rows, function(row) -row$keyness_statistic, numeric(1)), vapply(rows, function(row) row$feature, character(1)))]
  rows <- head(rows, top_n)
  new_result(manifest, list(method = method, correction = if (correction == "none") NULL else "bh", min_frequency = min_frequency, top_n = top_n, group_a = list(label = params$group_a_label %||% NULL, unit_count = nrow(inputs$units), token_count = length(tokens_a)), group_b = list(label = params$group_b_label %||% NULL, unit_count = nrow(inputs$units_b), token_count = length(tokens_b)), features = rows))
}
