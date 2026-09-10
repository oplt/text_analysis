`%||%` <- function(x, y) if (is.null(x)) y else x

dictionary_match_type <- function(expression, explicit = NULL) {
  candidate <- tolower(explicit %||% "")
  aliases <- c(exact = "token", word = "token", glob = "wildcard", exact_phrase = "phrase")
  if (candidate %in% names(aliases)) candidate <- aliases[[candidate]]
  if (candidate %in% c("token", "phrase", "wildcard", "regex")) return(candidate)
  if (nzchar(candidate)) stop("Unsupported dictionary match type")
  if (grepl("[*?]", expression)) return("wildcard")
  if (grepl("\\s", expression)) return("phrase")
  "token"
}

new_dictionary_entry <- function(item, path = character()) {
  if (is.character(item) && length(item) == 1L) {
    expression <- trimws(item)
    explicit <- NULL
    category <- NULL
    subcategory <- NULL
    entry_path <- NULL
  } else if (is.list(item) && !is.null(item$expression %||% item$term %||% item$phrase)) {
    expression <- trimws(as.character(item$expression %||% item$term %||% item$phrase))
    explicit <- item$match %||% item$match_type
    category <- item$category
    subcategory <- item$subcategory
    entry_path <- item$path
  } else {
    stop("Invalid dictionary entry")
  }
  if (!nzchar(expression)) stop("Dictionary expression must be non-empty")
  list(
    expression = expression,
    match_type = dictionary_match_type(expression, explicit),
    category = category %||% if (length(path)) path[[1]] else NULL,
    subcategory = subcategory %||% if (length(path) > 1L) paste(path[-1], collapse = "/") else NULL,
    path = entry_path %||% as.list(path)
  )
}

walk_dictionary <- function(node, path = character()) {
  if (is.character(node)) return(lapply(as.list(node), new_dictionary_entry, path = path))
  if (!is.list(node)) stop("Invalid dictionary hierarchy")
  if (!is.null(node$expression %||% node$term %||% node$phrase)) return(list(new_dictionary_entry(node, path)))
  names_node <- names(node)
  if (is.null(names_node)) return(unlist(lapply(node, new_dictionary_entry, path = path), recursive = FALSE))
  unlist(Map(function(name, value) walk_dictionary(value, c(path, name)), names_node, node), recursive = FALSE)
}

dictionary_entries <- function(params) {
  raw <- params$hierarchy %||% params$entries %||% params$dictionary_terms %||% params$terms
  if (is.null(raw)) stop("Dictionary definition is required")
  entries <- walk_dictionary(raw)
  if (!length(entries)) stop("Dictionary has no entries")
  exclusions <- if (is.null(params$exclusions)) list() else walk_dictionary(params$exclusions)
  list(entries = entries, exclusions = exclusions)
}

token_key <- function(value, case_sensitive) {
  normalized <- gsub("[^[:alnum:]_']", "", value)
  if (case_sensitive) normalized else tolower(normalized)
}

entry_spans <- function(tokens, entry, case_sensitive) {
  n <- length(tokens)
  if (!n) return(list())
  type <- entry$match_type
  if (type == "token") {
    found <- which(token_key(tokens, case_sensitive) == token_key(entry$expression, case_sensitive))
    return(lapply(found, function(i) list(start = i, end = i, tokens = tokens[[i]])))
  }
  if (type == "phrase" || type == "wildcard") {
    terms <- strsplit(trimws(entry$expression), "\\s+")[[1]]
    width <- length(terms)
    if (width > n) return(list())
    spans <- list()
    for (start in seq_len(n - width + 1L)) {
      values <- tokens[start:(start + width - 1L)]
      matches <- if (type == "phrase") {
        token_key(values, case_sensitive) == token_key(terms, case_sensitive)
      } else {
        vapply(seq_along(values), function(i) {
          pattern <- gsub("([.\\^$|()\\[\\]{}+])", "\\\\\\1", terms[[i]])
          pattern <- gsub("\\*", ".*", gsub("\\?", ".", pattern))
          grepl(paste0("^", pattern, "$"), values[[i]], ignore.case = !case_sensitive)
        }, logical(1))
      }
      if (all(matches)) spans[[length(spans) + 1L]] <- list(start = start, end = start + width - 1L, tokens = values)
    }
    return(spans)
  }
  if (type == "regex") {
    found <- which(grepl(entry$expression, tokens, ignore.case = !case_sensitive, perl = TRUE))
    return(lapply(found, function(i) list(start = i, end = i, tokens = tokens[[i]])))
  }
  stop("Unsupported dictionary match type")
}

overlaps_exclusion <- function(span, exclusions) any(vapply(exclusions, function(exclusion) span$start <= exclusion$end && exclusion$start <= span$end, logical(1)))

run_dictionary <- function(manifest, inputs) {
  params <- manifest$analysis$parameters
  parsed <- dictionary_entries(params)
  case_sensitive <- isTRUE(params$case_sensitive %||% FALSE)
  rate_per <- as.numeric(params$rate_per %||% 1000)
  if (!is.finite(rate_per) || rate_per <= 0) stop("rate_per must be positive")
  rows <- inputs$tokens[order(inputs$tokens$unit_id, inputs$tokens$token_position), ]
  grouped <- split(as.character(rows$token), as.character(rows$unit_id))
  unit_ids <- as.character(inputs$units$unit_id)
  matches <- list(); per_unit <- list(); categories <- list(); total_hits <- 0L
  for (unit_id in unit_ids) {
    tokens <- grouped[[unit_id]] %||% character()
    exclusions <- unlist(lapply(parsed$exclusions, entry_spans, tokens = tokens, case_sensitive = case_sensitive), recursive = FALSE)
    unit_hits <- 0L
    for (entry in parsed$entries) {
      for (span in entry_spans(tokens, entry, case_sensitive)) {
        if (overlaps_exclusion(span, exclusions)) next
        unit_hits <- unit_hits + 1L; total_hits <- total_hits + 1L
        matches[[length(matches) + 1L]] <- list(text_unit_id = unit_id, matched_expression = entry$expression, match_type = entry$match_type, matched_tokens = as.list(span$tokens), token_start = span$start - 1L, token_end = span$end, category = entry$category, subcategory = entry$subcategory, path = entry$path)
        key <- entry$category %||% "_uncategorized"
        bucket <- categories[[key]] %||% list(category = entry$category, hits = 0L)
        bucket$hits <- bucket$hits + 1L; categories[[key]] <- bucket
      }
    }
    per_unit[[length(per_unit) + 1L]] <- list(text_unit_id = unit_id, hits = unit_hits)
  }
  n_tokens <- nrow(inputs$tokens); units_with_hit <- sum(vapply(per_unit, function(row) row$hits > 0L, logical(1)))
  new_result(manifest, list(total_hits = total_hits, normalized_hits = if (n_tokens) total_hits / n_tokens * rate_per else 0, hits_per_1000_tokens = if (n_tokens) total_hits / n_tokens * 1000 else 0, rate_per = rate_per, unit_prevalence = if (length(unit_ids)) units_with_hit / length(unit_ids) else 0, document_prevalence = if (length(unit_ids)) units_with_hit / length(unit_ids) else 0, n_units = length(unit_ids), n_tokens = n_tokens, units_with_hit = units_with_hit, per_unit = per_unit, matches = matches, by_category = categories, dictionary = list(entry_count = length(parsed$entries), exclusion_count = length(parsed$exclusions), source = "user"), case_sensitive = case_sensitive))
}
