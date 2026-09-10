test_that("frequencies are deterministic", {
  manifest <- test_manifest("frequencies", list(top_n = 10, rate_per = 1000))
  expect_identical(run_frequencies(manifest, test_inputs())$results$frequencies, run_frequencies(manifest, test_inputs())$results$frequencies)
  expect_equal(run_frequencies(manifest, test_inputs())$results$frequencies[[1]]$term, "beta")
})

test_that("DFM and KWIC produce canonical outputs", {
  inputs <- test_inputs()
  dfm <- run_dfm(test_manifest("dfm", list(weighting = "count")), inputs)
  expect_equal(dfm$results$dimensions$documents, 2L)
  expect_equal(dfm$results$dimensions$features, 3L)
  expect_true(!is.null(dfm$results$sparse_coo))
  kwic <- run_kwic(test_manifest("kwic", list(keyword = "beta", window_size = 1, query_mode = "word")), inputs)
  expect_length(kwic$results$matches, 2L)
  expect_true(all(vapply(kwic$results$matches, function(row) row$keyword == "beta", logical(1))))
})

test_that("dictionary aggregates document prevalence by document_id", {
  inputs <- list(
    units = data.frame(unit_id = c("u1", "u2", "u3", "u4"), document_id = c("d1", "d1", "d2", "d3")),
    tokens = data.frame(
      unit_id = c("u1", "u1", "u2", "u3", "u4"),
      token_position = c(0L, 1L, 0L, 0L, 0L),
      token = c("alpha", "beta", "alpha", "noise", "alpha")
    ),
    metadata = data.frame(unit_id = c("u1", "u2", "u3", "u4"))
  )
  result <- run_dictionary(
    test_manifest("dictionary", list(hierarchy = list(theme = list(core = c("alpha"))))),
    inputs
  )
  expect_equal(result$results$total_hits, 3L)
  expect_equal(result$results$unit_prevalence, 0.75)
  expect_equal(result$results$document_prevalence, 2 / 3)
  expect_true(result$results$document_prevalence != result$results$unit_prevalence)
})

test_that("keyness methods are deterministic and attach adjusted p-values", {
  inputs <- list(
    units = data.frame(unit_id = c("a1", "a2")),
    tokens = data.frame(
      unit_id = c("a1", "a1", "a2", "a2"),
      token_position = c(0L, 1L, 0L, 1L),
      token = c("freedom", "education", "freedom", "rights")
    ),
    units_b = data.frame(unit_id = c("b1", "b2")),
    tokens_b = data.frame(
      unit_id = c("b1", "b1", "b2", "b2"),
      token_position = c(0L, 1L, 0L, 1L),
      token = c("market", "growth", "market", "trade")
    ),
    metadata = data.frame(unit_id = c("a1", "a2"))
  )
  for (method in c("log_likelihood", "chi_square", "fisher")) {
    first <- run_keyness(test_manifest("keyness", list(method = method, correction = "bh", top_n = 20)), inputs)
    second <- run_keyness(test_manifest("keyness", list(method = method, correction = "bh", top_n = 20)), inputs)
    expect_identical(first$results$features, second$results$features)
    expect_true(length(first$results$features) > 0)
    expect_true(all(vapply(first$results$features, function(row) !is.null(row$p_value_adjusted), logical(1))))
  }
})

test_that("co-occurrence association metrics are deterministic", {
  inputs <- list(
    units = data.frame(unit_id = c("u1", "u2")),
    tokens = data.frame(
      unit_id = c("u1", "u1", "u1", "u2", "u2", "u2"),
      token_position = c(0L, 1L, 2L, 0L, 1L, 2L),
      token = c("universal", "education", "policy", "universal", "education", "rights")
    ),
    metadata = data.frame(unit_id = c("u1", "u2"))
  )
  for (method in c("count", "pmi", "npmi", "dice", "log_dice", "t_score")) {
    first <- run_cooccurrence(
      test_manifest("cooccurrence", list(association_method = method, window_size = 2, top_n = 20)),
      inputs
    )
    second <- run_cooccurrence(
      test_manifest("cooccurrence", list(association_method = method, window_size = 2, top_n = 20)),
      inputs
    )
    expect_identical(first$results$pairs, second$results$pairs)
    expect_true(length(first$results$pairs) > 0)
  }
})

test_that("dispatcher rejects unsupported analysis", {
  expect_error(dispatch_analysis(test_manifest("topic_model"), test_inputs()), "Unsupported R analysis")
})
