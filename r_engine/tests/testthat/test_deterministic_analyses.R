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
  kwic <- run_kwic(test_manifest("kwic", list(keyword = "beta", window_size = 1, query_mode = "word")), inputs)
  expect_length(kwic$results$matches, 2L)
  expect_true(all(vapply(kwic$results$matches, function(row) row$keyword == "beta", logical(1))))
})

test_that("dispatcher rejects unsupported analysis", {
  expect_error(dispatch_analysis(test_manifest("topic_model"), test_inputs()), "Unsupported R analysis")
})
