test_that("manifest validation rejects malformed and unsupported requests", {
  expect_error(validate_manifest(list()), "Invalid manifest")
  manifest <- test_manifest(); manifest$schema_version <- "2.0"
  expect_error(validate_manifest(manifest), "Unsupported manifest schema")
  expect_error(validate_manifest(test_manifest("topic_model")), "Unsupported R analysis")
  manifest <- test_manifest(); manifest$inputs <- list(units = "units.parquet")
  expect_error(validate_manifest(manifest), "Invalid manifest inputs")
})

test_that("manifest parsing and input reads reject missing files", {
  expect_error(read_manifest(file.path(tempdir(), "missing.json")), "Manifest file is missing")
  expect_error(read_inputs(test_manifest(), tempdir()), "Missing units input file")
  manifest <- test_manifest(); manifest$inputs$units <- "../units.parquet"
  expect_error(read_inputs(manifest, tempdir()), "Invalid units input path")
})
