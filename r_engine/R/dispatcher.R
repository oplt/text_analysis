dispatch_analysis <- function(manifest, inputs) switch(
  manifest$analysis$type,
  frequencies = run_frequencies(manifest, inputs),
  dfm = run_dfm(manifest, inputs),
  kwic = run_kwic(manifest, inputs),
  dictionary = run_dictionary(manifest, inputs),
  keyness = run_keyness(manifest, inputs),
  cooccurrence = run_cooccurrence(manifest, inputs),
  stop("Unsupported R analysis")
)
