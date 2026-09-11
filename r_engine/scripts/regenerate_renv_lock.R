#!/usr/bin/env Rscript
# Regenerate a full scientific renv.lock (transitive graph) for R 4.4.0.
# Do not hand-edit renv.lock. Always regenerate with this script.

args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
script_path <- if (length(file_arg)) {
  normalizePath(sub("^--file=", "", file_arg[[1]]), mustWork = TRUE)
} else {
  normalizePath("regenerate_renv_lock.R", mustWork = FALSE)
}
project <- normalizePath(file.path(dirname(script_path), ".."), mustWork = TRUE)
setwd(project)

message("Regenerating renv.lock in ", project)

options(repos = c(CRAN = "https://cloud.r-project.org"))
Sys.setenv(LIBARROW_BINARY = "true")

if (!requireNamespace("renv", quietly = TRUE)) {
  install.packages("renv", repos = "https://cloud.r-project.org")
}

renv::init(project = project, bare = TRUE, restart = FALSE, force = TRUE)

# Prefer historical pins when available; otherwise CRAN resolves and lock captures them.
pkgs <- c(
  "quanteda@4.1.0",
  "quanteda.textstats@0.97.2",
  "arrow@18.1.0",
  "jsonlite@1.8.9",
  "testthat@3.2.3"
)
ok <- tryCatch({
  renv::install(pkgs, prompt = FALSE)
  TRUE
}, error = function(e) {
  message("Pinned install failed (", conditionMessage(e), "); falling back to unpinned CRAN")
  FALSE
})
if (!isTRUE(ok)) {
  renv::install(
    c("quanteda", "quanteda.textstats", "arrow", "jsonlite", "testthat"),
    prompt = FALSE
  )
}

renv::snapshot(project = project, type = "all", prompt = FALSE)

status <- renv::status(project = project)
message("renv::status() after snapshot:")
print(status)

lock <- jsonlite::fromJSON(file.path(project, "renv.lock"), simplifyVector = FALSE)
n_pkgs <- length(lock$Packages)
message("Lockfile packages: ", n_pkgs)
if (n_pkgs < 20L) {
  stop("Lockfile looks incomplete (", n_pkgs, " packages); expected a full transitive graph")
}
required <- c("quanteda", "quanteda.textstats", "arrow", "jsonlite", "testthat")
missing <- setdiff(required, names(lock$Packages))
if (length(missing)) {
  stop("Lockfile missing required packages: ", paste(missing, collapse = ", "))
}
message("OK: wrote ", file.path(project, "renv.lock"))
