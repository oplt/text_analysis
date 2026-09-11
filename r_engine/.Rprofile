# Activate the project library. Never install packages from interactive/runtime
# paths when RESEARCH_R_NO_INSTALL=1 (production worker / CI runtime).
source("renv/activate.R")
options(warn = 1)
if (identical(Sys.getenv("RESEARCH_R_NO_INSTALL"), "1")) {
  options(repos = character())
  options(renv.config.auto.snapshot = FALSE)
  options(renv.config.synchronized.check = FALSE)
  unlockBinding("install.packages", as.environment("package:utils"))
  utils_env <- as.environment("package:utils")
  assign(
    "install.packages",
    function(...) {
      stop(
        "R package installation is disabled at runtime; restore from renv.lock at image build",
        call. = FALSE
      )
    },
    envir = utils_env
  )
  lockBinding("install.packages", utils_env)
}
