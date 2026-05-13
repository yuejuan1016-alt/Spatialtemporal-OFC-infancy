"""Vertex/voxel-wise GAMM normative model for FC trajectories (0-36 months).

Model:   X ~ s(age, k) + gender + site, random = ~1 | Subject_ID,
         AR(1) correlation by measurement_order within Subject_ID.

For each voxel/edge we fit the GAMM via R's mgcv::gamm, then predict on a
0..36 month grid, marginalising over gender x site by their empirical weights
in the sample. Returns the fitted curve, its SE/CIs, and the first derivative.
"""

import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri

pandas2ri.activate()

R_GAMM = r"""
library(mgcv); library(nlme); library(gratia)

run_voxel_gamm <- function(voxel_data, demo, k) {
    data <- demo
    data$X <- voxel_data
    data$gender <- as.factor(data$gender)
    data$site   <- as.factor(data$site)

    n_age <- 37
    out <- list(fit = rep(NA_real_, n_age), se = rep(NA_real_, n_age),
                ci_lo = rep(NA_real_, n_age), ci_hi = rep(NA_real_, n_age),
                deriv = rep(NA_real_, n_age),
                p = NA_real_, r2 = NA_real_, edf = NA_real_)

    if (all(is.na(voxel_data)) || var(voxel_data, na.rm=TRUE) == 0) return(out)

    m <- try(gamm(X ~ s(age, k=k) + gender + site,
                  random = list(Subject_ID = ~1),
                  correlation = corAR1(form = ~ measurement_order | Subject_ID),
                  method = "REML", data = data), silent = TRUE)
    if (inherits(m, "try-error")) return(out)

    # Marginalise over gender x site with empirical weights
    grid <- expand.grid(age = 0:36,
                        gender = levels(data$gender),
                        site   = levels(data$site),
                        Subject_ID = NA, measurement_order = 1)
    w <- as.data.frame(as.table(table(data$gender, data$site) / nrow(data)))
    names(w) <- c("gender", "site", "weight")
    grid <- merge(grid, w, by = c("gender", "site"))

    p <- predict(m$gam, newdata = grid, type = "response", se.fit = TRUE)
    fit_mat <- matrix(p$fit    * grid$weight, nrow = n_age)
    se_mat  <- matrix(p$se.fit * grid$weight, nrow = n_age)
    out$fit   <- rowSums(fit_mat)
    out$se    <- sqrt(rowSums(se_mat^2))
    out$ci_lo <- out$fit - 1.96 * out$se
    out$ci_hi <- out$fit + 1.96 * out$se

    d <- derivatives(m$gam, term = "s(age)", data = grid)
    out$deriv <- rowSums(matrix(d$derivative * grid$weight, nrow = n_age))

    s <- summary(m$gam)
    out$p   <- s$s.table[4]
    out$r2  <- s$r.sq
    out$edf <- s$edf
    out
}
"""
robjects.r(R_GAMM)
_run = robjects.globalenv["run_voxel_gamm"]


def fit_voxel(voxel_ts, demo_r, k=3):
    """Fit GAMM for one voxel/edge timeseries; return dict of curves & stats."""
    res = _run(robjects.FloatVector(voxel_ts), demo_r, k)
    keys = ["fit", "se", "ci_lo", "ci_hi", "deriv", "p", "r2", "edf"]
    return {k_: np.asarray(res.rx2(k_)) for k_ in keys}


def prepare_demo(demo_df):
    """Cast demographic columns to the dtypes the R model expects."""
    d = demo_df.copy()
    d["age"]               = d["month"]
    d["gender"]            = d["gender"].astype(str).astype("category")
    d["site"]              = d["site"].astype(str).astype("category")
    d["Subject_ID"]        = d["subname"].astype("category")
    d["measurement_order"] = d.groupby("Subject_ID", observed=True).cumcount() + 1
    return pandas2ri.py2rpy(d)
