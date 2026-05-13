"""Whole-brain voxel-wise GAMM for each OFC subregion, controlling for the
others.

For voxel v and subregion s, we model the seed-to-voxel FC as

    X ~ s(age, k) + gender + site + sum_{s' != s} mean_FC(s'),
    random = ~1 | Subject_ID,  corAR1 by measurement_order

so each subregion's developmental trajectory is unique variance, not shared
across subregions. Predictions are returned on a 0–36 month grid at 0.1 mo
resolution (361 points).
"""

import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri

pandas2ri.activate()

R_GAMM = r"""
library(mgcv); library(nlme)
timepoints  <- seq(0, 36, by = 0.1)
n_tp        <- length(timepoints)

run_voxel_gamm <- function(voxel_data, demo, sr1, sr2, sr3, sr4, sr5, sr6,
                            k, current) {
    out <- rep(NA_real_, 12 + n_tp)
    if (all(is.na(voxel_data)) || var(voxel_data, na.rm = TRUE) == 0) return(out)

    d <- demo
    d$X <- voxel_data
    d$gender <- as.factor(d$gender)
    d$site   <- as.factor(d$site)
    d$X_sr1 <- sr1; d$X_sr2 <- sr2; d$X_sr3 <- sr3
    d$X_sr4 <- sr4; d$X_sr5 <- sr5; d$X_sr6 <- sr6

    # All subregions except the current one enter as linear covariates
    ctrl <- paste(paste0("X_sr", (1:6)[-current]), collapse = " + ")

    # ---- Main model: smooth age + controls ------------------------------
    f_main <- as.formula(paste("X ~ s(age, k = k) + gender + site +", ctrl))
    m <- try(gamm(f_main, random = list(Subject_ID = ~1),
                  correlation = corAR1(form = ~ measurement_order | Subject_ID),
                  method = "REML", data = d), silent = TRUE)

    if (!inherits(m, "try-error")) {
        s <- summary(m$gam)
        out[1] <- s$s.table["s(age)", "F"]
        out[2] <- s$s.table["s(age)", "p-value"]
        out[3] <- BIC(m$lme)
        out[8] <- s$s.table["s(age)", "edf"]
        out[9] <- m$gam$df.residual

        nd <- data.frame(age = timepoints,
                         gender = levels(d$gender)[1],
                         site   = levels(d$site)[1])
        for (i in (1:6)[-current])
            nd[[paste0("X_sr", i)]] <- mean(d[[paste0("X_sr", i)]], na.rm = TRUE)
        out[(12 + 1):(12 + n_tp)] <- predict(m$gam, newdata = nd, type = "response")
    }

    # ---- Gender-by-age interaction model --------------------------------
    f_gen <- as.formula(paste("X ~ s(age, k = k, by = gender) + gender + site +", ctrl))
    g <- try(gamm(f_gen, random = list(Subject_ID = ~1),
                  correlation = corAR1(form = ~ measurement_order | Subject_ID),
                  method = "REML", data = d), silent = TRUE)

    if (!inherits(g, "try-error")) {
        sg <- summary(g$gam)
        for (term in rownames(sg$s.table)) {
            if (grepl(":gender0", term)) {
                out[4]  <- sg$s.table[term, "F"]
                out[5]  <- sg$s.table[term, "p-value"]
                out[10] <- sg$s.table[term, "edf"]
            } else if (grepl(":gender1", term)) {
                out[6]  <- sg$s.table[term, "F"]
                out[7]  <- sg$s.table[term, "p-value"]
                out[11] <- sg$s.table[term, "edf"]
            }
        }
        out[12] <- g$gam$df.residual
    }
    out
}
"""
robjects.r(R_GAMM)
_run = robjects.globalenv["run_voxel_gamm"]


def fit_voxel(voxel_ts, demo_r, subregion_means, k=3, current=1):
    """Fit GAMM for one voxel; returns [12 stats, 361 predictions].

    subregion_means : list of 6 R FloatVectors with each subregion's mean FC
                      across scans (length == n_scans).
    current         : 1..6, which subregion is the target (others = controls).
    """
    res = _run(robjects.FloatVector(voxel_ts), demo_r, *subregion_means,
               k, current)
    arr = np.asarray(res)
    return arr[:12], arr[12:]      # (stats, predictions over 0..36 mo)


def prepare_demo(demo_df):
    """Cast demographic columns to dtypes the R model expects."""
    d = demo_df.copy()
    d["age"]               = d["month"]
    d["gender"]            = d["gender"].astype(str).astype("category")
    d["site"]              = d["site"].astype(str).astype("category")
    d["Subject_ID"]        = d["subname"].astype("category")
    d["measurement_order"] = d.groupby("Subject_ID", observed=True).cumcount() + 1
    return pandas2ri.py2rpy(d)
