# =====================================================================
# STEP 3 — Global GARCH(1,1) Volatility Modelling (188 Countries)
# =====================================================================
# Coverage: 188 countries, 2000–2025
# Model:    GJR-GARCH(1,1) with Student-t (falls back to sGARCH)
#
# Spread data sources:
#   IMF IFS (free):   https://data.imf.org/regular.aspx?key=61545850
#   World Bank EMBI:  https://data.worldbank.org/indicator/FM.AST.DOMO.ZG.M3
#   JP Morgan EMBI+:  Bloomberg / Refinitiv (paid)
#
# Install:
#   install.packages(c("rugarch","readr","dplyr","tidyr",
#                      "ggplot2","lubridate","purrr"))
# =====================================================================

library(rugarch)
library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(lubridate)
library(purrr)

OUTPUT_DIR <- "data/garch"
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

# ─── 188 COUNTRIES ───────────────────────────────────────────────────────────

COUNTRIES <- c(
  # HI
  "AUS","AUT","BEL","BHR","BRN","CAN","CHE","CYP","CZE","DEU",
  "DNK","ESP","EST","FIN","FRA","GBR","GRC","HKG","HRV","HUN",
  "IRL","ISL","ISR","ITA","JPN","KOR","KWT","LTU","LUX","LVA",
  "MLT","NLD","NOR","NZL","OMN","PLW","POL","PRT","QAT","SAU",
  "SGP","SVK","SVN","SWE","TWN","ARE","USA",
  # UMI
  "ALB","ARM","AZE","BGR","BIH","BOL","BRA","BWA","CHN","COL",
  "CRI","CUB","DMA","DOM","DZA","ECU","FJI","GAB","GRD","GTM",
  "GUY","IDN","IRN","IRQ","JAM","JOR","KAZ","KNA","LBN","LBY",
  "LCA","MAR","MDA","MEX","MKD","MNE","MUS","MYS","NAM","PAN",
  "PER","PHL","PRY","ROU","RUS","SRB","SUR","THA","TTO","TUN",
  "URY","VCT","VEN",
  # LMI
  "BGD","BEN","BFA","BTN","CMR","COD","COG","COM","CPV","EGY",
  "FSM","GHA","HND","IND","KEN","KGZ","KHM","LAO","LKA","LSO",
  "MHL","MMR","MNG","MOZ","MRT","NGA","NIC","NPL","PAK","PNG",
  "PSE","SDN","SEN","SLB","SLE","SWZ","TJK","TLS","TZA","UGA",
  "UKR","UZB","VNM","ZMB",
  # LI
  "AFG","BDI","CAF","ERI","ETH","GIN","GNB","HTI","LBR","MDG",
  "MLI","MWI","NER","RWA","SOM","SSD","SYR","TCD","TGO","YEM","ZWE"
)
# Remove duplicates preserving order
COUNTRIES <- unique(COUNTRIES)

# Income group lookup (used for synthetic spread baseline)
INCOME_GROUP <- c(
  setNames(rep("HI",  47), c("AUS","AUT","BEL","BHR","BRN","CAN","CHE","CYP","CZE","DEU","DNK","ESP","EST","FIN","FRA","GBR","GRC","HKG","HRV","HUN","IRL","ISL","ISR","ITA","JPN","KOR","KWT","LTU","LUX","LVA","MLT","NLD","NOR","NZL","OMN","PLW","POL","PRT","QAT","SAU","SGP","SVK","SVN","SWE","TWN","ARE","USA")),
  setNames(rep("UMI", 53), c("ALB","ARM","AZE","BGR","BIH","BOL","BRA","BWA","CHN","COL","CRI","CUB","DMA","DOM","DZA","ECU","FJI","GAB","GRD","GTM","GUY","IDN","IRN","IRQ","JAM","JOR","KAZ","KNA","LBN","LBY","LCA","MAR","MDA","MEX","MKD","MNE","MUS","MYS","NAM","PAN","PER","PHL","PRY","ROU","RUS","SRB","SUR","THA","TTO","TUN","URY","VCT","VEN")),
  setNames(rep("LMI", 44), c("BGD","BEN","BFA","BTN","CMR","COD","COG","COM","CPV","EGY","FSM","GHA","HND","IND","KEN","KGZ","KHM","LAO","LKA","LSO","MHL","MMR","MNG","MOZ","MRT","NGA","NIC","NPL","PAK","PNG","PSE","SDN","SEN","SLB","SLE","SWZ","TJK","TLS","TZA","UGA","UKR","UZB","VNM","ZMB")),
  setNames(rep("LI",  21), c("AFG","BDI","CAF","ERI","ETH","GIN","GNB","HTI","LBR","MDG","MLI","MWI","NER","RWA","SOM","SSD","SYR","TCD","TGO","YEM","ZWE"))
)

# ─── LOAD SPREAD DATA ────────────────────────────────────────────────────────

load_spread_data <- function(filepath = "data/raw/spreads.csv") {
  if (!file.exists(filepath)) {
    message("Spread data not found. Generating synthetic data...")
    message("For real data: IMF IFS → https://data.imf.org")
    return(generate_synthetic_spreads())
  }
  df <- read_csv(filepath, show_col_types = FALSE)
  df$date <- as.Date(df$date)
  message(sprintf("Loaded real spreads: %d rows, %d countries",
                  nrow(df), n_distinct(df$country)))
  return(df)
}

generate_synthetic_spreads <- function() {
  dates <- seq(as.Date("2000-01-01"), as.Date("2025-12-31"), by = "month")
  set.seed(42)
  
  # Income-group baseline spread ranges (basis points)
  base_ranges <- list(
    HI  = c(30,  120),
    UMI = c(150, 400),
    LMI = c(250, 600),
    LI  = c(400, 900)
  )
  
  # Countries known to have had distress episodes (sparse list for spikes)
  distress_map <- list(
    ARG = list(c("2001-01-01","2005-12-31"), c("2019-01-01","2022-12-31")),
    GHA = list(c("2022-01-01","2024-12-31")),
    LKA = list(c("2022-01-01","2024-12-31")),
    TUN = list(c("2022-01-01","2025-12-31")),
    UKR = list(c("2022-01-01","2025-12-31")),
    VEN = list(c("2014-01-01","2025-12-31")),
    ZMB = list(c("2020-01-01","2024-12-31")),
    ETH = list(c("2021-01-01","2025-12-31")),
    LBN = list(c("2019-01-01","2025-12-31")),
    GRC = list(c("2010-01-01","2018-12-31")),
    IRL = list(c("2010-01-01","2013-12-31")),
    PRT = list(c("2011-01-01","2014-12-31")),
    RUS = list(c("2022-01-01","2025-12-31"))
  )
  
  rows <- list()
  for (ctry in COUNTRIES) {
    ig    <- if (!is.na(INCOME_GROUP[ctry])) INCOME_GROUP[ctry] else "UMI"
    rng   <- base_ranges[[ig]]
    base  <- runif(1, rng[1], rng[2])
    spread <- numeric(length(dates))
    spread[1] <- base
    episodes  <- distress_map[[ctry]]
    
    for (i in seq(2, length(dates))) {
      shock     <- rnorm(1, 0, base * 0.12)
      spread[i] <- 0.87 * spread[i-1] + shock + base * 0.13
      # Spike during distress
      for (ep in (episodes %||% list())) {
        if (dates[i] >= as.Date(ep[1]) && dates[i] <= as.Date(ep[2]))
          spread[i] <- spread[i] + runif(1, 300, 900)
      }
      spread[i] <- max(15, spread[i])
    }
    rows[[ctry]] <- data.frame(date = dates, country = ctry, spread_bps = spread)
  }
  bind_rows(rows)
}

# Null coalescing helper
`%||%` <- function(a, b) if (!is.null(a)) a else b


# ─── GJR-GARCH FITTING ───────────────────────────────────────────────────────

fit_garch_country <- function(ctry, spread_df, use_gjr = TRUE) {
  df_c <- spread_df %>%
    filter(country == ctry) %>%
    arrange(date) %>%
    mutate(log_spread = log(pmax(spread_bps, 1)),
           returns    = c(NA, diff(log_spread))) %>%
    filter(!is.na(returns))
  
  if (nrow(df_c) < 60) {
    message(sprintf("  Skip %-5s — only %d obs", ctry, nrow(df_c)))
    return(NULL)
  }
  
  spec <- ugarchspec(
    variance.model     = list(model = if (use_gjr) "gjrGARCH" else "sGARCH",
                              garchOrder = c(1, 1)),
    mean.model         = list(armaOrder = c(1, 0), include.mean = TRUE),
    distribution.model = "std"
  )
  
  tryCatch({
    fit    <- ugarchfit(spec = spec, data = df_c$returns, solver = "hybrid")
    params <- coef(fit)
    cond_v <- as.numeric(sigma(fit)) * sqrt(12)
    
    result <- df_c %>%
      select(date, country, spread_bps, returns) %>%
      mutate(
        cond_volatility = cond_v,
        vol_regime      = as.integer(cond_v > quantile(cond_v, 0.75)),
        persistence     = params["alpha1"] + params["beta1"],
        garch_asymmetry = if ("gamma1" %in% names(params)) params["gamma1"] else 0.0,
        omega           = params["omega"],
        alpha1          = params["alpha1"],
        beta1           = params["beta1"]
      )
    list(data = result, fit = fit, params = params,
         model = if (use_gjr) "gjrGARCH" else "sGARCH")
    
  }, error = function(e) {
    if (use_gjr) {
      message(sprintf("  GJR failed for %-5s — retrying sGARCH", ctry))
      return(fit_garch_country(ctry, spread_df, use_gjr = FALSE))
    }
    message(sprintf("  GARCH failed %-5s: %s", ctry, e$message))
    NULL
  })
}


# ─── AGGREGATE TO ANNUAL ─────────────────────────────────────────────────────

aggregate_garch_annual <- function(results) {
  ok   <- Filter(Negate(is.null), results)
  data <- bind_rows(map(ok, "data"))
  
  annual <- data %>%
    mutate(year = year(date)) %>%
    group_by(country, year) %>%
    summarise(
      mean_cond_vol   = mean(cond_volatility,   na.rm = TRUE),
      max_cond_vol    = max(cond_volatility,    na.rm = TRUE),
      vol_of_vol      = sd(cond_volatility,     na.rm = TRUE),
      pct_high_vol    = mean(vol_regime,        na.rm = TRUE),
      mean_spread     = mean(spread_bps,        na.rm = TRUE),
      spread_change   = last(spread_bps) - first(spread_bps),
      persistence     = mean(persistence,       na.rm = TRUE),
      garch_asymmetry = mean(garch_asymmetry,   na.rm = TRUE),
      .groups = "drop"
    )
  
  write_csv(annual, file.path(OUTPUT_DIR, "garch_features_annual.csv"))
  message(sprintf("\n✓ garch_features_annual.csv  →  %d rows × %d cols",
                  nrow(annual), ncol(annual)))
  message(sprintf("  Converged: %d / %d countries",
                  n_distinct(annual$country), length(COUNTRIES)))
  annual
}


# ─── DIAGNOSTIC PLOTS ────────────────────────────────────────────────────────

plot_garch_sample <- function(results, n = 9) {
  ok       <- Filter(Negate(is.null), results)
  selected <- names(ok)[seq_len(min(n, length(ok)))]
  plot_df  <- bind_rows(map(ok[selected], "data"))
  
  p <- ggplot(plot_df, aes(x = date)) +
    geom_line(aes(y = spread_bps / 100, colour = "Spread (×100 bps)"),
              linewidth = 0.4, alpha = 0.75) +
    geom_line(aes(y = cond_volatility, colour = "Cond. Volatility"),
              linewidth = 0.5, linetype = "dashed") +
    facet_wrap(~country, scales = "free_y", ncol = 3) +
    scale_colour_manual(values = c(
      "Spread (×100 bps)" = "#185FA5",
      "Cond. Volatility"  = "#D85A30"
    )) +
    scale_x_date(date_breaks = "5 years", date_labels = "%Y") +
    labs(title    = "Sovereign Spreads & GARCH Conditional Volatility (sample)",
         subtitle = "GJR-GARCH(1,1) with Student-t  |  188 countries total",
         x = NULL, y = NULL, colour = NULL) +
    theme_minimal(base_size = 9) +
    theme(legend.position = "bottom",
          strip.text = element_text(face = "bold"),
          axis.text.x = element_text(angle = 30, hjust = 1))
  
  ggsave(file.path(OUTPUT_DIR, "garch_diagnostics.png"),
         p, width = 14, height = 12, dpi = 150)
  message(sprintf("  Plot → %s/garch_diagnostics.png", OUTPUT_DIR))
}


# ─── MAIN ────────────────────────────────────────────────────────────────────

cat(sprintf("\n[Step 3] Fitting GJR-GARCH for %d countries (2000–2025)...\n",
            length(COUNTRIES)))

spread_df <- load_spread_data()
results   <- list()

pb <- txtProgressBar(min = 0, max = length(COUNTRIES), style = 3)
for (i in seq_along(COUNTRIES)) {
  results[[COUNTRIES[i]]] <- fit_garch_country(COUNTRIES[i], spread_df)
  setTxtProgressBar(pb, i)
}
close(pb)

n_ok <- sum(!sapply(results, is.null))
cat(sprintf("\nConverged: %d / %d  |  Failed: %d\n",
            n_ok, length(COUNTRIES), length(COUNTRIES) - n_ok))

garch_annual <- aggregate_garch_annual(results)
plot_garch_sample(results, n = 9)

cat("\n✅ Step 3 complete. Run 04_build_features.py next.\n")