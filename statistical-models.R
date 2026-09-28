library(readxl)
library(dplyr)
library(lme4)
library(lmerTest)
library(performance)
library(irr)
library(psych)
library(ggplot2)
library(extrafont)
library(tidyr)
library(DescTools)

loadfonts(device = "win")
windowsFonts(Times = windowsFont("Times New Roman"))
# Use Times New Roman for all ggplot2 figures
BASE_FONT_FAMILY <- "Times"
theme_set(theme_minimal(base_size = 16, base_family = BASE_FONT_FAMILY))


#######don't forget to change formula!!!!!!!!!!!
landmark_name = "hip"
BAsavepath = "./plot-BA/B-A-hip-TNR.png"
Scatpath = "./plot-sc/Linear-hip-TNR.png"
df <- read.csv("./pair_data/extracted_measurement/measurements_1.csv")

df$id <- as.factor(df$id)
df$scanner <- as.factor(df$scanner)
df$stage <- as.factor(df$stage)

# -----------------------------
# 1. Linear Mixed Model
# -----------------------------
model <- lmer(measurement ~ scanner + stage + (1 | id), data = df) # with stage when stage number is 2 or more

cat("\n--- LMM Summary ---\n")
print(summary(model))

# -----------------------------
# 2. ICC
# -----------------------------
df_wide <- df %>%
  group_by(id, scanner) %>%  # compare two scanners on all data
  summarise(measurement = mean(measurement), .groups = "drop") %>%
  pivot_wider(names_from = scanner, values_from = measurement) %>%
  drop_na()

icc_result <- icc(
  df_wide[, c("f3", "poly")],
  model = "twoway",
  type = "agreement",
  unit = "single"
)

cat("\n--- ICC---\n")
print(icc_result)

result <- df %>%
  dplyr::group_by(scanner) %>%
  dplyr::summarise(
    mean = mean(measurement),
    var = var(measurement),
    sd = sd(measurement)
  )

print(result)

# -----------------------------
# 3. Bland–Altman
# -----------------------------
df_wide <- df_wide %>%
  mutate(
    mean_val = (f3 + poly) / 2,
    diff_val = poly - f3
  )

# Remove missing paired observations
df_ba <- df_wide %>%
  filter(!is.na(f3), !is.na(poly))

# Basic Bland-Altman statistics
n <- nrow(df_ba)
bias <- mean(df_ba$diff_val)
sd_diff <- sd(df_ba$diff_val)

loa_upper <- bias + 1.96 * sd_diff
loa_lower <- bias - 1.96 * sd_diff

# Critical t-value
t_crit <- qt(0.975, df = n - 1)

# 95% CI for bias
se_bias <- sd_diff / sqrt(n)

bias_ci_lower <- bias - t_crit * se_bias
bias_ci_upper <- bias + t_crit * se_bias

# 95% CI for limits of agreement
se_loa <- sd_diff * sqrt(
  1 / n + (1.96^2) / (2 * (n - 1))
)

loa_upper_ci_lower <- loa_upper - t_crit * se_loa
loa_upper_ci_upper <- loa_upper + t_crit * se_loa

loa_lower_ci_lower <- loa_lower - t_crit * se_loa
loa_lower_ci_upper <- loa_lower + t_crit * se_loa


# Print results
cat("\n--- Bland-Altman ---\n")
cat("N:", n, "\n")

cat("Bias:", round(bias, 2),
    "95% CI:", round(bias_ci_lower, 2),
    "to", round(bias_ci_upper, 2), "\n")

cat("Upper LoA:", round(loa_upper, 2),
    "95% CI:", round(loa_upper_ci_lower, 2),
    "to", round(loa_upper_ci_upper, 2), "\n")

cat("Lower LoA:", round(loa_lower, 2),
    "95% CI:", round(loa_lower_ci_lower, 2),
    "to", round(loa_lower_ci_upper, 2), "\n")


# Plot
p <- ggplot(df_ba, aes(x = mean_val, y = diff_val)) +
  
  geom_point() +
  
  # 95% CI bands
  annotate(
    "rect",
    xmin = -Inf, xmax = Inf,
    ymin = bias_ci_lower, ymax = bias_ci_upper,
    alpha = 0.15
  ) +
  
  annotate(
    "rect",
    xmin = -Inf, xmax = Inf,
    ymin = loa_upper_ci_lower, ymax = loa_upper_ci_upper,
    alpha = 0.15
  ) +
  
  annotate(
    "rect",
    xmin = -Inf, xmax = Inf,
    ymin = loa_lower_ci_lower, ymax = loa_lower_ci_upper,
    alpha = 0.15
  ) +
  
  # Main lines
  geom_hline(yintercept = bias, linetype = "dashed") +
  geom_hline(yintercept = loa_upper, linetype = "dashed") +
  geom_hline(yintercept = loa_lower, linetype = "dashed") +
  
  coord_cartesian(ylim = c(-80, 60)) +
  
  labs(
    title = paste("Bland-Altman:", landmark_name),
    x = "Mean",
    y = "Difference (PolyCam - Fit3D)"
  )

print(p)
ggsave(BAsavepath, dpi=600, dev='png', height=4.5, width=6.5, units="in")

# -----------------------------
# 4. CV
# -----------------------------
cv_results <- df %>%
  group_by(scanner) %>%
  summarise(
    mean = mean(measurement),
    sd = sd(measurement),
    CV_percent = (sd / mean) * 100
  )

cat("\n--- CV ---\n")
print(cv_results)

# -----------------------------
# 4. RMSE & MAE
# -----------------------------
# to wide format（as ICC）
df_wide <- df %>%
  group_by(id, stage, scanner) %>%
  summarise(measurement = mean(measurement), .groups = "drop") %>%
  pivot_wider(names_from = scanner, values_from = measurement) %>%
  drop_na()

# error
errors <- df_wide$poly - df_wide$f3

# RMSE
rmse <- sqrt(mean(errors^2))

# MAE
mae <- mean(abs(errors))

cat("\n--- Error Metrics (PolyCam vs Fit3D) ---\n")
cat("RMSE:", rmse, "\n")
cat("MAE:", mae, "\n")

wilcox_test <- wilcox.test(
  df_wide$poly,
  df_wide$f3,
  paired = TRUE
)

print(wilcox_test)

# -----------------------------
# 7. Pearson Correlation
# -----------------------------

# to paired wide format
df_wide <- df %>%
  group_by(id, stage, scanner) %>%
  summarise(measurement = mean(measurement), .groups = "drop") %>%
  pivot_wider(names_from = scanner, values_from = measurement) %>%
  drop_na()

# Pearson correlation test
cor_test <- cor.test(
  df_wide$poly,
  df_wide$f3,
  method = "pearson"
)

print(cor_test)

# extract results
cat("\n--- Pearson Correlation ---\n")
cat("Correlation (r):", cor_test$estimate, "\n")
cat("p-value:", cor_test$p.value, "\n")

# interpretation
if (cor_test$p.value < 0.05) {
  cat("Conclusion: Significant linear correlation between PolyCam and Fit3D.\n")
} else {
  cat("Conclusion: No significant linear correlation.\n")
}

# -----------------------------
# 7.1. Lin's concordance correlation coefficient (CCC)
# -----------------------------

df_wide <- df %>%
  group_by(id, stage, scanner) %>%
  summarise(
    measurement = mean(measurement),
    .groups = "drop"
  ) %>%
  pivot_wider(
    names_from = scanner,
    values_from = measurement
  ) %>%
  drop_na()

# Lin's Concordance Correlation Coefficient (CCC)

ccc_result <- CCC(
  df_wide$poly,
  df_wide$f3,
  ci = "z-transform",
  conf.level = 0.95
)

print(ccc_result)

# Extract and report results
cat("\n--- Lin's Concordance Correlation Coefficient ---\n")
cat("CCC:", round(ccc_result$rho.c$est, 3), "\n")
cat(
  "95% CI:",
  round(ccc_result$rho.c$lwr.ci, 3),
  "to",
  round(ccc_result$rho.c$upr.ci, 3),
  "\n"
)

# Remove missing values
df_reg <- df_wide %>%
  filter(!is.na(f3), !is.na(poly))

# Linear regression
lm_model <- lm(poly ~ f3, data = df_reg)

# Extract regression coefficients
intercept <- coef(lm_model)[1]
slope <- coef(lm_model)[2]

# Standard Error of Estimate (SEE)
# SEE = sqrt(SSE / (n - 2))
SEE <- sqrt(sum(residuals(lm_model)^2) / df.residual(lm_model))

# R-squared
r_squared <- summary(lm_model)$r.squared

# Regression equation
equation <- paste0(
  "y = ", round(slope, 3), "x ",
  ifelse(intercept >= 0, "+ ", "- "),
  round(abs(intercept), 3)
)

# Print results
cat("\n--- Linear Regression ---\n")
cat("Equation:", equation, "\n")
cat("SEE:", round(SEE, 3), "\n")
cat("R-squared:", round(r_squared, 3), "\n")


# Plot
p2 <- ggplot(df_reg, aes(x = f3, y = poly)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  
  annotate(
    "text",
    x = Inf, y = -Inf,
    label = paste0(
      equation,
      "\nSEE = ", round(SEE, 2)
    ),
    hjust = 1.1,
    vjust = -0.5,
    size = 4
  ) +
  
  labs(
    title = "PolyCam vs Fit3D on xxx Circumference Measurement",
    x = "Fit3D measurement",
    y = "PolyCam measurement"
  )

print(p2)
ggsave(Scatpath, dpi=600, dev='png', height=4.5, width=6.5, units="in")
