#!/usr/bin/env Rscript

# Load required libraries
suppressPackageStartupMessages({
    library(ggplot2)
    library(stringr)
    library(plyr)
    library(dplyr)
    library(lubridate)
    library(reshape2)
    library(scales)
    library(ggthemes)
    library(Metrics)
})

# ----------------------------------------------------------------------
# Parse command line arguments (optional)
# ----------------------------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)
if (length(args) >= 1) {
    data_dir <- args[1]
} else {
    data_dir <- "."   # current directory
}
predictions_file <- file.path(data_dir, "r2plus1d_18_32_2_pretrained_test_predictions.csv")
size_file <- file.path(data_dir, "size.csv")
filelist_file <- file.path(data_dir, "FileList.csv")

# ----------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------
data <- read.csv(predictions_file, header = FALSE)
str(data)

dataNoAugmentation <- data[data$V2 == 0, ]
str(dataNoAugmentation)

dataGlobalAugmentation <- data %>% group_by(V1) %>% summarize(meanPrediction = mean(V3), sdPred = sd(V3))
str(dataGlobalAugmentation)

sizeData <- read.csv(size_file)
sizeData <- sizeData[sizeData$ComputerSmall == 1, ]
str(sizeData)

sizeRelevantFrames <- sizeData[c(1,2)]
sizeRelevantFrames$Frame <- sizeRelevantFrames$Frame - 32
sizeRelevantFrames[sizeRelevantFrames$Frame < 0, ]$Frame <- 0

beatByBeat <- merge(sizeRelevantFrames, data, by.x = c("Filename", "Frame"), by.y = c("V1", "V2"))
beatByBeat <- beatByBeat %>% group_by(Filename) %>% summarize(meanPrediction = mean(V3), sdPred = sd(V3))
str(beatByBeat)

# Load ground truth EF values
ActualNumbers <- read.csv(filelist_file)
ActualNumbers <- ActualNumbers[c(1,2)]
str(ActualNumbers)

# Merge and compute errors for no-augmentation case
dataNoAugmentation <- merge(dataNoAugmentation, ActualNumbers, by.x = "V1", by.y = "FileName", all.x = TRUE)
dataNoAugmentation$AbsErr <- abs(dataNoAugmentation$V3 - dataNoAugmentation$EF)
str(dataNoAugmentation)

cat("\n=== No augmentation ===\n")
cat("Mean absolute error:", mean(abs(dataNoAugmentation$V3 - dataNoAugmentation$EF)), "\n")
cat("RMSE:", rmse(dataNoAugmentation$V3, dataNoAugmentation$EF), "\n")
modelNoAugmentation <- lm(dataNoAugmentation$EF ~ dataNoAugmentation$V3)
cat("R-squared:", summary(modelNoAugmentation)$r.squared, "\n\n")

# Beat‑by‑beat analysis
beatByBeat <- merge(beatByBeat, ActualNumbers, by.x = "Filename", by.y = "FileName", all.x = TRUE)
cat("=== Beat‑by‑beat (mean per video) ===\n")
cat("Mean absolute error:", mean(abs(beatByBeat$meanPrediction - beatByBeat$EF)), "\n")
cat("RMSE:", rmse(beatByBeat$meanPrediction, beatByBeat$EF), "\n")
modelBeatByBeat <- lm(beatByBeat$EF ~ beatByBeat$meanPrediction)
cat("R-squared:", summary(modelBeatByBeat)$r.squared, "\n\n")

# Prepare for sampling simulation
beatByBeatAnalysis <- merge(sizeRelevantFrames, data, by.x = c("Filename", "Frame"), by.y = c("V1", "V2"))
str(beatByBeatAnalysis)

MAEdata <- data.frame(counter = 1:500)
MAEdata$sample <- -9999
MAEdata$error <- -9999

for (i in 1:500) {
    # Use slice_sample instead of deprecated sample_n
    samplingBeat <- beatByBeatAnalysis %>%
        group_by(Filename) %>%
        slice_sample(n = 1 + floor((i-1)/100), replace = TRUE) %>%
        group_by(Filename) %>%
        dplyr::summarize(meanPred = mean(V3))
    
    samplingBeat <- merge(samplingBeat, ActualNumbers, by.x = "Filename", by.y = "FileName", all.x = TRUE)
    samplingBeat$error <- abs(samplingBeat$meanPred - samplingBeat$EF)
    
    MAEdata$sample[i] <- 1 + floor((i-1)/100)
    MAEdata$error[i] <- mean(samplingBeat$error)
}

str(MAEdata)

# Plot boxplot
beatBoxPlot <- ggplot(data = MAEdata) +
    geom_boxplot(aes(x = sample, y = error, group = sample), outlier.shape = NA) +
    theme_classic() +
    theme(legend.position = "none", axis.text.y = element_text(size = 7)) +
    xlab("Number of Sampled Beats") +
    ylab("Mean Absolute Error") +
    scale_fill_brewer(palette = "Set1", direction = -1)

ggsave("beat_boxplot.pdf", beatBoxPlot, width = 6, height = 4)
print(beatBoxPlot)