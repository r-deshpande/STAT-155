# Project Proposal


## Narrative

I am investigating whether an AI detection model is more likely to flag
a certain style of writing. Through this project, my goal is to predict
whether a human-written sample and an AI-generated sample is likely to
be flagged based on the writing conventions present. I chose this
dataset because of the rampant use of AI and AI detection models
(TurnItIn, Grammarly) that tend to output false positives. The raw data
comes from Simon Couch’s detectors R package and the processed data
present in Project 1 comes from TidyTuesday - GPT Detectors
(07/18/2023).

## Data

``` r
rm(list=ls())
tuesdata <- tidytuesdayR::tt_load('2023-07-18')
```

    ---- Compiling #TidyTuesday Information for 2023-07-18 ----
    --- There is 1 file available ---


    ── Downloading files ───────────────────────────────────────────────────────────

      1 of 1: "detectors.csv"

``` r
tuesdata <- tidytuesdayR::tt_load(2023, week = 29)
```

    ---- Compiling #TidyTuesday Information for 2023-07-18 ----
    --- There is 1 file available ---


    ── Downloading files ───────────────────────────────────────────────────────────

      1 of 1: "detectors.csv"

``` r
detectors <- tuesdata$detectors

detectors <- readr::read_csv('https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2023/2023-07-18/detectors.csv')
```

    Rows: 6185 Columns: 9
    ── Column specification ────────────────────────────────────────────────────────
    Delimiter: ","
    chr (7): kind, .pred_class, detector, native, name, model, prompt
    dbl (2): .pred_AI, document_id

    ℹ Use `spec()` to retrieve the full column specification for this data.
    ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.

``` r
library(tibble)
glimpse(data)
```

    function (..., list = character(), package = NULL, lib.loc = NULL, verbose = getOption("verbose"), 
        envir = .GlobalEnv, overwrite = TRUE)  
