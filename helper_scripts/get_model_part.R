library(lubridate)
library(stringr)
library(tidyverse)

get_model_part <- function(date = NULL, part = c("model", "weights", 
                          "predictions", "full image")) {
  
  # List all model folders, with and without paths
  model_folders_full <- list.files("/models/shoe_nn/TrainedModels",
                                    full.names = T)
  model_folders_base <- basename(model_folders_full)
  
  # Check for odd cases (likely two folders with same date/time)
  duplicates <- is.na(ymd_hms(model_folders_base, quiet = T)) %>%
    which()
  
  # Remove "_2" from duplicated folder
  model_folders_base[duplicates] %<>%  
    str_split(pattern = "_") %>%
    sapply(., "[[", 1)
  
  # Extract just days for easier parsing and selecting
  days <- model_folders_base %>%
    ymd_hms() %>%
    date()
  
  # If date is null, use most recent without warning
  if(is_null(date)) {
    date <- max(days)
  }
  
  # Format user-specified date and verify model was trained that day
  user_date <- ymd(date, quiet = T)
  
  if(length(user_date) < 1 || is.na(user_date)) {
    user_date <- max(days)
    warning(paste("Date must be provided in YMD format. Most recent date used:",
                  user_date))
  }
  
  if(!(user_date %in% days)) {
    user_date <- max(days)
    warning(paste("No model exists for selected date. Most recent date used:",
                  user_date))
  }
  
  # If multiple models were trained in a day, warn the user
  if(sum(grepl(days, pattern = user_date)) > 1) {
    warning("Note: Multiple folders exist for selected date. Most recent used.")
  }
  
  # Get most recent folder with given date and grab full folder name
  folder_path <- model_folders_full[which(days == user_date) %>% max()]
  
  # Return file path for selected model part
  part <- match.arg(part)
  
  pattern <- switch(part,
                    "weights" = "_256-weights",
                    "model" = "_256.h5",
                    "predictions" = "_256.Rdata",
                    "full image" = "fullimage")
  
  file <- list.files(folder_path, pattern = pattern)
  
  # If no matching file is found, return error
  if(length(file) < 1) {
    stop("Requested model part does not exist for specified date.")
  } else {
    return(file)
  }
}

# Test incorrect date format
get_model_part(part = "weights", date = "1-6-2019")

# Test no date
get_model_part(part = "weights")

# Test value that is not a date
get_model_part(part = "weights", date = 7)

# Test date that had no model
get_model_part(part = "weights", date = "2019-1-6")

# Test date with multiple folders
get_model_part(part = "model", date = "2018-11-18")

# Test part that isn't an option
get_model_part(part = "cow", date = "2018-11-26")

# Test part that doesn't exist for given date
get_model_part(part = "weights", date = "2018-9-24")
