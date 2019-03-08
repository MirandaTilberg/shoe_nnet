library(lubridate)
library(stringr)
library(tidyverse)

#source("/home/tiltonm/shoe_nnet/helper_scripts/get_image_directory.R")

get_image_directory <- function(date = NULL, type = c("full", "train",
                                "validation", "test")) {
  
  # List all image folders, with and without paths
  image_folders_full <- list.files("/models/shoe_nn/RProcessedImages",
                                   pattern = "[[:digit:]]",
                                   full.names = T)
  image_folders_base <- basename(image_folders_full)
  
  # Extract just days for easier parsing and selecting
  days <- image_folders_base %>%
    ymd_hms() %>%
    date()

  # If date is null, use most recent without warning
  if(is_null(date)) {
    date <- max(days)
  }
  
  # Format user-specified date and verify images were processed that day
  user_date <- ymd(date, quiet = T)
  
  if(length(user_date) < 1 || is.na(user_date)) {
    user_date <- max(days)
    warning(paste("Date must be provided in YMD format. Most recent date used:",
                  user_date))
  }
  
  if(!(user_date %in% days)) {
    user_date <- max(days)
    warning(paste("No image folder for selected date. Most recent date used:",
                  user_date))
  }
  
  # If multiple models were trained in a day, warn the user
  if(sum(grepl(days, pattern = user_date)) > 1) {
    warning("Note: Multiple folders exist for selected date. Most recent used.")
  }
  
  folder_path <- image_folders_full[which(days == user_date) %>% max()]
  
  # Return file path for selected image directory
  type <- match.arg(type)
  
  if(type == "full") {
    return(folder_path)
  } else {
    return(file.path(folder_path, type))
  }
}

#get_image_directory(type = "full")
#get_image_directory(type = "test")
