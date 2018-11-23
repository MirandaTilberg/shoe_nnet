get_most_recent <- function(type = "predictions", verbose = F) {
  
  # If type is supported, the matching pattern is returned, else NULL
  pattern <- switch(type,
                    "weights" = "256-weights",
                    "model" = "256.h5",
                    "predictions" = "256.Rdata",
                    "full image" = "fullimage")
  
  if (is.null(pattern)) {
    warning(paste("Selected file type is not supported by this function.", 
                  "Returning most recent predictions."))
    pattern <- "256.Rdata"
  }
  
  files <- list.files("/models/shoe_nn/TrainedModels", 
                  recursive = T, full.names = T)
  
  file_id <- grepl(pattern, files) %>% which %>% max
  
  if(verbose) {
    model_time <- substr(basename(files[file_id]), 1, 19) %>% 
      gsub("_", " at ", .) %>%
      paste0(., "\n")
    cat("Using model trained on", model_time)
  }
  
  return(files[file_id])
}
