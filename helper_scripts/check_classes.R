### This function takes an image directory as an argument and detects whether
##### "hexagon" and "pentagon" are considered separately or together as
##### "polygon". It returns the appropriate vector of classes for the directory.

library(magrittr)
check_classes <- function(dir) {
  str <- list.files(dir) %>% 
    substr(., 1, regexpr("-", .) - 1)
  polygon <- grepl("polygon", str) %>% sum > 0
  
  classes = c("bowtie", "chevron", "circle", "line",
              "quad", "star", "text", "triangle")
  
  if (polygon) {
    classes <- c(classes, "polygon") %>% sort
  } else {
    classes <- c(classes, "hexagon", "pentagon") %>% sort
  }
  
  return(classes)
}

## test
 check_classes("/models/shoe_nn/RProcessedImages/20180921-110941/test/")
 check_classes("/models/shoe_nn/RProcessedImages/20181023-162730/test/")





