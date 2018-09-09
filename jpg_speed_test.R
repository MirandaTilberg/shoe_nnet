library(imager)
library(jpeg)


spdtst <- function(imgr = T){
  a <- list.files("~/shoe_nnet/shoes/circtriquad/train/circle", full.names = T)
  
  if (imgr) {
    for (i in 1:length(a)) {
      b <- load.image(a[i])
    }
  } else {
    for (i in 1:length(a)) {
      b <- readJPEG(a[i])
    }
  }
}

system.time(spdtst())
system.time(spdtst(imgr=F))

# b <- load.image(a[1])
# d <- readJPEG(a[1])
