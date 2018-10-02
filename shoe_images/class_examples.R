classes <- c("bowtie", "chevron", "circle", "hexagon", "line",
  "pentagon", "quad", "star", "text", "triangle")

image_dir <- "/home/tiltonm/shoe_nnet/shoes/onehot/train/"
#image_dir <- "/models/shoe_nn/RProcessedImages/20180930-211352/train/"
files <- list.files(image_dir, full.names = T)

get_labs <- function(directory, sample_count, verbose = F) {
  
  labels <- array(0, dim = c(sample_count, length(classes)))
  files <- list.files(directory)
  
  for (i in 1:sample_count) {
    if (verbose) cat(paste(i, ", ", sep=""))
    
    fname <- files[i]
    str <- substr(fname, 1, regexpr("-",fname)-1)
    for (j in 1:length(classes)) {
      labels[i, j] <- grepl(classes[j], str)
    }
  }
labels
}

labels <- get_labs(image_dir, length(list.files(image_dir)))
#labels %>% apply(., 2, sum)

n_ex_per_class <- 3
choice <- matrix(0, nrow = length(classes), ncol = n_ex_per_class)
rownames(choice) <- classes
choice[1,] <- c(29, 422, 441) #bowtie
choice[2,] <- c(815,1344,945) #chevron
choice[3,] <- c(2513,2539,2537) #circle
choice[4,] <- c(3581,3530,3584) #hexagon
choice[5,] <- c(4193,3836,4217) #line
choice[6,] <- c(6097,6103,9159) #pentagon
choice[7,] <- c(7113,8092,6469) #quad, c(6932, 8069, 7164)
choice[8,] <- c(9059,2091,9217) #star, 8975
choice[9,] <- c(9686,10493,9545) #text, 10537, 10560, 9760, 9386
choice[10,] <- c(11258,11216,10996) #triangle, 4804, 11003, 11204

# This code randomly chooses the rows of "choice"
# for (i in 1:(length(classes)+2)) {
#   x <- which(labels[,i] == 1)
#   choice[i,] <- sample(x, 3)
# }

library(imager)
#par(mfrow = c(length(classes), n_ex_per_class))

# # layout(matrix(1:60, byrow = T, ncol = 3))
# 
# par(mfrow = c(length(choice)/6,6), mar = rep(.1, 4))
# for (i in 1:length(choice)) {
#   img <- imager::load.image(files[choice[i]])
#   plot(img, axes = F)
# }

blank <- load.image("~/shoe_nnet/shoe_images/blank.png")

par(mfrow = c(5,2))
for (i in 1:length(classes)) {
  examples <- choice[i,]
  examples.list <- vector("list", length = n_ex_per_class)
  for (j in 1:length(examples.list)) {
    examples.list[[j]] <- load.image(files[examples[j]])
  }
  examples.list <- as.imlist(examples.list)
  combo <- imappend(examples.list, axis = "x")
  plot(combo, axes = F)
  
}


par(mfrow = c(1,1))

for (i in 1:1){#length(classes)) {
  examples <- choice[i,]
  examples.list <- vector("list", length = n_ex_per_class)
  
  for (j in 1:length(examples.list)) {
    examples.list[[j]] <- load.image(files[examples[j]])
  }
  
  examples.list <- as.imlist(examples.list)
 
  combo <- imappend(examples.list, axis = "x")
  
  plot(combo, axes = F)
  #imager::save.image(im = combo, file = paste0("~/shoe_nnet/shoe_images/class_examples/", classes[i], "_examples.png"))
  
}



# dim(examples.list[[1]])
# 
# mar <- array(0, dim = c(256, 20, 1, 3))
# 
# examples.list[[2]] <- abind(mar, examples.list[[2]], mar, along = 2) %>% as.matrix %>% t %>% as.cimg()
# plot(examples.list[[2]], axes=F)
