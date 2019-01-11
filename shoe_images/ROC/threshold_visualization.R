library(magrittr)
library(ggplot2)
source("/home/tiltonm/shoe_nnet/helper_scripts/get_most_recent.R")

predictions_file <- get_most_recent(verbose = T)
load(predictions_file)
classes <- colnames(preds)
n_classes <- length(classes)
preds <- preds %>% round(2)
confusion_mat <- matrix(0, ncol = n_classes, nrow = n_classes)
confusion_arr <- array(data = confusion_mat, dim = c(dim(confusion_mat), 101))

s <- 0:100/100


for (j in 1:length(s)) {
  for (i in 1:n_classes) {
    true_i <- test_labs[,i]==1
    n_i <- sum(true_i)
    
    preds_i <- preds
    preds_i[,-i] <- (preds_i[,-i] - test_labs[,-i])
    preds_i[preds_i < 0] <- NA
    
    
    f <- function(vec) {
      thresh = s[j]
      sum(vec > thresh, na.rm = T)
    }
    metric <- (apply(preds_i[true_i,], 2, f) / n_i) %>% round(2)
    
    confusion_mat[i,] <- metric
  }
  confusion_arr[,,j] <- confusion_mat
}


############################### Thresholds
plot_matrix_thresholds <- function() {
  par(mfrow = c(n_classes, n_classes), mar = rep(0,4))
  for (i in n_classes:1) {
    for (j in 1:n_classes) {
      col <- ifelse(i==j, "red", "blue")
      plot(s, confusion_arr[i,j,], type = "l", xlim = 0:1, ylim = 0:1,
           xlab = classes[i], ylab = classes[j], axes = F, col = col)
      lines(rep(.2, 9), (1:9)/10, col = "gray30", lty = "dashed")
    }
  }
}

plot_stacked_thresholds <- function() {
  par(mfrow = c(1,1), mar = c(4,5,1,1))
  plot(s, confusion_arr[1,1,], type = "l", xlim = 0:1, ylim = 0:1, col = "red",
       xlab = "Probability Threshold", 
       ylab = "% Correctly Identified \n above Threshold")
  for (i in n_classes:1) {
    for (j in 1:n_classes) {
      col <- ifelse(i==j, "red", "blue")
      lines(s, confusion_arr[i,j,], type = "l", xlim = 0:1, ylim = 0:1,
            axes = F, col = col)
    }
  }
  lines(rep(.2, 11), (0:10)/10, col = "gray30", lty = "dashed")
  legend("topright", 
         fill = c("red", "blue"), 
         legend = c("Correct", "Incorrect"))
}

#plot_matrix_thresholds()
#plot_stacked_thresholds()

############################### ROC/AUC
true_pos <- apply(confusion_arr, 3, diag) %>% set_rownames(classes) %>% 
  cbind (1, ., 0)
false_pos <- apply(confusion_arr, 3, 
                   function(x) {x - diag(diag(x))})
dim(false_pos) <- dim(confusion_arr)
false_pos <- apply(false_pos, 3, function(x) {apply(x, 1, max)}) %>%
  cbind(1, ., 0)


AUC <- c()
for (i in 1:9) {
  lower <- sum(true_pos[i,][-1] * 
                 ((false_pos[i,][-length(s)] - false_pos[i,][-1])+1e-5))
  upper <- sum(true_pos[i,][-length(s)] * 
                 ((false_pos[i,][-length(s)] - false_pos[i,][-1])+1e-5))
  
  AUC[i] <- .5* (lower + upper)
}
names(AUC) <- classes

plot_roc <- function() {
  par(mfrow = c(3,3), mar = rep(2, 4))
  for (i in 1:9) {
    plot(false_pos[i,], true_pos[i,], type = "l", 
         main = paste0(tools::toTitleCase(classes[i]), " (AUC = ", round(AUC[i],2), ")"))
    lines(s,s, lty = "dashed", col = "gray30")
  }
}

#plot_roc()


############ Ratios
ratio <- (true_pos/(false_pos+1e-5))[,-c(1, 103)]

summary(ratio[1,])

round(2*max(apply(ratio, 1, median)))

plot_ratio <- function() {
  par(mfrow = c(3,3), mar = rep(2, 4))
  for (i in 1:9) {
    plot(s, ratio[i,], type = "l", 
         main = paste0(tools::toTitleCase(classes[i])),
         ylim = c(0, round(2*max(apply(ratio, 1, median)))))
    abline(v =.2, lty = "dashed", col = "gray30")
  }
}
plot_ratio()



plot_matrix_thresholds()
plot_stacked_thresholds()

png(paste0("~/shoe_nnet/shoe_images/ROC/AUC_",
           names(predictions_file), ".png"),
       height = 4.5, width = 6, units = "in", res = 300)
plot_roc()
dev.off()
plot_ratio()
