library(magrittr)
library(ggplot2)
########## Choose model file and detect classes
#load("/home/tiltonm/shoe_nnet/shoe_models/OneHot/092018_vgg16_onehotaug_10class_256_2.Rdata")
#load("/home/tiltonm/shoe_nnet/shoe_models/OneHot/092518_vgg16_onehotaug_10class_256_2.Rdata")
load("/models/shoe_nn/TrainedModels/20181004-154202/2018-10-04_18:13:55_vgg16_onehotaug_9class_256.Rdata")

classes <- colnames(preds)
n_per_class <- apply(test_labs, 2, sum)
n_per_class

n_classes <- length(classes)
preds <- preds %>% round(2)


########## Create confusion matrix
confusion_mat <- matrix(0, ncol = n_classes, nrow = n_classes)
class_rel <- matrix(0, ncol = n_classes, nrow = n_classes)

for (i in 1:n_classes) {
  true_i <- test_labs[,i]==1
  n_i <- sum(true_i)
  
  class_rel[i,] <- test_labs[true_i,] %>% apply(., 2, sum)
  
  preds_i <- preds
  preds_i[,-i] <- (preds_i[,-i] - test_labs[,-i])
  preds_i[preds_i < 0] <- NA
  
  
  f <- function(vec) {
    thresh = .3
    sum(vec > thresh, na.rm = T)
  }
  
  
  #metric <- apply(preds_i[true_i,], 2, max, na.rm=T) %>% round(2)
  #metric <- apply(preds_i[true_i,], 2, mean, na.rm=T) %>% round(2)
  metric <- (apply(preds_i[true_i,], 2, f) / n_i) %>% round(2)
  
  confusion_mat[i,] <- metric
}

colnames(confusion_mat) <- rownames(confusion_mat) <- classes
colnames(class_rel) <- rownames(class_rel) <- classes

########## Plot correlation matrix
source("/home/tiltonm/shoe_nnet/shoe_images/ggcorrplot2.R")
cols <- colorRampPalette(c("white", "cornflowerblue"))
conf_plot <- ggcorrplot2(confusion_mat, hc.order = F, outline.col = "white", 
                         colors = c("white", cols(2)), legend.title = "Value",
                         lab = T, lab_size = 3.5, show.legend = F) + 
  xlab("Truth") + ylab("Predictions")

quantity_plot <- ggcorrplot2(class_rel, limit = c(0, max(class_rel)),
                             colors = c("white", cols(2)), legend.title = "Quantity", lab = T) +
  theme(axis.title.x = ggplot2::element_blank(),
        axis.title.y = ggplot2::element_blank())

conf_plot
#quantity_plot

