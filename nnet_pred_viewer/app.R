library(shiny)
library(dplyr)
library(data.table)
library(DT)
library(magrittr)


mod.num <- 3

choose.model <- function(n) {
  if (n==1) { #first onehot model, before quality control
    mf <- "~/shoe_nnet/shoe_models/OneHot/081518_vgg16_onehot_256_2.Rdata"
    imfolder <- "onehotV1"
    }
  if (n==2) { #after quality control, 3 classes
    mf = "~/shoe_nnet/shoe_models/OneHot/090818_vgg16_onehot_3class_256_2.Rdata"
    imfolder <- "onehotV2"
    }
  if (n==3) { #after QC, 8 classes
    mf = "~/shoe_nnet/shoe_models/OneHot/090918_vgg16_onehot_8class_256_2.Rdata"
    imfolder <- "onehotV2"
    }
  return(list(model.file = mf, folder = imfolder))
}
mod.list <- choose.model(mod.num)
model.file <- mod.list$model.file
folder <- mod.list$folder

ui <- fluidPage(
  
  # Application title
  titlePanel("Neural Network Shoeprint Shape Predictions"),
  
  mainPanel(
    DT::dataTableOutput("out")
  )
)

# Define server logic required to draw a histogram
server <- function(input, output) {
  load(model.file)
  classes <- colnames(preds)
  fnames <- list.files(paste("~/shoe_nnet/nnet_pred_viewer/www/", folder, sep=""))
  truth <- test_labs
  colnames(truth) <- colnames(test_labs) %>% 
    paste("truth", ., sep="_")
  
  
  color_vals <- (2 * truth + preds) %>% round(2)
  colnames(color_vals) <- paste("color", classes, sep="_")
  #goodcol <- sample(c("palegreen", "cornflowerblue", "plum"), 1)
  goodcol <- "cornflowerblue"
  correct <- colorRampPalette(c("white", goodcol))
  incorrect <- colorRampPalette(c("white", "grey30"))
  
  
  html_urls <- sprintf("https://bigfoot.csafe.iastate.edu/rstudio/files/shoe_nnet/shoes/onehot/test/%s",
                       fnames)

  html_fnames <- sprintf('<img src ="%s" width = "100%%"/>', 
                         paste(folder, fnames, sep="/"))
  
  hyperlinks <- sprintf('<a href="%s">%s</a>', html_urls, html_fnames)
  
  obj <- data.table(cbind(Image = hyperlinks, round(preds, 2), color_vals))
  set.seed(2)
  obj <- obj[sample(1:length(fnames), length(fnames)),]
  
  output$out <- DT::renderDataTable({
    datatable(obj,
              escape = F,
              extensions = c("FixedHeader"),
              options = list(
                fixedHeader = T,
                pageLength = 100,
                columnDefs = list(list(targets = 
                                         (ncol(obj)-length(classes)+1):(ncol(obj)), 
                                       visible = FALSE))
              )) %>%
      formatStyle(classes, 
                  valueColumns = paste("color", classes, sep="_"), 
                  target = "cell",
                  backgroundColor = styleEqual(c(0:100, 200:300)/100,
                                               c(incorrect(101), correct(141)[-(1:40)]))
      )
  })
}

# Run the application 
setwd("~/shoe_nnet/nnet_pred_viewer/")
shinyApp(ui = ui, server = server)

