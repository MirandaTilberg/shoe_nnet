library(shiny)
library(dplyr)
library(data.table)
library(DT)
library(magrittr)


mod.num <- 6

choose.model <- function(n) {
  mod.path <- "~/shoe_nnet/shoe_models/OneHot/"
  
  if (n==1) { #first onehot model, before quality control
    mf <- "081518_vgg16_onehot_256_2.Rdata"
    imfolder <- "onehotV1"
  } else if (n==2) { 
    mf = "090818_vgg16_onehot_3class_256_2.Rdata"
    imfolder <- "onehotV2"
  } else if (n==3) { 
    mf = "090918_vgg16_onehot_8class_256_2.Rdata"
    imfolder <- "onehotV2"
  } else if (n==4) { 
    mf = "091018_vgg16_onehot_12class_256_2.Rdata"
    imfolder <- "onehotV2"
  } else if (n==5) { 
    mf = "091018_vgg16_onehot_10class_256_2.Rdata"
    imfolder <- "onehotV2"
  } else if (n==6) { 
    mf = "091818_vgg16_onehotaug_8class_256_2.Rdata"
    imfolder <- "onehotV2"
  } else {
    return(choose.model(2))
  }
  return(list(model.file = paste(mod.path, mf, sep=""), folder = imfolder))
}

mod.list <- choose.model(mod.num)
model.file <- mod.list$model.file
folder <- mod.list$folder

ui <- fluidPage(
  
  # Application title
  titlePanel("Neural Network Shoeprint Shape Predictions"),
  
  mainPanel(width = 12,
    DT::dataTableOutput("out",width = '100%')
  )
)

# Define server logic required to draw a histogram
server <- function(input, output) {
  load(model.file)
  classes <- colnames(preds)
  fnames <- list.files(paste("~/shoe_nnet/nnet_pred_viewer/www/", 
                             folder, sep=""))
  truth <- test_labs
  colnames(truth) <- colnames(test_labs) %>% 
    paste("truth", ., sep="_")
  
  color_vals <- 2*(truth + 1) + (round(preds, 2))
  colnames(color_vals) <- paste("color", classes, sep="_")

  goodcol <- "cornflowerblue"
  correct <- colorRampPalette(c("white", goodcol))
  incorrect <- colorRampPalette(c("white", "grey40"))
  
  
  html_urls <- sprintf("https://bigfoot.csafe.iastate.edu/rstudio/files/ShinyApps/NNPreview/www/%s/%s",
                       folder, fnames)

  html_fnames <- sprintf('<img src ="%s" width = "100%%"/>', 
                         paste(folder, fnames, sep="/"))
  
  hyperlinks <- sprintf('<a href="%s">%s</a>', html_urls, html_fnames)
  
  obj <- data.table(cbind(Image = hyperlinks, round(preds, 2), color_vals))
  set.seed(2)
  obj <- obj[sample(1:length(fnames), length(fnames)),]
  
  output$out <- DT::renderDataTable({
    DT::datatable(obj,
              escape = F,
              extensions = c("FixedHeader"),
              options = list(
                fixedHeader = T,
                pageLength = 100,
                autoWidth = FALSE,
                columnDefs = list(list(targets = 
                                         (ncol(obj)-length(classes)+1):(ncol(obj)), 
                                       visible = FALSE))
              )) %>%
      formatStyle(classes, 
                  valueColumns = paste("color", classes, sep="_"), 
                  target = "cell",
                  backgroundColor = styleEqual(c(200:300, 400:500)/100,
                                               c(incorrect(101), 
                                                 correct(131)[-(1:30)]))
      )
  })
}

# Run the application 
setwd("~/shoe_nnet/nnet_pred_viewer/")
shinyApp(ui = ui, server = server)

