library(shiny)
library(dplyr)
library(data.table)
library(DT)
library(magrittr)

imgs <- "/models/shoe_nn/RProcessedImages"
modelres <- "/models/shoe_nn/TrainedModels"

models <- data_frame(
  name = c("onehotV1", "onehotV2 - 3 class", "onehotV2 - 8 class",
           "onehotV2 - 12 class", "onehotV2 - 10 class", "aug - 8 class",
           "aug - 10 class"),
  imfolder = c("onehotV1", rep("onehotV2", 6)),
  data = paste0("/home/tiltonm/shoe_nnet/shoe_models/OneHot/",
                c("081518_vgg16_onehot_256_2.Rdata",
                  "090818_vgg16_onehot_3class_256_2.Rdata",
                  "090918_vgg16_onehot_8class_256_2.Rdata",
                  "091018_vgg16_onehot_12class_256_2.Rdata",
                  "091018_vgg16_onehot_10class_256_2.Rdata",
                  "091818_vgg16_onehotaug_8class_256_2.Rdata",
                  "092018_vgg16_onehotaug_10class_256_2.Rdata")
  )
)

auto_models <- list.files("/models/shoe_nn/TrainedModels", ".[Rr]data$", recursive = T)
auto_models <- auto_models[!grepl("fullimage.rdata", auto_models)]
auto_models <- auto_models[!grepl("train-valid-test-data.Rdata", auto_models)]
auto_models <- auto_models[file.exists(file.path("/models/shoe_nn/TrainedModels", stringr::str_replace(auto_models, "\\..*$", ".h5")))]
auto_model_data_date <- stringr::str_extract(auto_models, "201\\d\\d{2}\\d{2}-\\d{2}\\d{2}\\d{2}")
auto_model_date <- stringr::str_extract(auto_models, "201\\d-\\d{2}-\\d{2}[ _]\\d{2}:\\d{2}:\\d{2}")
auto_model_nicedate <- auto_model_date %>%
  stringr::str_replace("_", " ")

auto_model_name <- stringr::str_extract(auto_models, "(vgg16_.*_\\d{1,3})\\..*$") %>%
  stringr::str_replace("vgg16_onehot(.*?)_(\\d{1,}class)_\\d{1,}\\..*$", "\\1 \\2")

tmp <- sapply(unique(auto_model_data_date), function(x) {
  fp <- file.path("/models/shoe_nn/RProcessedImages", x)
  fpnew <- file.path(getwd(), "www", x)
  if (!dir.exists(fpnew)) {
    file.symlink(from = fp, to = fpnew)
  } else {
    TRUE
  }
})
# file.symlink(from = file.path("/models/shoe_nn/RProcessedImages",
#                               unique(auto_model_data_date)),
#              to = file.path(getwd(), "www", unique(auto_model_data_date)))


models2 <- data_frame(
  name = paste(auto_model_nicedate, auto_model_name),
  imfolder = file.path(auto_model_data_date, "test"),
  data = auto_models
)

models <- bind_rows(models, models2)
ui <- fluidPage(

  # Application title
  titlePanel("Neural Network Shoeprint Shape Predictions"),

  mainPanel(
    width = 12,
    selectInput("model", label = "Select model", choices = models$name,
                selected = rev(models$name)[1]),
    DT::dataTableOutput("out",width = '100%')
  )
)

server <- function(input, output) {

  addResourcePath("imgs", "/models/shoe_nn/RProcessedImages/")
  addResourcePath("modelres", "/models/shoe_nn/TrainedModels/")

  # Load Data
  model_data <- reactive({
    # This should update when input$model updates
    filter(models, name == input$model)
  })

  output$out <- DT::renderDataTable({
    message(model_data()$data)

    load(file.path(modelres, model_data()$data))

    validate(
      need(exists("preds"), "preds is not loaded"),
      need(exists("test_labs"), "preds is not loaded")
    )

    classes <- colnames(preds)
    fnames <- list.files(file.path(imgs, model_data()$imfolder))

    head(fnames)

    truth <- test_labs
    colnames(truth) <- colnames(test_labs) %>%
      paste("truth", ., sep="_")

    color_vals <- 2*(truth + 1) + (round(preds, 2))
    # [2,3] -> gray, [4,5] -> blue
    colnames(color_vals) <- paste("color", classes, sep="_")

    correct <- colorRampPalette(c("white", "cornflowerblue"))
    incorrect <- colorRampPalette(c("white", "grey40"))


    # image_urls <- sprintf("https://bigfoot.csafe.iastate.edu/rstudio/files/ShinyApps/NNPreview/www/%s/%s",
    #                      folder, fnames)
    whole_shoe <- stringr::str_replace_all(fnames, "^[[a-z\\(\\)RE]*_]*-[\\d\\.]*-", "")
    image_urls <- sprintf("https://bigfoot.csafe.iastate.edu/LabelMe/tool.html?actions=a&folder=Shoes&image=%s", whole_shoe)

    image_tags <- sprintf('<img src ="%s" width = "100%%"/>',
                          file.path("imgs", stringr::str_replace(model_data()$imfolder, "test", "images"), fnames))

    hyperlinks <- sprintf('<a href="%s" target="_blank">%s</a>', image_urls, image_tags)

    obj <- data.table(cbind(Image = hyperlinks, round(preds, 2), color_vals))
    set.seed(2)
    obj <- obj[sample(1:length(fnames), length(fnames)),]

    DT::datatable(obj,
              escape = F,
              extensions = c("FixedHeader"),
              options = list(
                fixedHeader = T,
                pageLength = 100,
                autoWidth = FALSE,
                columnDefs = list(list(targets = tail(1:ncol(obj), length(classes)),
                                         #(ncol(obj)-length(classes)+1):(ncol(obj)),
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
# setwd("~/shoe_nnet/nnet_pred_viewer/")
shinyApp(ui = ui, server = server)

