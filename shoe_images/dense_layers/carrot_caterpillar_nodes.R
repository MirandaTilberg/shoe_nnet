library(magrittr)
library(purrr)
library(dplyr)
library(tidygraph)
library(ggplot2)
library(ggraph)


make_node_name <- function(level, label){
  paste("l", level, label, sep = "_")
}


make_nodes_df <- function(n){
  
  map_dfr(
    seq_along(n),
    function(i){
      label = seq_len(n[i])
      tibble(
        label = label %>% as.character(),
        node = make_node_name(i, label)
      )
    }
  ) %>%
    mutate(
      id = seq_along(node),
      type = NA,
    ) %>%
    select(id, type, label, node)
}


make_edges_df <- function(n, nodes = make_nodes_df(n)){
  
  embedded_nodes <- n %>%
    embed(dimension = 2) %>%
    .[, c(2, 1)] %>%
    matrix(ncol = 2)
  
  map_dfr(seq_len(length(n) - 1), function(i){
    level <-  i
    emb <- embedded_nodes[i, ]
    from <-  rep(seq_len(emb[1]), each = emb[2])
    to <-  rep(seq_len(emb[2]), emb[1])
    tibble(
      from = make_node_name(level, from),
      to = make_node_name(level + 1, to)
    )
  }) %>%
    left_join(nodes, by = c("from" = "node")) %>%
    select(to, id) %>%
    rename(from = id) %>%
    left_join(nodes, by = c("to" = "node")) %>%
    select(from, id) %>%
    rename(to = id)
}


layout_keras <- function(graph, n_nodes){
  positions <- map_dfr(seq_along(n_nodes), function(i){
    max_nodes <- max(n_nodes)
    layers <- length(n_nodes)
    data.frame(
      x = seq_len(n_nodes[i]) / (n_nodes[i] + 1) * layers,
      y = length(n_nodes) - i
    )
  })
  ggraph:::layout_igraph_manual(graph, positions, circular = FALSE)
}

plot_deepviz2 <- function(n, edge_col = "grey50", line_type = "solid", rad = .1){
  nodes <- make_nodes_df(n)
  edges <- make_edges_df(n)
  
  tbl_graph(nodes = nodes, edges = edges) %>%
    ggraph(layout = "manual", node.position = layout_keras(., n)) +
    geom_edge_diagonal0(edge_colour = edge_col, linetype = line_type) +
    geom_node_circle(aes(r = rad), fill = "grey40") +
    coord_fixed() +
    theme_void()
}



col_vec <- rep("grey50", 32)
col_vec[seq(1, 29, by = 4)] <- rep(c("blue", "orange"), 4)

lty <- rep("solid", 32)
lty[c(21,25)] <- "dashed"

plot <- plot_deepviz2(c(4,2), edge_col = col_vec, line_type = lty)

df <- data.frame(x = plot$data$x, 
                 y = plot$data$y,
                 labs = c("Orange", "Long", "Fuzzy", "Pointy", 
                          "Caterpillar", "Carrot"))

plot + geom_label(aes(x = x, y = y, label = labs), data = df)
ggsave("/home/tiltonm/shoe_nnet/shoe_images/dense_layers/carrot_caterpillar_nodes.png",
       height = 3, width = 4)

plot_deepviz2(c(4, 12), r = .005) + coord_flip()
plot_deepviz2(c(12, 4), r = .005)

ggsave("/home/tiltonm/shoe_nnet/shoe_images/dense_layers/dense_layers_v.png",
       height = 3, width = 4)
