#########################################################
#######        COMPUTATIONAL BIOLOGY         ############
#######             HOMEWORK 3               ############
#########################################################
#                                                       #
# Reconstruct the phylogenetic tree of given sequences  #
# using UPGMA with Hamming, JC69, K80 distances.        #
#                                                       #
#########################################################
#########################################################

#########################################################
######    Code below this should not be changed   #######
#########################################################

library(ape)
library(Matrix)

transform_to_phylo = function(sequences, named_edges, edge_lengths, node_description) {
    # Produce a tree of the phylo class from the matrix of edges, vector of lengths and
    # the data frame containing node descriptions.
    #    sequences: a list of the original sequences as strings;
    #    named_edges: an Mx2 matrix of pairs of nodes connected by an edge,
    #                 where the M rows are the different edges and the 2 columns
    #                 are the parent node and the child node of an edge as numeric values;
    #    edge_lengths: a vector of length M of the corresponding edge lengths as numeric values;
    #    node_description: a data frame of the node descriptions as defined in
    #                      initialize_node_description and extended by the UPGMA code.
    
    edges = named_edges
    for (name in rownames(node_description)) {
        index = which(rownames(node_description) == name)
        edges[which(edges == name)] = as.numeric(index)
    }
    edges = matrix(as.numeric(edges), ncol = 2)
    
    edges[which(edges > length(sequences))] = - edges[which(edges > length(sequences))]
    root = setdiff(edges[,1], edges[,2])
    edges[which(edges==root)] = length(sequences) + 1
    
    k = length(sequences) + 2
    for(x in unique(edges[which(edges < 0)])) {
        edges[which(edges==x)] = k
        k = k + 1
    }
    
    tree = list()
    class(tree) = "phylo"
    tree$edge = edges
    tree$edge.length = edge_lengths
    tree$Nnode = as.integer(length(sequences) - 1)
    tree$tip.label = names(sequences)
    
    # Return the tree in the form of the phylo class from ape
    return(tree)
}


plot_tree = function(tree) {
    # Plot the phylogenetic tree with node labels and edge lengths.
    #    tree: an object of class phylo from the ape package

    plot(tree)
    edgelabels(format(tree$edge.length, digits = 2))
}


initialize_node_description = function(sequences) {
    # Initialize the structure that will hold node descriptions.
    # The created structure is a data frame where the rows are node names, and the columns are
    # node_height -- distance from the node to any tip in the ultrametric tree as a numeric value, and
    # node_size -- number of tips that this node is ancestral to as a numeric value.
    #    sequences: a list of the original sequences as strings

    N = length(sequences)
    node_names = names(sequences)
    node_sizes = rep(1, times = N)
    node_heights = rep(0, times = N)
    node_description = data.frame(node_sizes, node_heights)
    rownames(node_description) = node_names

    # Return a data frame that contains information on currently existing tip nodes.
    # node_description: a data frame containing information on the currently existing nodes.
    #                   The row names are the names of the currently existing tips, i.e.
    #                   are the same as the names in the sequence list, node_height is
    #                   0 and node_size is 1 as all the currently existing nodes are tips.
    return(node_description)
}


add_new_node = function(node_description, merging_nodes) {
    # Add new merged node to the node description data frame.
    # The new node is a combination of the nodes supplied in the merging_nodes,
    # e.g. if one needs to merge nodes "bird" and "fish", the new node in the
    # data frame will be called "bird.fish".
    #    node_description: the data frame created by initialize_node_description, containing
    #                      current node sizes and node heights as numeric values;
    #    merging_nodes: a vector of two names of the nodes being merged as strings.

    new_node_name = paste(merging_nodes, collapse = ".")
    new_node_row = data.frame(node_sizes = 0, node_heights = 0)
    rownames(new_node_row) = new_node_name
    new_node_description = rbind(node_description, new_node_row)
    
    # Return the node_description data frame with a row for the new node added, and
    # the new node name.
    #    node_description: the data frame where the rows are labelled by current nodes and columns
    #                      contain the node heights and sizes as numeric values.
    #    new_node_name: the name of the newly added node as a string, created from names in merging_nodes.
    return(list(node_description = new_node_description, new_node_name = new_node_name))
}

#########################################################
######    Code above this should not be changed   #######
#########################################################

get_hamming_distance = function(sequence1, sequence2) {
    # Compute the Hamming distance between two sequences.
    #    sequence1: first sequence (string)
    #    sequence2: second sequence (string)
    sequence1 <- c(strsplit(sequence1, split = "")[[1]]) # split up string 1 and store in a vector
    sequence2 <- c(strsplit(sequence2, split = "")[[1]]) # split up string 2 and store in a vector
    distance <- sum(sequence1 != sequence2) # sum up the number of differences
    return(distance)
}


get_JC69_distance = function(sequence1, sequence2) {
    # Compute the JC69 distance between two sequences.
    #    sequence1: first sequence (string)
    #    sequence2: second sequence (string)
    sequence1 <- c(strsplit(sequence1, split = "")[[1]]) # split up string 1 and store in a vector
    sequence2 <- c(strsplit(sequence2, split = "")[[1]]) # split up string 2 and store in a vector
    ham_dist <- sum(sequence1 != sequence2) # sum up the number of differences
    p <- ham_dist/length(sequence1)
    distance <- (-3/4)*log(1-((4/3)*p))
    # Return the numerical value of the distance
    return(distance)
}


get_K80_distance = function(sequence1, sequence2) {
    # Compute the K80 distance between two sequences.
    #    sequence1: first sequence (string)
    #    sequence2: second sequence (string)
    sequence1 <- c(strsplit(sequence1, split = "")[[1]]) # split up string 1 and store in a vector
    sequence2 <- c(strsplit(sequence2, split = "")[[1]]) # split up string 2 and store in a vector
    tp_count <- 0
    tvp_count <- 0
    n <- length(sequence1)

    for(i in 1:length(sequence1)){
      if(sequence1[i] != sequence2[i]){ # first check if they are not the same
        if(sequence1[i] == "T" & sequence2[i] == "C"){
          tp_count <- tp_count + 1
        }
        else if(sequence1[i] == "C" & sequence2[i] == "T"){
          tp_count <- tp_count + 1
        }
        else if(sequence1[i] == "A" & sequence2[i] == "G"){
          tp_count <- tp_count + 1
        }
        else if(sequence1[i] == "G" & sequence2[i] == "A"){
          tp_count <- tp_count + 1
        }
        else { # otherwise they are transversions
          tvp_count <- tvp_count + 1
        }
      }
    }
    tp <- tp_count/n
    tvp <- tvp_count/n
    distance <- -0.5*log(1-2*tp-tvp)-0.25*log(1-2*tvp)
    # Return the numerical value of the distance
    return(distance)
}


compute_initial_distance_matrix = function(sequences, distance_measure) {
    # Compute the initial distance matrix using one of the distance measures.
    # The matrix is of dimension NxN, where N is the number of sequences.
    # The matrix columns and rows should be labelled with tip names, each row and column
    # corresponding to the appropriate sequence.
    # The matrix can be filled completely (i.e symmetric matrix) or only the upper half
    # (as shown in the lecture).
    # The diagonal elements of the matrix should be Inf.
    #    sequences: the sequence alignment in the form of a list of species names and 
    #               the associated genetic sequences as strings
    #    distance_measure: a string indicating whether the 'hamming', 'JC69' or 'K80' 
    #                      distance measure should be used
    N <- as.numeric(length(sequences))
    distance_matrix <- matrix(nrow = N, ncol = N)
    diag(distance_matrix) <- Inf
    rownames(distance_matrix) <- names(sequences)
    colnames(distance_matrix) <- names(sequences)
    
    # HAMMING
    if(distance_measure == "hamming"){
      for(i in 1:(length(sequences)-1)){
        for(j in (i+1):(length(sequences))){
           distance_matrix[i,j] <- get_hamming_distance(sequences[[i]],sequences[[j]])
        }
      }
    }
    
    # JC69
    if(distance_measure == "JC69"){
      for(i in 1:(length(sequences)-1)){
        for(j in (i+1):(length(sequences))){
          distance_matrix[i,j] <- get_JC69_distance(sequences[[i]],sequences[[j]])
        }
      }
    }
    
    # K80
    if(distance_measure == "K80"){
      for(i in 1:(length(sequences)-1)){
        for(j in (i+1):(length(sequences))){
          distance_matrix[i,j] <- get_K80_distance(sequences[[i]],sequences[[j]])
        }
      }
    }
    # Return the NxN matrix of numeric inter-sequence distances with Inf on the diagonal
    return(distance_matrix)
}


get_merge_node_distance = function(node_description, distance_matrix, merging_nodes, existing_node) {
    # Compute the new distance between the newly created merge node and an existing old node in the tree
    #    node_description: a data frame containing information on the currently existing nodes
    #    distance_matrix: the matrix of current distances between nodes
    #    merging_nodes: a vector of two node names that are being merged in this step
    #    existing_node: one of the previously existing nodes, not included in the new node

    ni <- node_description[merging_nodes[1],1]
    nj <- node_description[merging_nodes[2],1]
    
    dim <- distance_matrix[existing_node,merging_nodes[1]]
    if(is.na(dim)){
      dim <- distance_matrix[merging_nodes[1],existing_node]
    }
    
    djm <- distance_matrix[existing_node,merging_nodes[2]]
    if(is.na(djm)){
      djm <- distance_matrix[merging_nodes[2],existing_node]
    }
    new_distance <- (ni * dim + nj * djm)/(ni+nj)
  
    # Returns the numeric distance between the newly created merge node and the existing node
    return(new_distance)
}


update_distance_matrix = function(node_description, distance_matrix, merging_nodes, new_node_name) {
    # Update the distance matrix given that two nodes are being merged.
    #    node_description: a data frame containing information on the currently existing nodes
    #    distance_matrix: the current distance matrix that needs to be updated
    #    merging_nodes: a vector of two node names that need to be merged in this step
    #    new_node_name: the name with which the merged node should be labelled
    # The resulting matrix should be one column and one row smaller, i.e. if the given distance matrix
    # was MxM, then the updated matrix will be (M-1)x(M-1), where the 2 rows and columns represent the separate
    # nodes undergoing the merge are taken out and a new row and column added that represents the new node.

    # adding new row and column to matrix
    distance_matrix <- cbind(distance_matrix,rep(0, nrow(distance_matrix)))
    distance_matrix <- rbind(distance_matrix,rep(NA,ncol(distance_matrix)))
    distance_matrix[ncol(distance_matrix),nrow(distance_matrix)] <- Inf
    
    # add new name to new row and column
    colnames(distance_matrix)[ncol(distance_matrix)] <- new_node_name
    rownames(distance_matrix)[nrow(distance_matrix)] <- new_node_name
    
    for(i in 1:(nrow(distance_matrix)-1)){
      existing_node <- rownames(distance_matrix)[i]
      distance_matrix[i,ncol(distance_matrix)] <- get_merge_node_distance(node_description, distance_matrix, merging_nodes, existing_node)
    }
    
    # remove the unwanted rows and columns
    distance_matrix <- distance_matrix[!rownames(distance_matrix) %in% merging_nodes, !colnames(distance_matrix) %in% merging_nodes]

    updated_distance_matrix <- distance_matrix
    # Returns the updated matrix of numeric cluster distances
    return(updated_distance_matrix)
}


upgma_one_step = function(node_description, distance_matrix, edges, edge_lengths) {
  
    # Performs one step of the UPGMA algorithm, i.e. the nodes with the smallest distance are merged, 
    # the node height of the new node is calculated and the distance matrix is newly calculated.
    #    node_description: a data frame containing information on the currently existing nodes
    #    distance_matrix: the current distance matrix that needs to be updated (LxL)
    #    edges: a matrix of pairs of nodes connected by an edge, where the rows are the different
    #           edges and the 2 columns are the parent node and the child node of an edge.
    #    edge_lengths: a vector with the corresponding numeric edge lengths.
  
    # find position of smallest value in distance_matrix to get merging_nodes
    inds <- which(distance_matrix <= distance_matrix[which.min(distance_matrix)], arr.ind = TRUE)
    rname <- rownames(distance_matrix)[inds[1,1]]
    cname <- colnames(distance_matrix)[inds[1,2]]
    
    merging_nodes <- c(rname, cname)
    new_node_name <- paste(merging_nodes, collapse = ".")
    
    # update node description
    node_description <- add_new_node(node_description, merging_nodes)$node_description # add new node to node_description
    
    #update values of new node
    node_description$node_sizes[nrow(node_description)] <- node_description[nrow(node_description)-1,1] +1
    node_description$node_heights[nrow(node_description)] <- distance_matrix[inds[1,1],inds[1,2]]/2 # node_heights
    
    # edge_lengths
    if(length(edge_lengths) == 0){
      branch1 <- distance_matrix[inds][1]/2
      branch2 <- distance_matrix[inds][1]/2
      edge_lengths <- c(branch1,branch2)
    }
    else{
      branch2 <- distance_matrix[inds][1]/2 - edge_lengths[length(edge_lengths)-1] 
      branch1 <- distance_matrix[inds][1]/2 
      edge_lengths <- append(edge_lengths,branch1)
      edge_lengths <- append(edge_lengths,branch2)
    }
    
    # edges
    parent_node <- new_node_name
    child_node1 <- merging_nodes[1]
    child_node2 <- merging_nodes[2]
    new_row1 <- c(parent_node, child_node1)
    new_row2 <- c(parent_node, child_node2)
    edges <- rbind(edges, new_row1)
    edges <- rbind(edges, new_row2)
    
    distance_matrix <- update_distance_matrix(node_description, distance_matrix, merging_nodes, new_node_name)
    
    # Return the updated distance matrix, edge description matrix, edge length vector and 
    # node_description data frame
    #    node_description: data frame containing sizes and heights of all nodes 
    #        (to be updated using add_new_node())
    #    distance_matrix: the updated matrix of numeric distances between nodes ((L-1)x(L-1))
    #    edges: a matrix of pairs of nodes connected by an edge, where the rows are the different
    #           edges and the 2 columns are the parent node and the child node of an edge.
    #    edge_lengths: a vector with the corresponding numeric edge lengths.
    
    
    
    return(list(node_description = node_description, distance_matrix = distance_matrix,
                edges = edges, edge_lengths = edge_lengths))
}


build_upgma_tree = function(sequences, distance_measure) {
    # Build the tree from given sequences using the UPGMA algorithm.
    #    sequences: the sequences in the format of a list of species names and the associated genetic 
    #               sequences as strings.
    #    distance_measure: a string indicating whether the 'hamming', 'JC69' or 'K80' distance measure
    #                      should be used
    N <- as.numeric(length(sequences))
    node_description <- initialize_node_description(sequences)
    edges <- matrix(nrow = 0, ncol = 2)
    edge_lengths <- vector(mode = "numeric", length = 0)
    
    # initialize distance_matrix
    distance_matrix <- compute_initial_distance_matrix(sequences, distance_measure)
    
    # iterate through all the steps with upgma_one_step
    for(i in 1:(N-1)){
      results <- upgma_one_step(node_description, distance_matrix, edges, edge_lengths)
      # update values
      edges <- results$edges
      edge_lengths <- results$edge_lengths
      node_description <- results$node_description
      distance_matrix <- results$distance_matrix
    }
    # Return the UPGMA tree of sequences in the form of the phylo class from ape
    tree <- transform_to_phylo(sequences, edges, edge_lengths, node_description)
    return(tree)
}


test_tree_building = function() {
    sequences <- list(orangutan = "TCACACCTAAGTTATATCTATATATAATCCGGGCCGGG",
                     chimp = "ACAGACTTAAAATATACATATATATAATCCGGGCCGGG",
                     human = "AAAGACTTAAAATATATATATATATAATCCGGGCCGGG",
                     gorilla = "ACACACCCAAAATATATATATATATAATCCGGGCCGGG",
                     unicorn = "ACACACCCAAAATATATACGCGTATAATCCGGGCCGAA")
    #distance_measure <- 'hamming'
    distance_measure <- 'JC69'
    #distance_measure <- 'K80'

    tree <- build_upgma_tree(sequences, distance_measure)
    plot_tree(tree)
}

test_tree_building()




