#####################################################
#######        COMPUTATIONAL BIOLOGY         ########
#######             HOMEWORK 1               ########
#####################################################
#                                                   #
# Implement the pairwise alignment algorithms       #
# Needleman-Wunsch and Smith-Waterman.              #
#                                                   #
#####################################################
#####################################################

# In all functions the following parameters are the same:
# seqA: the first sequence to align
# seqB: the second sequence to align
# score_gap: score for a gap
# score_match: score for a character match
# score_mismatch: score for a character mismatch
# local: (logical) True if alignment is local, False otherwise

# GLOBAL --> Needleman-Wunsch algorithm
# LOCAL -->  Smith-Waterman algorithm

init_score_matrix = function(nrow, ncol, local, score_gap) {
    score_matrix <- matrix(0, nrow, ncol)# Initialize the score matrix with zeros.
    # If the alignment is global, the leftmost column and the top row will have incremental gap scores,
    # i.e. if the gap score is -2 and the number of columns is 4, the top row will be [0, -2, -4, -6].
    # nrow: (numeric) number of rows in the matrix
    # ncol: (numeric) number of columns in the matrix
    if (local == FALSE){
      
      counter = 0
      for (i in 1:ncol){
        score_matrix[1,i] <- counter
        score_matrix[i,1] <- counter
        counter <-  score_gap + counter
      }
    }
    # if local == TRUE, just continue with the Null Matrix
    
    # Return the initialized empty score matrix
    # score_matrix: (numeric) nrow by ncol matrix
    return(score_matrix)
}

score_matrix <- init_score_matrix(3,3, FALSE, -2)
score_matrix

init_path_matrix = function(nrow, ncol, local) {
    # Initialize the path matrix with empty values ("").
    # Additionally, for GLOBAL alignment (i.e. local==FALSE), make the first row
    # have "left" on all positions except 1st, and make the first column
    # have "up" on all positions except 1st.
    # nrow: (numeric) number of rows in the matrix
    # ncol: (numeric) number of columns in the matrix
  
    path_matrix <- matrix("",nrow, ncol)
    
    if (local == FALSE){
      for (i in 2:ncol){
        path_matrix[1,i] <- "left"
        path_matrix[i,1] <- "up"
      }
    }
    # if local == TRUE, continue with empty path matrix
    # Return the initialized empty path matrix
    # path_matrix: (character) nrow by ncol matrix
    return(path_matrix)
}

init_path_matrix(3,3,FALSE)

get_best_score_and_path = function(row, col, nucA, nucB, score_matrix, score_gap, score_match, score_mismatch, local) {
    # Compute the score and the best path for a particular position in the score matrix
    # nucA: (character) nucleotide in sequence A
    # nucB: (character) nucleotide in sequence B
    # row: (numeric) row-wise position in the matrix
    # col: (numeric) column-wise position in the matrix
    # score_matrix: (double) the score_matrix that is being filled out
    if (local == FALSE){#Needleman-Wunsch algorithm
      vec_score <- c(0,0,0)
      vec_path <- c(0,0,0)

      # assign all possible scores to the respective condition
      if (nucA == nucB){#MATCH
        vec_score[1] <- score_match
        vec_score[2:3] <- -10
      }
      if (nucA != nucB){#GAP
        vec_score[2] <- score_mismatch
        vec_score[3] <- score_gap
        vec_score[1] <- -10
      }

      max_score <- max(vec_score) #get the highest score
      
      #score_location <- which(max(vec_score)==vec_score) # get position of highest score in the vector
      
      # assign path variable based on the max value returned
      if(max_score == score_match){
        best_path <- "diag"
        # not sure come back
      }
      if(max_score == score_mismatch){
        best_path <- "diag"
      }
      if(max_score == score_gap) #GAP
        best_path <- "up"
      
      # if(as.numeric(score) == 3){
      #   path <- left
      # }
      score <- max_score
      path <- best_path
    }
    # ???

    # Return the best score for the particular position in the score matrix
    # In the case that there are several equally good paths available, return any one of them.
    # score: (numeric) best score at this position
    # path: (character) path corresponding to the best score, one of ["diag", "up", "left"] in the global case and of ["diag", "up", "left", "-"] in the local case
    return(list("score"=score, "path"=path))
}

get_best_score_and_path(1,2,nucA,nucB,score_matrix,-2,3,-1, FALSE)
nucA <- "A"
nucB <- "A"

seqA <- "AB"
seqB <- "AA"
score_matrix <- init_score_matrix(3,3, FALSE, -2)
score_matrix


fill_matrices = function(seqA, seqB, score_gap, score_match, score_mismatch, local, score_matrix, path_matrix) {
    # Compute the full score and path matrices
    # score_matrix: (numeric)  initial matrix of the scores
    # path_matrix: (character) initial matrix of paths
  if (local == FALSE) {
  # turn split the strings and out into a vector 
  seqA <- strsplit(seqA, "")[[1]]
  seqB <- strsplit(seqB, "")[[1]]
  
  best_score <- as.numeric(get_best_score_and_path(i,j,seqA[1],seqB[1],score_matrix,score_gap,score_match, score_mismatch,FALSE)[1])
  
  
  for (i in 2:length(seqA)+1){
    for (j in 2:length(seqB)+1){
      best_score <- as.numeric(get_best_score_and_path(i,j,seqA[i-1],seqB[j-1],score_matrix,score_gap,score_match, score_mismatch,FALSE)[1])
      best_path <- as.numeric(get_best_score_and_path(i,j,seqA[i-1],seqB[j-1],score_matrix,score_gap,score_match, score_mismatch,FALSE)[2])
      
      if (best_score == score_match){
        score_matrix[i,j] <- score_matrix[i-1,j-1] + score_match 
      }
      if (best_score == score_mismatch){
        score_matrix[i,j] <- score_matrix[i-1,j-1] + score_mismatch 
      }
      if (best_score == score_gap){
        score_matrix[i,j] <- score_matrix[i-1,j] + score_gap 
        score_matrix[i,j] <- score_matrix[i,j-1] + score_gap 
      }
      
      
      # if (seqA[i-1] == seqB[j-1]){
      #   score_matrix[i,j] <- best_score
      # }
      # if (seqA[i-1] != seqB[j-1]){
      #   score_matrix[i+1,j] <- best_score
      #   score_matrix[i,j+1] <- best_score
      # }
    }
  }
  
  
  
  # iterate through
  # for (i in 1:length(seqA)){
  #   for (j in 1:length(seqB)){
  #     best_s <- as.numeric(get_best_score_and_path(i,j,seqA[i],seqB[j],score_matrix,score_gap,score_match,score_mismatch,local)[1])
  #     if (seqA[i] == seqB[j]){
  #       score_matrix[i+1,j+1] <- best_s
  #     }
  #     if (seqA[i] != seqB[j]){
  #       score_matrix[i+1,j] <- best_s
  #       score_matrix[i,j+1] <- best_s
  #     }
  #   }
  # }
  score_matrix <- mat_score
  path_matrix <- 0
  }
  
    # ???

    # Return the full score and path matrices
    # score_matrix: (numeric) filled up matrix of the scores
    # path_matrix: (character) filled up matrix of paths
    return(list("score_matrix"=score_matrix, "path_matrix"=path_matrix))
}
fill_matrices("ATGC", "ACGC", -2, 3, -1, FALSE, score_matrix, path_matrix)
  

get_best_move = function(nucA, nucB, path, row, col) {
    # Compute the aligned characters at the given position in the score matrix and return the new position,
    # i.e. if the path is diagonal both the characters in seqA and seqB should be added,
    # if the path is up or left, there is a gap in one of the sequences.
    # nucA: (character) nucleotide in sequence A
    # nucB: (character) nucleotide in sequence B
    # path: (character) best path pre-computed for the given position
    # row: (numeric) row-wise position in the matrix
    # col: (numeric) column-wise position in the matrix

    # ???

    # Return the new row and column and the aligned characters
    # newrow: (numeric) row if gap in seqA, row - 1 otherwise
    # newcol: (numeric) col if gap in seqB, col - 1 otherwise
    # char1: (character) '-' if gap in seqA, appropriate character if a match
    # char2: (character) '-' if gap in seqB, appropriate character if a match
    return(list("newrow"=newrow, "newcol"=newcol, "char1"=char1, "char2"=char2))
}

get_best_alignment = function(seqA, seqB, score_matrix, path_matrix, local) {
    # Return the best alignment from the pre-computed score matrix
    # score_matrix: (numeric) filled up matrix of the scores
    # path_matrix: (character) filled up matrix of paths

    # get_best_move(nucA, nucB, path, row, col) #need to use this one here

    # Return the best score and alignment (or one thereof if there are multiple with equal score)
    # score: (numeric) score of the best alignment
    # alignment: (character) the actual alignment in the form of a vector of two strings
    return(list("score"=score, "alignment"=alignment))
}

align = function(seqA, seqB, score_gap, score_match, score_mismatch, local) {
    # Align the two sequences given the scoring scheme
    # For testing purposes, use seqA for the rows and seqB for the columns of the matrices
  
    # Initialize score and path matrices
    score_matrix <- init_score_matrix(length(seqA),length(seqB),local, score_gap)
    path_matrix <- init_path_matrix(length(seqA),length(seqB),local)
    
    # Fill in the matrices with scores and paths using dynamic programming
    fill_matrices(seqA,seqB,score_gap, score_match, score_mismatch, local,score_matrix,path_matrix)
  
    # Get the best score and alignment (or one thereof if there are multiple with equal score)
    get_best_alignment(seqA, seqB, score_matrix, path_matrix, local)
    
    result <- list("score"=score,"alignment"=alignment)
    # Return the best score and alignment (or one thereof if there are multiple with equal score)
    # Returns the same value types as get_best_alignment
    return(result)
}

test_align = function() {
    seqA = "TCACACTAC"
    seqB = "AGCACAC"
    score_gap = -2
    score_match = +3
    score_mismatch = -1
    local = F
    result = align(seqA, seqB, score_gap, score_match, score_mismatch, local)
    print(result$alignment)
    print(result$score)
}

test_align()
nucA <- "A"
nucB <- "A"
local <- FALSE
