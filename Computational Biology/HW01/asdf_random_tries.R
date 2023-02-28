

get_best_score_and_path = function(row, col, nucA, nucB, score_matrix, score_gap, score_match, score_mismatch, local) {
  paths <- matrix(c("diag","left","up", "-"),1,4)
  
  if(nucA ==nucB){
    m <- score_match
  }
  else{
    m <- score_match
  }
  
  if(local){
    scores <- matrix(c(score_matrix[row-1,col-1]))
  }
}  


vec <- matrix(c(1,2,3,4), 1,4)
max_vec <- max(vec)
which(vec == max_vec, arr.ind = TRUE)

path <- paths[ind[1],ind[2]]


#############







if (local == FALSE){
  
  
  # vec will be c(match, mismatch, gap, gap)
  # if (nucA == nucB){# MATCH
  #   scores[1,1] <- prev_pos + score_match
  #   scores[1,2] <- prev_pos + (-50)
  #   scores[2,1] <- prev_pos_left + (-40)
  #   scores[2,2] <- prev_pos_up + (-60)
  
  if
  
  
  
}
if (nucA != nucB){# MISMATCH and GAP
  vec_score[2] <- prev_pos + score_mismatch
  vec_score[3] <- prev_pos_left + score_gap
  vec_score[4] <- prev_pos_up + score_gap
  vec_score[1] <- prev_pos + (-70)
}

max_score <- max(vec_score)
max_position <- as.numeric(which.max(vec_score))

if(max_position == 1){ #MATCH
  best_path <- "diag"
  best_score <- vec_score[1]
}
if(max_position == 2){ #MISMATCH
  best_path <- "diag"
  best_score <- vec_score[2]
}
if(max_position == 3){ #GAP left
  best_path <- "left"
  best_score <- vec_score[3]
}
if(max_position == 4){ #GAP up
  best_path <- "up"
  best_score <- vec_score[4]
}

score <- best_score
path <- best_path
}



if(local == TRUE){ #Smith Waterman algorithm LOCAL, match - mismatch - gap - gap - 0
  scores <- matrix(c(prev_pos + score_match,
                     prev_pos + score_mismatch,
                     prev_pos_left + score_gap, 
                     prev_pos_up + score_gap, 
                     0),1,5)
  
  score <- max(scores)
  max_path <- which(scores == score, arr.ind = TRUE)
  path <- paths[max_path[1], max_path[2]]
  
}

return(list("score"=score, "path"=path))
}

get_best_score_and_path(2,2,"B","A",score_matrix,-2,3,-1, TRUE)
nucA <- "A"
nucB <- "B"

score_matrix <- init_score_matrix(5,5,TRUE, -1)
path_matrix <- init_path_matrix(5,5,TRUE)


#######
score_matrix<- matrix(c(-5,4,4,0),nrow = 2)

prev_pos <- score_matrix[1,1]
prev_pos_left <- score_matrix[2,1]
prev_pos_up <- score_matrix[1,2]

################

row = 2
newrow <- row-1




for(i in 5:2){
  print(i)
}


paste(month.abb, "is the", nth, "month of the year.")

paste0("A","B")

pa <- 2
pb <- 2

c(pa,pb) <- c(3,3)

max_score <- which(max(score_matrix)==score_matrix, arr.ind = T)
score_matrix
max_score


which(score_matrix > sort(score_matrix, TRUE)[15], arr.ind = TRUE)


max.col(score_matrix)

max_score <- max(score_matrix)
max_location <- which(score_matrix == max_score, arr.ind = TRUE)
xx <- as.numeric(max_location[-1,1])



max_score <- max(score_matrix)
max_location <- which(score_matrix == max_score, arr.ind = TRUE)
score <- max_score
pA <- as.numeric(max_location[-1,1])
pB <- as.numeric(max_location[-1,2])

x <- score_matrix[pA,pB]


typeof(score_matrix[pA,pB])


which(score_matrix == max_score, arr.ind = TRUE)



max_score <- max(score_matrix)
score <- max_score

max_loc <- which(score_matrix == max_score, arr.ind = TRUE)[1,]
pA <- as.numeric(max_loc[1])
pB <- as.numeric(max_loc[2])
pA



# pA <- nA+1
# pB <- nB+1
# score <- score_matrix[pA,pB]
# while(pA>1 || pB>1){
#   if (path_matrix[pA,pB] == "diag"){
#     aliA <- paste0(seqA[pA-1],aliA)
#     aliB <- paste0(seqB[pB-1],aliB)
#     pA <- pA-1
#     pB <- pB-1
#   } else if(path_matrix[pA,pB] == "left"){
#     aliA <- paste0("-",aliA)
#     aliB <- paste0(seqB[pB-1],aliB)
#     pA <- pA
#     pB <- pB-1
#   } else if(path_matrix[pA,pB] == "up"){
#     aliA <- paste0(seqA[pA-1],aliA)
#     aliB <- paste0("-",aliB)
#     pA <- pA-1
#     pB <- pB
#   }
# }
# alignment <- c(aliA,aliB)
# 





# if(path_matrix[pA,pB] == "diag"){
#   x <- get_best_move(seqA[pA], seqB[pB],)
#   aliA <- paste0(x$char1,aliA)
#   aliB <- paste0(seqB[pB-1],aliB)
#   pA <- x$newrow
#   pB <- x$newcol
# } else if(path_matrix[pA,pB] == "left"){
#   aliA <- paste0("-",aliA)
#   aliB <- paste0(seqB[pB-1],aliB)
#   pA <- pA
#   pB <- pB-1
# } else if(path_matrix[pA,pB] == "up"){
#   aliA <- paste0(seqA[pA-1],aliA)
#   aliB <- paste0("-",aliB)
#   pA <- pA-1
#   pB <- pB
# }







