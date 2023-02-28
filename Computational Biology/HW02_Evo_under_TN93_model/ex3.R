
beta = 0.035
alpha1 = 0.044229
alpha2 = 0.021781
pi = c(0.22, 0.26, 0.33, 0.19)
  
Q <- create_TN93_Q_matrix(pi, alpha1, alpha2, beta)

for (t in seq(1, 10^3, 50)){
  max_diff <- max(abs(expm(t*Q) - expm(2*t*Q)))
  thresh <- 10^-7
  if (max_diff < thresh){
    print(t)
    break
  }
}


