

linear_regression(updated_description$traits[1:4,1], updated_description$traits[1:4,2])



linear_regression(updated_description$normalized_contrasts[5:7,1], updated_description$normalized_contrasts[5:7,2])

mod <- lm(updated_description$normalized_contrasts[,1] ~ updated_description$normalized_contrasts[,2])
summary(mod)
