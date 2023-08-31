tests_dir <- "/Users/johno/Documents/University/ETH HS22/Computational Biology/HW/HW05/student_test_suite"
if(tests_dir == "path/to/tests") stop("tests_dir needs to be set to a proper path")

library("RUnit")

original_dir <- getwd()
setwd(tests_dir)

testsuite <- defineTestSuite("HW", ".")

out <- runTestSuite(testsuite)
printTextProtocol(out)

setwd(original_dir)
