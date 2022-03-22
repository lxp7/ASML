install.packages("readr")
install.packages("skimr")
install.packages("tidyverse")
install.packages("ggplot2")
install.packages("dplyr")
install.packages("rgeos")
install.packages("DataExplorer")
install.packages("data.table")
install.packages("mlr3verse")
install.packages("paradox")
install.packages("ranger")
install.packages("xgboost")
bank <- readr::read_csv("https://www.louisaslett.com/Courses/MISCADA/bank_personal_loan.csv")
skimr::skim(bank)
library("data.table")
library("mlr3verse")
library("tidyverse")

skimr::skim(bank)
bank$ZIP.Code  <- NULL
bank$Personal.Loan  <- as.factor(bank$Personal.Loan)
bank$Securities.Account   <- as.factor(bank$Securities.Account)
bank$CD.Account   <- as.factor(bank$CD.Account )
bank$Online  <- as.factor(bank$Online)
bank$CreditCard  <- as.factor(bank$CreditCard)
skimr::skim(bank)

DataExplorer::plot_bar(bank, ncol = 3)
DataExplorer::plot_histogram(bank, ncol = 3)
DataExplorer::plot_boxplot(bank, by = "Personal.Loan", ncol = 3)

set.seed(100)
bank_task <- TaskClassif$new(id = "bank",
                               backend = bank, # <- NB: no na.omit() this time
                               target = "Personal.Loan",
                             positive = "1")


lrn_baseline <- lrn("classif.featureless", predict_type = "prob")
lrn_cart <- lrn("classif.rpart", predict_type = "prob")
lrn_lda <- lrn("classif.lda", predict_type = "prob")
lrn_lr <- lrn("classif.log_reg", predict_type = "prob")

pl_missing <- po("fixfactors") %>>%
  po("removeconstants") %>>%
  po("imputesample", affect_columns = selector_type(c("ordered", "factor"))) %>>%
  po("imputemean")

pl_log_reg <- pl_missing %>>%
  po(lrn_lr)

cv5 <- rsmp("cv", folds = 5)
cv5$instantiate(bank_task)


res <- benchmark(data.table(
  task       = list(bank_task),
  learner    = list(lrn_baseline,
                    lrn_cart,
                    lrn_cart_cp,
                    lrn_lr,
                    pl_log_reg,
                    lrn_lda),
  resampling = list(cv5)
), store_models = TRUE)
res

res$aggregate(list(msr("classif.ce"),
                       msr("classif.acc"),
                       msr("classif.auc"),
                       msr("classif.fpr"),
                       msr("classif.fnr")))

learner$param_set
trees <- res$resample_result(2)

tree1 <- trees$learners[[1]]

tree1_rpart <- tree1$model

plot(tree1_rpart, compress = TRUE, margin = 0.1)
text(tree1_rpart, use.n = TRUE, cex = 0.8)

lrn_cart_cv <- lrn("classif.rpart", predict_type = "prob", xval = 10)

res_cart_cv <- resample(bank_task, lrn_cart_cv, cv5, store_models = TRUE)
rpart::plotcp(res_cart_cv$learners[[5]]$model)

lrn_cart_cp <- lrn("classif.rpart", predict_type = "prob", cp = 0.014, id = "cartcp")


lrn_ranger   <- lrn("classif.ranger", predict_type = "prob")
lrn_xgboost  <- lrn("classif.xgboost", predict_type = "prob")
lrnsp_log_reg <- lrn("classif.log_reg", predict_type = "prob", id = "super")

pl_missing <- po("fixfactors") %>>%
  po("removeconstants") %>>%
  po("imputesample", affect_columns = selector_type(c("ordered", "factor"))) %>>%
  po("imputemean")

pl_factor <- po("encode")

spr_lrn <- gunion(list(
  gunion(list(
    po("learner_cv", lrn_baseline),
    po("learner_cv", lrn_cart),
    po("learner_cv", lrn_cart_cp)
  )),
  pl_missing %>>%
    gunion(list(
      po("learner_cv", lrn_ranger),
      po("learner_cv", lrn_log_reg),
      po("nop")
    )),
  pl_factor %>>%
    po("learner_cv", lrn_xgboost)
)) %>>%
  po("featureunion") %>>%
  po(lrnsp_log_reg)
spr_lrn$plot()
res_spr <- resample(bank_task, spr_lrn, cv5, store_models = TRUE)
res_spr$aggregate(list(msr("classif.ce"),
                       msr("classif.acc"),
                       msr("classif.fpr"),
                       msr("classif.fnr")))

library()
