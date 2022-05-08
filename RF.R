library(ranger)
library(tidyverse)

train_data <- read.csv("train.csv", header = TRUE) %>%
  select(-c("id"))

#plotting response variable

ggplot(train_data, aes(x=as.factor(target))) +
  geom_bar() +
  xlab("Response Variable") +
  theme(axis.text=element_text(size=12),
        axis.title=element_text(size=14,face="bold"))
ggsave("resp.png")

test_data <- read.csv("test.csv", header = TRUE) %>%
  select(-c("id"))

ntree <- 3500

inbag <- replicate(ntree,
                   {
                     bvar <- numeric(nrow(train_data))
                     indx <- c(sample(which(train_data$target==0), 70, replace = TRUE),
                               sample(which(train_data$target==1), 70, replace = TRUE))
                     bvar[indx]=1
                     bvar
                   }
                   , simplify = FALSE)

RF_mod <- ranger(target~., num.trees = ntree, mtry = 25, min.node.size =2,
                 data = train_data, classification = TRUE, replace = FALSE,
                 importance = "permutation", probability = TRUE, oob.error = TRUE,
                 inbag = inbag, splitrule = "gini", keep.inbag = TRUE) #case.weights = weights,


pred <- predict(RF_mod, data = test_data, type = "response")

results <- data.frame(id=250:19999, target=pred$predictions[,2])
colnames(results)=c("id","target")
write.csv(results, file = "submission.csv", row.names = F)

#######################################################################
ntree <- 5000

inbag <- replicate(ntree,
                   {
                     bvar <- numeric(nrow(train_data))
                     indx <- c(sample(which(train_data$target==0), 50, replace = TRUE),
                               sample(which(train_data$target==1), 50, replace = TRUE))
                     bvar[indx]=1
                     bvar
                   }
                   , simplify = FALSE)



## Or t-test
frmla <- as.formula(values ~ group)
t_pval <- sapply(3:299, 
                 function(var){
                   dat <- data.frame(group=train_data$target, values=train_data[,var])
                   frmla <- as.formula(frmla)
                   ttest <- t.test(frmla, data=dat)$p.value
                 }
)


ndat <- train_data[,c(FALSE, TRUE, t_pval<0.02)] %>%
  mutate(target=train_data$target, X17 = train_data$X17, X30 = train_data$X30,
         X16=train_data$X16, X101 = train_data$X101, X13 = train_data$X13,
         X45 = train_data$X45, X132 = train_data$X132, X82 = train_data$X82) %>%
  mutate(target = as.factor(target))

RF_mod <- ranger(target~., num.trees = ntree, mtry = 5, min.node.size =1,
                 data = ndat, classification = TRUE, replace = FALSE,
                 importance = "permutation", probability = TRUE, oob.error = TRUE,
                 inbag = inbag, splitrule = "gini", keep.inbag = TRUE) #case.weights = weights,


pred <- predict(RF_mod, data = test_data, type = "response")

results <- data.frame(id=250:19999, target=pred$predictions[,2])
colnames(results)=c("id","target")
write.csv(results, file = "submission.csv", row.names = F)

#######################################################################
oob_idx <- ifelse(simplify2array(RF_mod$inbag.counts) == 0, TRUE, NA)

preds <- predict(RF_mod, ndat, predict.all = TRUE)$predictions

probs1 <- oob_idx * preds[,1,] # use [,1,] for trained_data

x <- rep(NA, ntree)
y <- rep(NA, ntree)
z <- rep(NA, ntree)
for (i in 1:ntree) {
  # x[i] <- (table(probs1[,i],train_data$target)[2]/
  #         sum(table(probs1[,i],train_data$target)[1:2]))
  # 
  # y[i] <- 1-(sum(diag(table(probs2[,i],train_data$target)))/
  #            sum(table(probs2[,i],train_data$target)))
  
  x[i] <- table(probs1[,i],train_data$target)[2,1]/ # OOB: fraction of misclafd smpls
             sum(table(probs1[,i],train_data$target)[,1])
  
  y[i] <- table(probs1[,i],train_data$target)[1,2]/
             sum(table(probs1[,i],train_data$target)[,2])
  
  z[i] <- (table(probs1[,i],train_data$target)[2,1]+
             table(probs1[,i], train_data$target)[1,2])/
             sum(table(probs1[,i],train_data$target))
               
  
}

averg_err_x <- cummean(x)
averg_err_y <- cummean(y) 
averg_err_z <- cummean(z)

ymin <- min(min(averg_err_x), min(averg_err_y))
ymax <- max(max(averg_err_x), max(averg_err_y))


pdf(file = "ndat_OOB_plot.pdf")
plot(averg_err_x, type = "l", col="red", ylim = c(ymin, ymax+.05),
    ylab = "Error", xlab = "Trees", cex.lab=1.5, cex.axis=1, lwd=2)
points(averg_err_y, type = "l", col="green", lwd=2)
points(averg_err_z, type="l", col="black", lwd=2)
legend(x="topright", legend = c("OOB for 0", "OOB for 1", "OOB"), 
       col=c("green", "red", "black"), lty=1:2, cex=0.8,
       title="", text.font=4)
mtext("(c)",at=0.5, line = 0.2, cex= 1.5)
dev.off()


RF_mod_def <- ranger(target~.,data = train_data, classification = TRUE,
                 probability = TRUE, oob.error = TRUE, keep.inbag = TRUE)


# create hyperparameter grid
hyper_grid <- expand.grid(
  mtry = seq(5, 60, 10),
  #floor((length(train_data)-1) * c(.05, .08, .1 , .12, .15, .17, .20, .25, .3)),
  min.node.size = seq(1, 15, 5), 
  #replace = c(TRUE, FALSE),                               
  #sample.fraction = c(.3, .4, .5, .63),                       
  rmse = NA                                               
)

# execute full cartesian grid search
for(i in seq_len(nrow(hyper_grid))) {
  # fit model for ith hyperparameter combination
  fit <- ranger(
    formula         = target ~ ., 
    data            = train_data, 
    num.trees       = ntree,#floor((length(train_data)-1)) * 10
    mtry            = hyper_grid$mtry[i],
    min.node.size   = hyper_grid$min.node.size[i],
    replace         = hyper_grid$replace[i],
    classification = TRUE,
    probability = TRUE,
    #sample.fraction = hyper_grid$sample.fraction[i],
    verbose         = FALSE,
    #seed            = 123,
    #respect.unordered.factors = 'order',
    #case.weights = weights,
    inbag = inbag,
    splitrule =  "gini",
    oob.error = TRUE
  )
  # export OOB error 
  hyper_grid$rmse[i] <- sqrt(fit$prediction.error)
}

(default_rmse <- sqrt(RF_mod_def$prediction.error))
# assess top 10 models
hyper_grid %>%
  arrange(rmse) %>%
  mutate(perc_gain = (default_rmse - rmse) / default_rmse * 100) %>%
  head(10)

