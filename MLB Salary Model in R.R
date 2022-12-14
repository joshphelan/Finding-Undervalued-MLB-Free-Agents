# Model with 2015-2021 data (excluding 2020)
batting = read.csv('Batting Dataframe 2015-2021.csv')
batting <- subset(batting, select = -c(X, Name.additional))

none <- lm(batting$Salary ~ 1,data=batting)
all <- lm (batting$Salary ~.,data = batting)
backward <- step(all, direction='backward', scope=formula(all), trace=0)
forward <- step(none, direction='forward', scope=formula(all), trace=0)
summary(forward)
summary(backward)

# Forward: adj r^2 = .6756 on 22 x, Backward: .6788 on 59 x

library(car)
vif(forward)

# OPS_Career has VIF of 24 and is not significant

forward_new <- lm(formula = batting$Salary ~  CS_Career + Rpos + 
     IBB_Career + Rrep + HBP_Career + SO_Career + ISO + Age + 
     G + Year + OPS._Career + RBI_dummy + TB_dummy + LD. + OPS_dummy + 
     IBB + Rbaser + R + H_dummy + WPA + RBI, data = batting)
summary(forward_new)
vif(forward_new)

# RBI has VIF of 5.8 and is not significant

forward_new <- lm(formula = batting$Salary ~  CS_Career + Rpos + 
                    IBB_Career + Rrep + HBP_Career + SO_Career + ISO + Age + 
                    G + Year + OPS._Career + RBI_dummy + TB_dummy + LD. + OPS_dummy + 
                    IBB + Rbaser + R + H_dummy + WPA, data = batting)
summary(forward_new)
vif(forward_new)

# R has VIF of 6.26 and is not significant
forward_new <- lm(formula = batting$Salary ~  CS_Career + Rpos + 
                    IBB_Career + Rrep + HBP_Career + SO_Career + ISO + Age + 
                    G + Year + OPS._Career + RBI_dummy + TB_dummy + LD. + OPS_dummy + 
                    IBB + Rbaser + H_dummy + WPA, data = batting)
summary(forward_new)
vif(forward_new)

# removing WPA
forward_new <- lm(formula = batting$Salary ~  CS_Career + Rpos + 
                    IBB_Career + Rrep + HBP_Career + SO_Career + ISO + Age + 
                    G + Year + OPS._Career + RBI_dummy + TB_dummy + LD. + OPS_dummy + 
                    IBB + Rbaser + H_dummy, data = batting)
summary(forward_new)
library(car)
vif(forward_new)


# Library for lasso regression
library(glmnet)

y <- batting$Salary
x <- data.matrix(batting[,-c(59)]) # removing salary from x variables
#k-fold cross validation
cv_model <- cv.glmnet(x,y,alpha=1)

# Finding and plotting optimal lambda
best_lambda <- cv_model$lambda.min
best_lambda
plot(cv_model)

# Creating model with optimal lambda
best_model <- glmnet(x,y,alpha =1, lambda = best_lambda)
coef(best_model)
summary(best_model)

#r^2 value
best_model$dev.ratio

# Creating a list of the coefficients from the best model
coef <- coef(best_model)
dimnames(coef)
coefnum = coef@i +1
predictors <- coef@Dimnames[[1]][coefnum]
predictors = predictors[2:38]

# linear regression model based on coefficients
lmlasso <- lm(formula = batting$Salary ~ rOBA + HardH. + LD. + Age + G + R + SB + SLG
              + HBP + IBB + Rbaser + Rdp + Rpos + Rrep + dWAR + CS_Career + BB_Career +
                SO_Career + BA_Career + OPS_Career + OPS._Career + HBP_Career + IBB_Career+
                X2+X3+X5+X6+X8+HR_dummy+RBI_dummy+X3B_dummy+TB_dummy+WAR_dummy+OPS_dummy+H_dummy
              +AB_dummy+Year, data = batting)
summary(lmlasso)
vif(lmlasso)

# Lasso: adj r^2 = .6899 on 37 x

# Removing high collinearity and insignificant variables

lmlasso_new <- lm(formula = batting$Salary ~  + LD. + Age + G + R 
                + IBB + Rbaser + Rpos + Rrep + CS_Career +
                SO_Career  + OPS._Career + HBP_Career + IBB_Career+
                X3+X5+HR_dummy+RBI_dummy+TB_dummy+WAR_dummy+OPS_dummy+H_dummy
                +Year, data = batting)
summary(lmlasso_new)
vif(lmlasso_new)