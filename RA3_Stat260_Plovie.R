
#############Practice Questions Before Assignment#############

#Single mean: t.test(x, alternative, mu, conf.level)
  #Only need to specify vector x and conf.level
#Two means: t.test(x, y, alternative, mu, conf.level, var.equal)


### Creating a confidence interval, single mean ###
#Must specify x, conf.level

w.bolt <- c(12.3, 12.5, 12.7, 12.1, 12.6) #Vector of bolt weights
t.test(w.bolt,conf.level=0.97) #Creating a 97% CI
#[12.08483, 12.79517] is the CI from the output (second half)
#Mean of x = 12.44 from output



### Testing Alternative Hypothesis ###

#Must specify vector x, hypothesized mu, and alternative
#alternative = two.sided, less, OR greater

#Test true mean of bolts is less than 13
#H0: mu=13. H1: mu<13
t.test(w.bolt, mu=13, alternative="less")
#Output tells us:
  #t = -5.1995 = observed value of test statistic
  #df = 4 = degrees of freedom
  #p-value = 0.003259 = p-value for Hypothesis test (which is <0.1)
#Note: R is NOT telling you whether or not H0 should be rejected



### CI and Hyp Tests for Two Means ###

#Pooled: var.equal=TRUE
#Unpooled: var.equal=FALSE

#Creating 99% CI for mu1, mu2 given
c.1 <- c(4.1,4.3,5.1,5.2,5.4)
c.2 <- c(6.1,4.3,7.9,2.8,5.1,4.7)
sd(c.1)
sd(c.2)
sd(c.2)/sd(c.1)
#If sd2/sd1 >= 1.4, use UNPOOLED. Else use POOLED. 
#Make sure to set var.equal accordingly.

t.test(c.1, c.2, conf.level=0.99, var.equal=FALSE)
#99% CI is [-3.061548, 2.401548]



### Another similar example ###

#H0: mu1-m2=0, H1: mu1-mu2!=0
#Unpooled, var.equal=FALSE
#mu=0
#alternative="two.sided"

t.test(c.1, c.2, mu=0, alternative="two.sided", var.equal=FALSE)
#From output:
  #t_obs = -0.43919
  #p-value = 0.6752 found with df=6.3028 degrees of freedom
  #Little to no evidence against H0
  #ie. data are consistent with two means being equal



### Paired t-test ###

#Command: t.test(x1, x2, mu =, alternative =, conf.level =, paired = TRUE)
#Where xD = x1 - x2

Appr1 <- c(22.10, 92.70, 2.76, 75.60, 4.13)
Appr2 <- c(21.30, 92.10, 1.54, 78.90, 4.78)
t.test(Appr1, Appr2, mu=0, alternative="two.sided", paired=TRUE, conf.level=0.99)
#From output:
  #xD = -0.266 = mean of differences
  #NOTE: Use t_obs and xD to find ese using formula from notes




#############Assignment3#############


###########Question 1#############
soup <- c(510, 520, 515, 516, 517, 519, 522, 513)

#a)
t.test(soup, conf.level=0.96) #Creating 96% CI
#96% confidence interval is [513.0374,519.9626]

#b)
#Since 515 lies within the confidence interval, it is a reasonable estimate for mu.



###########Question 2#############
slabs <- c(35.1, 34.4, 35.8, 36.1, 37.7)

#a)
t.test(slabs, mu=35, alternative="greater")

#b)
#From output in part a), observed test statistic is t_obs = 1.479

#c)
#From output in part a), p-value = 0.1066

#d)
#Since p-value=0.1066 > alpha=0.01, we fail to reject H0 



###########Question 3#############
brand1 <- c(580, 592, 588, 589, 581)
brand2 <- c(579, 582, 577, 591, 583)

#a)
sd(brand1)
sd(brand2)
sd(brand2)/sd(brand1)
#Given that sd2/sd1 < 1.4, we will use a pooled procedure, var.equal=TRUE

#b)
#H0: mu1-mu2=0, H1: mu1-mu2!=0
#Pooled, var.equal=TRUE
#alternative="two.sided"
t.test(brand1, brand2, mu=0, alternative="two.sided", var.equal=TRUE)

#c)
#From output in part b), p-value = 0.3146

#d)
#Since p-value=0.3146 > 0.1, there is little to no evidence against H0



###########Question 4#############
pre <- c(130,103,116,113,124,122,128,124,123,108,134,108,111,129,134)
post <- c(134,106,110,115,122,126,130,118,125,110,138,111,115,125,130)

#a)
t.test(pre, post, conf.level=0.95, paired=TRUE)
#From output, confidence interval for 95th percentile is [-2.635481, 1.568815]

#b)
t.test(pre, post, mu=0, alternative="two.sided", paired=TRUE)
#Since the p-value=0.8834 > 0.1, there is little to no evidence suggesting an effect on blood pressure

#c)
#t_obs = -0.54415

#d)
#The test statistic has a t-distribution, with df=14 degrees of freedom

#e)
#p-value = 0.5949


#Question 5 and 6 do not have any code to *produce*. Modified from given code.













