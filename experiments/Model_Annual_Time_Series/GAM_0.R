getwd()
setwd("/Users/peteryatsyshin/Documents/GitHub/Turing_BAS/dobson-eq/experiments")

t <- read.csv(file = 't_ozone.csv')
x <- read.csv(file = 'x_ozone.csv' )

library(ggplot2)
library(plotly)

# standardise data
ggplotly(qplot(t$dt,scale(x$x), geom = c('line', 'point')))

         