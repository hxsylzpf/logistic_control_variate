library(ggplot2)
setwd("/home/jbaker/code/logistic_container/logistic_control_variate/data/")

stepsize = "1e-05"
seed_list = 1:5
n_seeds = length( seed_list )
length_sample = length( read.table( paste0( "cover_type_sgld","/",stepsize,"/",1,"/logloss.dat" ) )$V1 )

lower_std = rep( NA, length_sample )
mean_std = rep( NA, length_sample )
upper_std = rep( NA, length_sample )

lower_cv = rep( NA, length_sample )
mean_cv = rep( NA, length_sample )
upper_cv = rep( NA, length_sample )

# Initialise storage
stdll = matrix( rep( NA, 5*length_sample ), ncol = 5 )
cvll = matrix( rep( NA, 5*length_sample ), ncol = 5 )

for ( seed_num in seed_list ) {
    stdll[,seed_num] = as.numeric( read.table( paste0( "cover_type_sgld","/",stepsize,"/",seed_num,"/logloss.dat") )$V1 )
    cvll[,seed_num] = as.numeric( read.table( paste0( "cover_type_sgld_cv","/",stepsize,"/",seed_num,"/logloss.dat") )$V1 )
}

for ( i in 1:length_sample ) {
    lower_std[i] = min( stdll[i,] )
    lower_cv[i] = min( cvll[i,] )
    mean_std[i] = mean( stdll[i,] )
    mean_cv[i] = mean( cvll[i,] )
    upper_std[i] = max( stdll[i,] )
    upper_cv[i] = max( cvll[i,] )
}

plot_frame = data.frame( iteration = rep( 1:length_sample*100, 2 ),
                         lower = c( lower_std, lower_cv ),
                         avg = c( mean_std, mean_cv ),
                         upper = c( upper_std, upper_cv ),
                         method = rep( c( "SGLD", "SGLD-CV" ), each = length_sample ))

p = ggplot( plot_frame, aes( x = iteration, y = avg, ymin = lower, ymax = upper, fill = method, color = method ) ) +
        geom_line() +
        geom_ribbon( alpha = 0.2 ) +
        scale_y_continuous( "Log loss" ) +
        scale_x_log10()
ggsave( "/home/jbaker/code/logistic_container/logistic_control_variate/plots/cover_cv.pdf", p, width = 8, height = 5 )
