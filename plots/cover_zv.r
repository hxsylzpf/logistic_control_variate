library(ggplot2)
setwd("/home/jbaker/code/logistic_container/logistic_control_variate/data/simulations")

stepsize_list = c("1e-06", "1e-05", "3e-05", "7e-05", "0.0001")
seed_list = 1:5
n_runs = length( stepsize_list )
n_seeds = length( seed_list )
length_sample = length( read.table( paste0( stepsize_list[1],"/",1,"/old.dat" ) )$V1 )

lower_old = rep( NA, n_runs )
mean_old = rep( NA, n_runs )
upper_old = rep( NA, n_runs )

lower_new = rep( NA, n_runs )
mean_new = rep( NA, n_runs )
upper_new = rep( NA, n_runs )

# Initialise storage
avg_over_seed = rep( NA, n_seeds )
oldll = matrix( rep( NA, 5*length_sample ), ncol = 5 )
newll = matrix( rep( NA, 5*length_sample ), ncol = 5 )

for ( i in 1:n_runs ) {
    step = stepsize_list[i]
    for ( seed_num in seed_list ) {
        oldll[,seed_num] = as.numeric( read.table( paste0(step,"/",seed_num,"/old.dat") )$V1 )
        newll[,seed_num] = as.numeric( read.table( paste0(step,"/",seed_num,"/new.dat") )$V1 )
    }
    for ( j in seed_list ) {
        avg_over_seed[j] = quantile( oldll[,j], 0.05 )
        lower_old[i] = mean( avg_over_seed )
    }
    for ( j in seed_list ) {
        avg_over_seed[j] = mean( oldll[,j] )
        mean_old[i] = mean( avg_over_seed )
    }
    for ( j in seed_list ) {
        avg_over_seed[j] = quantile( oldll[,j], 0.95 )
        upper_old[i] = mean( avg_over_seed )
    }
    for ( j in seed_list ) {
        avg_over_seed[j] = quantile( newll[,j], 0.05 )
        lower_new[i] = mean( avg_over_seed )
    }
    for ( j in seed_list ) {
        avg_over_seed[j] = mean( newll[,j] )
        mean_new[i] = mean( avg_over_seed )
    }
    for ( j in seed_list ) {
        avg_over_seed[j] = quantile( newll[,j], 0.95 )
        upper_new[i] = mean( avg_over_seed )
    }
}

plot_frame = data.frame( stepsize = rep( as.numeric( stepsize_list ), 2 ),
                         lower = c( lower_old, lower_new ),
                         avg = c( mean_old, mean_new ),
                         upper = c( upper_old, upper_new ),
                         method = rep( c( "SGLD", "SGLD-ZV" ), each = n_runs ))

dir.create("/home/jbaker/code/logistic_container/logistic_control_variate/plots", showWarnings = FALSE)
p = ggplot( plot_frame, aes( x = stepsize, y = avg, ymin = lower, ymax = upper, fill = method, color = method ) ) +
        geom_line() +
        geom_ribbon( alpha = 0.2 ) +
        scale_y_continuous( "Log loss" ) +
        scale_x_log10()
ggsave( "/home/jbaker/code/logistic_container/logistic_control_variate/plots/simulation.pdf", p, width = 8, height = 5 )
