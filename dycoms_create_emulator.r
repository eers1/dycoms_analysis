# Load libraries 
library(DiceKriging)
library(sensitivity)

# Define functions
validation_plot = function(model_data, predicted_data) {
  minXY <- min(predicted_data$mean, model_data)
  maxXY <- max(predicted_data$mean, model_data)
  minX  <- min(model_data)
  maxX  <- max(model_data)
  minY  <- min(predicted_data$mean) - max(predicted_data$sd)
  maxY  <- max(predicted_data$mean) + max(predicted_data$sd)
  rmse  <- sqrt(mean((model_data - predicted_data$mean)^2))
  main  <- sprintf("Validation of Emulator Model - rmse: %s", rmse)
  plot(model_data, predicted_data$mean, pch="", xlim=c(minX,maxX), ylim=c(minY,maxY), 
       xlab="MONC output", ylab="Emulator Prediction", main=main)
  LEQseq <- seq(minXY, maxXY, length=10)
  lines(LEQseq, LEQseq, col="darkgrey", lwd=2)
  for(it in 1:length(model_data)){
    lines(c(model_data[it], model_data[it]), c(predicted_data$lower95[it], predicted_data$upper95[it]), lwd=1.2, col=1)
  }
  points(model_data, predicted_data$mean, pch=20, col=1, cex=1.2)
  rm(minXY, maxXY, minX, maxX, minY, maxY, LEQseq, it)
}

krig.mean = function(Xnew,m) {
  mean_response_surface=predict.km(m,Xnew,'UK',se.compute=FALSE,checkNames=FALSE)$mean
  return(mean_response_surface)
}

################################################################################
#### Main ####
################################################################################
# Load data and create variables -----------------------------------------------
# File extensions
nd   <- "low"
calc <- "cloud_frac" # lwp_cloud cloud_frac
type <- "mean"       # mean teme
noise_type <- "unnormal_2_ungrouped"

# indices of the ensembles that are being used:
iterator<-list(2,5)     # list(0,1,2,3,4,5,6,7,8)

# Load data
data_path  <- sprintf("data_%s/dycoms_data_%s_nd_%s_%s.csv", calc, nd, calc, type)
DycomsData <- read.csv(data_path,header=FALSE,stringsAsFactors=FALSE)

# Create parameter table
Min <- c(2,-9)
Max <- c(20,0)
Parameter_Name    <- c("theta","q_t")
DesignRangesTable <- data.frame(Parameter_Name,Min,Max)
rm(Parameter_Name,Min,Max)

# Create input vector
validation_input <- DycomsData[25:32,1:2]
if (noise_type=="exact") {
  training_input <- rbind(DycomsData[1:20,1:2])
} else {
  training_input <- rbind(DycomsData[1:20,1:2],DycomsData[33:38,1:2])
}

# Make unit
training_input[,1] <- (training_input[,1] - DesignRangesTable$Min[1])/(DesignRangesTable$Max[1] - DesignRangesTable$Min[1])
training_input[,2] <- (training_input[,2] - DesignRangesTable$Min[2])/(DesignRangesTable$Max[2] - DesignRangesTable$Min[2])
validation_input[,1] <- (validation_input[,1] - DesignRangesTable$Min[1])/(DesignRangesTable$Max[1] - DesignRangesTable$Min[1])
validation_input[,2] <- (validation_input[,2] - DesignRangesTable$Min[2])/(DesignRangesTable$Max[2] - DesignRangesTable$Min[2])

n_training   <- length(training_input[,1])
n_validation <- length(validation_input[,1])

# Create output vector
validation_output <- DycomsData[25:32,3]
if (noise_type=="exact") {
  training_output <- c(DycomsData[1:20,3])
} else {
  training_output <- c(DycomsData[1:20,3],DycomsData[33:38,3])
}

# Plot to check designs --------------------------------------------------------
# Plot design in 2D space
pairs(rbind(training_input, validation_input),
      pch=20, upper.panel=NULL,
      col=c(rep(1,times=n_training), rep(2,times=n_validation)))
# ... as histograms
par(mfrow=c(2,2))
hist(training_input[,1],breaks=10,xlab=DesignRangesTable$Parameter_Name[1],
     main=paste("Histogram for ",DesignRangesTable$Parameter_Name[1],sep=""))
hist(training_input[,2],breaks=10,xlab=DesignRangesTable$Parameter_Name[2],
     main=paste("Histogram for ",DesignRangesTable$Parameter_Name[2],sep=""))
# Plot 2D inputs vs output
par(mfrow=c(1,1))
pairs(cbind(rbind(training_input, validation_input),lwp=c(training_output, validation_output)),
      pch=20,upper.panel=NULL,col=c(rep(1,times=n_training),rep(2,times=n_validation)))

# Deal with ensembles and noise vectors ----------------------------------------
# If using initial-condition ensembles, replace the training data at the 
# ensemble points with the mean
if (noise_type!="exact" & noise_type!="extras" & noise_type!="1mag" & noise_type!="2mag" & noise_type!="trial") {
  indices<-c(3,9,11,14,15,17,18,19,20)
  
  ensembleData<-read.csv(sprintf("data_%s/ensemble_%s_mean.csv", calc, calc),
                         header=FALSE,stringsAsFactors=FALSE)
  if (type=="mean") {
      ensemble<-ensembleData[,1]
  } else {
      ensemble<-ensembleData[,2]
  }
  
  for (j in iterator) {
    training_output[indices[j+1]]<-mean(ensemble[(j*5+1) : (j*5+5)])
  }
}

# Load noise vector 
rm(NV)
if (noise_type!="exact" & noise_type!="extras") {
  noise_read = sprintf("noise_files/nv_0424_%s_%s_%s.csv", 
                                            calc, type, noise_type)
  noise_vector<-read.csv(noise_read,header=FALSE,stringsAsFactors=FALSE)
  NV<-c(noise_vector[1:20,1],noise_vector[33:38,1])
  NV
}

# Emulate ----------------------------------------------------------------------
if (noise_type=="exact" | noise_type=="extras") {
  EmModel <- km(formula=~., design=training_input, response=training_output,
              covtype="matern5_2", nugget.estim=FALSE, optim.method="BFGS", control=list(maxit=500))
} else if (noise_type=="r_nug") {
  EmModel <- km(formula=~., design=training_input, response=training_output,
                covtype="matern5_2", nugget.estim=TRUE, optim.method="BFGS", control=list(maxit=500))
} else {
  EmModel <- km(formula=~., design=training_input, response=training_output,
                covtype="matern5_2", noise.var=NV, optim.method="BFGS", control=list(maxit=500))
}

# Validate ---------------------------------------------------------------------
validation_predictions <- predict(object=EmModel, newdata=data.frame(validation_input),
                                  type="UK", checkNames=FALSE, light.return=TRUE)

validation_plot(validation_output, validation_predictions)

# Predict other values ---------------------------------------------------------
# Predict at a set of grid values
grid_values <- read.csv("misc/sample_vals.csv", header=FALSE, stringsAsFactors=FALSE)
grid_values[,1] <- (grid_values[,1] - DesignRangesTable$Min[1])/(DesignRangesTable$Max[1] - DesignRangesTable$Min[1])
grid_values[,2] <- (grid_values[,2] - DesignRangesTable$Min[2])/(DesignRangesTable$Max[2] - DesignRangesTable$Min[2])

grid_predictions <- predict(object=EmModel, newdata=data.frame(grid_values),
                            type="UK", checkNames=FALSE, light.return=TRUE)

# Predict at training points (for when a nugget term is added)
training_predictions <- predict(object=EmModel, newdata = data.frame(training_input), 
                                type="UK", checkNames=FALSE, light.return=TRUE)

# Write to file ---------------------------------------------------------------
saveRDS(EmModel,
        file=sprintf("emulators/%s/%s_%s_%s", calc, calc, type, noise_type))

write.csv(validation_predictions,
          sprintf("predictions/%s/pre_val_%s_%s_%s.csv", calc, calc, type, noise_type),
          row.names=FALSE,quote=FALSE)

write.csv(grid_predictions,
          sprintf("predictions/%s/pre_tot_%s_%s_%s.csv", calc, calc, type, noise_type),
          row.names=FALSE,quote=FALSE)

write.csv(training_predictions,
          sprintf("predictions/%s/pre_design_%s_%s_%s.csv", calc, calc, type, noise_type),
          row.names=FALSE,quote=FALSE)

 # # Sensitivity analysis ---------------------------------------------------------
# distlist = rep('qunif',2)
# SA.model = fast99(model=krig.mean,factors=2,n=1000,q=(distlist),q.arg=list(min=0,max=1),m=EmModel)
# sample_direct = SA.model$y
# par(mfrow=c(1,1))
# plot(SA.model)
# 
# sa_main  = SA.model$D1/SA.model$V
# sa_total = 1-SA.model$Dt/SA.model$V
# 
# write.csv(sa_main,"predictions/sa_analysis/sa_main_cf_teme.csv",row.names=FALSE,quote=FALSE)
# write.csv(sa_total,"predictions/sa_analysis/sa_total_cf_teme.csv",row.names=FALSE,quote=FALSE)
