plot3D_proportions <- function(directory, 
                               celltypes,
                               celltype_colors,
                               um=c(1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1),
                               axis_rescale=c(1,1,1),
                               spot_radius=0.5,
                               alpha_threshold=0.2,
                               alpha_background=0.02){

    #load cell-type proportions and 3D coordinates of spots
    file_list <- list.files(path = directory)
    n_slice <- sum(unlist(lapply(file_list, function(x){startsWith(x, "prop_slice")})))
    #cell-type proportions
    for (i in (0:(n_slice-1))){
        prop <- read.table(paste0(directory, "/prop_slice", i, ".csv"), sep=",", header=TRUE)
        if (i == 0){
            prop_all <- prop
        }else{
            prop_all <- rbind(prop_all, prop)
        }
    }
    colnames(prop_all)[1] <- "spot"
    prop_all <- prop_all[, c("spot", celltypes)]
    #3D coordinates
    coor_3d <- read.table(paste0(directory, "/3D_coordinates.csv"), sep=",", header=TRUE)
    colnames(coor_3d)[1] <- "spot"

    spots.table <- merge(coor_3d, prop_all, by=c("spot"))


    #set colors and alpha values for spots
    if (length(celltypes) > 1){
        prop <- spots.table[, celltypes]
    }else{
        prop <- data.frame(ct = spots.table[, celltypes])
        colnames(prop) <- celltypes
    }
    prop$max_prop <- apply(prop, 1, max)
    prop$max_celltype <- apply(prop, 1, function(x){celltypes[which.max(x)]})
    
    prop$color <- "gray"
    prop$alpha <- prop$max_prop

    #for others
    prop$alpha[prop$max_prop <= alpha_threshold] <- alpha_background

    #for target cell types
    for (c in 1:length(celltypes)){
        prop$color[(prop$max_prop > alpha_threshold) & (as.vector(prop$max_celltype) == celltypes[c])] <- celltype_colors[c]
    }


    #3D plot
    open3d(windowRect = c(0, 0, 720, 720))
    par3d(persp)
    view3d(userMatrix = matrix(um, byrow=TRUE, nrow=4))
    spheres3d(spots.table$x*axis_rescale[1], spots.table$y*axis_rescale[2], spots.table$z*axis_rescale[3],
              col = prop$color, radius=spot_radius, 
              alpha = prop$alpha)
    decorate3d()
}


plot3D_clusters <- function(directory, 
                            clusters,
                            cluster_colors,
                            um=c(1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1),
                            axis_rescale=c(1,1,1),
                            spot_radius=0.5,
                            alpha_threshold=0.2,
                            alpha_background=0.02){

    #load cluster assignments and 3D coordinates of spots
    cluster_df <- read.table(paste0(directory, "/clustering_result.csv"), sep=",", header=TRUE)
    colnames(cluster_df) <- c("spot", "cluster")
    coor_3d <- read.table(paste0(directory, "/3D_coordinates.csv"), sep=",", header=TRUE)
    colnames(coor_3d)[1] <- "spot"

    spots.table <- merge(coor_3d, cluster_df, by=c("spot"))
    spots.table$cluster <- as.integer(spots.table$cluster)

    #3D plot
    open3d(windowRect = c(0, 0, 720, 720))
    par3d(persp)
    view3d(userMatrix = matrix(um, byrow=TRUE, nrow=4))
    for (c in 1:length(clusters)){
        spheres3d(spots.table[spots.table$cluster==clusters[c], ]$x*axis_rescale[1], 
                  spots.table[spots.table$cluster==clusters[c], ]$y*axis_rescale[2], 
                  spots.table[spots.table$cluster==clusters[c], ]$z*axis_rescale[3],
                  col = cluster_colors[c], radius=spot_radius, 
                  alpha=1)
    }
    spheres3d(spots.table$x*axis_rescale[1], spots.table$y*axis_rescale[2], spots.table$z*axis_rescale[3],
              col='gray', radius=spot_radius, 
              alpha=alpha_background)
    decorate3d()
}
