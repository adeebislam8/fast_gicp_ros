Originally forked from https://github.com/anjulo/scan_matching_localizer
The above implementation had PCL icp and NDT implementation. Follow the above link to install the dependencies for those 2 scan matching methods
https://github.com/koide3/ndt_omp

I changed the implementations of the forked package to publish the proper transform to ros tf and to publish the odometry
Additionally added the fgicp scan-matching algorithm which performed better than icp and ndt
https://github.com/SMRT-AIST/fast_gicp
