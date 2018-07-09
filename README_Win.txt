In order to run this code a Compute Capability 3.5 capable NVIDIA GPU is required.

To compile and run this code in Windows follow the following steps:

1 - Create a CUDA project
2 - Remove the main file and add HungarianCUDA.cu to the project.
3 - Change the configuration to x64 instead of x86
4 - In the project propertie change "CUDA C/C++ \ Device \ Code Generation" to compute_35,sm_35
5 - Run the code pressing the play button.

If you want to use dynamic parallelism you also need to:

6 - Uncomment the corresponding line in HungarianCUDA.cu
7 - In the project properties change CUDA C/C++ \ Common \ Generate Relocatable Device Code" to yes
8 - Add the following file to the "Linker \ Input \ Additional Dependencies": cudadevrt.lib