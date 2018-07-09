HungarianCUDA: HungarianCUDA.cu
	nvcc -rdc=true --generate-code arch=compute_35,code=compute_35 HungarianCUDA.cu \-std=c++11 -o HungarianCUDA