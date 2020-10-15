from Tools import ImageOpenCV,Classify_all_simulations,ReplicaAnalysis

# Example of image Boolean classifier (b=0.5, F_r = 50%, and T_p = 50h) 
print(ImageOpenCV("output/Output_B05_F050_T050.jpg"))

# Analysis of 20 replicates for b=0.5, F_r = 50%, and T_p = 50h (Folder \OneCase)
ReplicaAnalysis()

# Classifying all simulations from folder \output
Classify_all_simulations()


 