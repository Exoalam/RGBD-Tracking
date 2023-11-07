import numpy as np
from transforms3d.axangles import axangle2aff
from transforms3d.affines import compose

translation = (1, 2, 3)  
axis = (0, 1, 0)         
angle = np.radians(45)   

rotation_matrix = axangle2aff(axis, angle)
affine_matrix = compose(T=translation, R=rotation_matrix[:3, :3], Z=np.ones(3))

point = np.array([4, 5, 6, 1])
transformed_point = affine_matrix.dot(point)

print(transformed_point[:3])
