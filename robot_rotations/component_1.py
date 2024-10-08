import numpy as np
import math
from utils import create_se2_matrix, create_se3_matrix
#region check_SOn

#Test Inputs
theta = math.pi / 4

rotation_matrix_2D = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)]
])

rotation_matrix_3D_aroundX = np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta),  np.cos(theta)]
    ])
#Function
def check_SOn(matrix: np.array, epsilon: float=0.01) -> bool:
    if (matrix.shape[0] != matrix.shape[1]):
        return False
    if not (np.allclose(np.matmul(matrix.T, matrix), np.identity(matrix.shape[0]), atol=epsilon)):
        return False
    if not (np.isclose(np.linalg.det(matrix), 1, atol=epsilon)):
        return False
    return True
#Testing
# print(check_SOn(rotation_matrix_2D))
# print(check_SOn(rotation_matrix_3D_aroundX))

#endregion

#region check_quaternion

quaternion_vector = np.array([
    [0],[1],[0],[0]
])

quaternion_vector2 = np.array([
    [0.5],[0.5],[0.5],[0.5]
])
#Function
def check_quaternion(vector: np.array, epsilon: float=0.01) -> bool:
    if vector.shape != (4,1): #Specific for s3
        return False
    sum_of_squares = np.sum(np.square(vector))
    return math.isclose(sum_of_squares, 1, abs_tol=epsilon)

#Testing
# print(check_quaternion(quaternion_vector))
# print(check_quaternion(quaternion_vector2))

#endregion

#region check_SEn

#Function
def check_SEn(matrix: np.array, epsilon: float=0.01) -> bool:
    if (matrix.shape[0] != matrix.shape[1]):
        return False
    n = matrix.shape[0] - 1

    R = matrix[:n, :n]
    t= matrix[:n, n]
    bot = matrix[n, :]

    is_R_correct = check_SOn(R)
        
    is_bot_correct = np.allclose(bot, [0] * n + [1], atol=epsilon)
    
    return is_R_correct and is_bot_correct
#Testing
SE2_matrix = create_se2_matrix(math.pi / 2, 2, 3)
SE3_matrix = create_se3_matrix(math.pi/2, math.pi/4, math.pi, 1, 2, 3)

# print(check_SEn(SE2_matrix))
# print(check_SEn(SE3_matrix))

#endregion