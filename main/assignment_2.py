import numpy as np
from numpy.linalg import inv

def nevilles_method(x_points, y_points, x):

    matrix = np.zeros((len(x_points), len(y_points)))
    # Create array
    for counter, row in enumerate(matrix):
        row[0] = y_points[counter]
    # Fill up the first row with the y point
    num_of_points = len(x_points)
    
    #Fill up the matrix using the functions for the method
    for i in range(1, num_of_points):
        for j in range(1, i+1):
            first_multiplication = (x - x_points[i-j]) * matrix[i][j-1]
            second_multiplication = (x - x_points[i]) * matrix[i-1][j-1]
            denominator = x_points[i] - x_points[i-j]
            # this is the value that we will find in the matrix
            coefficient = (first_multiplication - second_multiplication)/(denominator)
            matrix[i][j] = coefficient
    
    #We need to print the 2nd degree value, which is 2,2
    print(matrix[2][2])
    
    return None


def divided_difference_table(x_points, y_points):

    # set up the matrix
    size: int = len(x_points)
    matrix = np.zeros((size, size))
    # fill the matrix
    for index, row in enumerate(matrix):
        row[0] = y_points[index]
    
    # populate the matrix (end points are based on matrix size and max operations we're using)
    for i in range(1, size):
        for j in range(1, i+1):
            # the numerator are the immediate left and diagonal left indices...
            numerator = (matrix[i][j-1] - matrix[i-1][j-1])
            # the denominator is the X-SPAN...
            denominator = (x_points[i] - x_points[i-j])
            operation = numerator / denominator
            # cut it off to view it more simpler
            matrix[i][j] = (operation)

    return matrix

def get_approximate_result(matrix, x_points, value):
    # p0 is always y0 and we use a reoccuring x to avoid having to recalculate x 
    reoccuring_x_span = 1
    reoccuring_px_result = matrix[0][0]
    
    
    # we only need the diagonals...and that starts at the first row...
    for index in range(1, len(x_points)):
        polynomial_coefficient = matrix[index][index]
        # we use the previous index for x_points....
        reoccuring_x_span *= (value - x_points[index-1])
        
        # get a_of_x * the x_span
        mult_operation = polynomial_coefficient * reoccuring_x_span
        # add the reoccuring px result
        reoccuring_px_result += mult_operation
        
    
    # final result
    return reoccuring_px_result

def apply_div_dif(matrix: np.array):
    size = len(matrix)
    for i in range(2, size):
        for j in range(2, i+2):
            # skip if value is prefilled or if exceeds bounds
            if j >= len(matrix[i]) or matrix[i][j] != 0:
                continue
            
            #Get left
            left: float = matrix[i][j-1]
            #Get right
            diagonal_left: float = matrix[i-1][j-1]
            #Numerator is left - diagonal left
            numerator: float = left - diagonal_left
            #Denominator is as follows...
            denominator = (matrix[i][0] - matrix[i-j+1][0])
            #Divide numerator by denominator and save into the matrix
            operation = numerator / denominator
            matrix[i][j] = operation
    
    return matrix

def hermite_interpolation():
    #Declare inputs
    x_points = [3.6, 3.8, 3.9]
    y_points = [1.675, 1.436, 1.318]
    slopes = [-1.195, -1.188, -1.182]
   
    num_pts = len(x_points)
    matrix = np.zeros((2*num_pts, 2*num_pts))
    #Fill x values
    for x in range(2 * num_pts - 1):
        matrix[x][0] = x_points[int(x/2)]
        matrix[x + 1][0] = x_points[int(x/2)]
        x += 1
    
    #Fill y values
    for x in range(2 * num_pts - 1):
        matrix[x][1] = y_points[int(x/2)]
        matrix[x + 1][1] = y_points[int(x/2)]
        x += 1


    #Fill derivatives
    for x in range(num_pts):
        matrix[(x*2) + 1][2] = slopes[int(x)]

    #Now utilize the divided difference method 
    filled_matrix = apply_div_dif(matrix)

    print(filled_matrix, "\n")

    return None

def matrixCalculator():
    #Create the matrix and fill in the static values
    matrix = np.zeros((4, 4))

    matrix[0][0] = 1
    matrix[3][3] = 1

    #Define the x and y points
    x_points = [2,5,8,10]
    y_points = [3,5,7,9]
    h = np.zeros(3)

    #Determine h(x) at each point
    for x in range(3):
        h[x] = (x_points[x+1] - x_points[x])
    
    #Do the math to fill in the respective places within the matrix
    for x in range(2):
        for y in range(3):
            if (y == 0):
                matrix[x+1][y+x] = h[x]
            elif (y == 1):
                matrix[x+1][y+x] = 2*(h[x] + h[x+1])
            else:
                matrix[x+1][y+x] = h[x+1]

    #Print result
    print(matrix, "\n")

    #Solve for vector b and print it out
    b = np.zeros(4)
    b[0] = 0
    b[3] = 0
    b[1] = (3/h[1])*(y_points[2]-y_points[1]) - (3/h[0])*(y_points[1]-y_points[0])
    b[2] = (3/h[2])*(y_points[3]-y_points[2]) - (3/h[1])*(y_points[2]-y_points[1])

    print(b, "\n")

    #Multiply both sides by the inverse of the matrix
    inverse = inv(matrix)
    xVector = np.zeros(4)
    
    #Nested for loop for multiplying b by inverse of matrix
    for x in range(4):
        add = 0
        for y in range(4):
            add += inverse[x][y] * b[y]
        #This allows us to determine the x vector
        xVector[x] = add

    print(xVector)

    return None






if __name__ == "__main__":

    np.set_printoptions(precision=7, suppress=True, linewidth=100)
    # point setup
    x_points = [3.6, 3.8, 3.9]
    y_points = [1.675, 1.436, 1.318]
    approximating_value = 3.7
    nevilles_method(x_points, y_points, approximating_value)
    
    x_pts = [7.2, 7.4, 7.5, 7.6]
    y_pts = [23.5492, 25.3913, 26.8224, 27.4589]
    divided_table = divided_difference_table(x_pts, y_pts)
    # find approximation

    #We use this format to match the output file exactly for divided table
    print(end = "\n[")
    for index in range (1, len(x_pts)):
        if (index != (len(x_pts) - 1)):
            print(divided_table[index][index], end = ", ")
        else:
            print(divided_table[index][index], end = "]")

    print("\n")

    approximating_x = 7.3
    final_approximation = get_approximate_result(divided_table, x_pts, approximating_x)
    
    print(final_approximation, "\n")

    hermite_interpolation()

    matrixCalculator()




