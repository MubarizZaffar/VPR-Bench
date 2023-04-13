import cv2
import numpy as np
cimport numpy as np
import math
import matplotlib.pyplot as plt
cimport cython

DTYPE_FLOAT = np.float64
ctypedef np.float_t DTYPE_FLOAT_t
DTYPE_INT = int
DTYPE_UINT8 = np.uint8
ctypedef np.int_t DTYPE_INT_t

import time

cdef int height=0
cdef int width=0
cdef int cell_size=16
cdef int bin_size=8
cdef int angle_unit=360/bin_size
cdef np.ndarray img = np.zeros((height, width),dtype=DTYPE_UINT8)


#@cython.boundscheck(False)  # Deactivate bounds checking
#@cython.wraparound(False)   # Deactivate negative indexing.
cpdef initialize(image,int cellsize=16,int binsize=8):
    global height
    global width
    global cell_size
    global bin_size
    global angle_unit
    global img
    img = image
    img = np.sqrt(img / float(np.max(img)))
    img = img * 255
    height, width = img.shape[0], img.shape[1]
    cell_size = cellsize
    bin_size = binsize
    angle_unit = 360 / binsize
    
    return height,width,angle_unit

#@cython.boundscheck(False)  # Deactivate bounds checking
#@cython.wraparound(False)   # Deactivate negative indexing.
def extract():
    
    global height
    global width
    global cell_size
    global bin_size
    global angle_unit
    cdef int itr=0
    global img

    cdef np.ndarray gradient_magnitude = np.zeros((height , width),dtype=DTYPE_FLOAT)
    cdef np.ndarray gradient_angle = np.zeros((height, width),dtype=DTYPE_FLOAT)    
    cdef np.ndarray cell_magnitude = np.zeros((cell_size, cell_size),dtype=DTYPE_FLOAT)
    cdef double [:, :] cell_magnitude_view = cell_magnitude
    cdef np.ndarray cell_angle = np.zeros((cell_size,cell_size),dtype=DTYPE_FLOAT)
    cdef double [:, :] cell_angle_view = cell_angle
    cdef np.ndarray cell_gradient_vector = np.zeros((height / cell_size, width / cell_size, bin_size),dtype=DTYPE_FLOAT)
#    cdef double [:, :, :] cell_gradient_vector_view = cell_gradient_vector
    cdef list block_vector
    cdef float magnitude 
    
    cdef np.ndarray hog_vector = np.zeros(((cell_gradient_vector.shape[0] - 1)*(cell_gradient_vector.shape[1] - 1), bin_size*4),dtype=DTYPE_FLOAT)
    

    gradient_magnitude, gradient_angle = global_gradient()
    gradient_magnitude = abs(gradient_magnitude)
    cdef double [:, :] gradient_magnitude_view = gradient_magnitude
    cdef double [:, :] gradient_angle_view = gradient_angle        
    
    cdef Py_ssize_t i=0
    cdef Py_ssize_t j=0
    cdef Py_ssize_t imax=cell_gradient_vector.shape[0]
    cdef Py_ssize_t jmax=cell_gradient_vector.shape[1]
    
#    cell_time=time.time()
    for i in range(imax):
        for j in range(jmax):
#                          mag_time=time.time()
                      cell_magnitude_view = gradient_magnitude_view[i * cell_size:(i + 1) * cell_size,
                             j * cell_size:(j + 1) * cell_size]
#                          print('cell mag time:', time.time()-mag_time)
#                          angle_time=time.time()
                      cell_angle_view = gradient_angle_view[i * cell_size:(i + 1) * cell_size,
                         j * cell_size:(j + 1) * cell_size]
#                          print('cell angle mm time',time.time()-angle_time)
#                          grad_vectortime=time.time()
#                          print('cell_grad',cell_gradient_vector[i,j,:])
                      cell_gradient_vector[i,j] = cell_gradient(cell_magnitude_view, cell_angle_view)
#                          print('cell gradvector time',time.time()-grad_vectortime)
    #hog_image = self.render_gradient(np.zeros([height, width]), cell_gradient_vector)
#    print('cell time:',time.time()-cell_time)
    
#    block_time=time.time()
    for i in range(imax - 1):
        for j in range(jmax - 1):
            block_vector = []
            block_vector.extend(cell_gradient_vector[i][j])
            block_vector.extend(cell_gradient_vector[i][j + 1])
            block_vector.extend(cell_gradient_vector[i + 1][j])
            block_vector.extend(cell_gradient_vector[i + 1][j + 1])
            mag = lambda vector: math.sqrt(sum(i ** 2 for i in vector))
            magnitude = mag(block_vector)
            if magnitude != 0:
                normalize = lambda block_vector, magnitude: [element / magnitude for element in block_vector]
                block_vector = normalize(block_vector, magnitude)
            hog_vector[itr]=block_vector
            itr=itr+1
#    print('block time:',time.time()-block_time)
    
    return hog_vector

#@cython.boundscheck(False)  # Deactivate bounds checking
#@cython.wraparound(False)   # Deactivate negative indexing.
def global_gradient():
    global height
    global width
    global img
    
    cdef np.ndarray gradient_values_x = np.zeros((height , width),dtype=DTYPE_FLOAT)
    cdef np.ndarray gradient_values_y = np.zeros((height , width),dtype=DTYPE_FLOAT)
    cdef np.ndarray gradient_magnitude = np.zeros((height , width),dtype=DTYPE_FLOAT)
    cdef np.ndarray gradient_angle = np.zeros((height , width),dtype=DTYPE_FLOAT)
    
    gradient_values_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    gradient_values_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    gradient_magnitude = cv2.addWeighted(gradient_values_x, 0.5, gradient_values_y, 0.5, 0)
    gradient_angle = cv2.phase(gradient_values_x, gradient_values_y, angleInDegrees=True)
    return gradient_magnitude, gradient_angle


#@cython.boundscheck(False)  # Deactivate bounds checking
#@cython.wraparound(False)   # Deactivate negative indexing.
cdef cell_gradient(double [:,:] cell_magnitude, double [:,:] cell_angle):
#        cell_grad_func_time=time.time()
#        print ('here1')

    global bin_size
    global angle_unit
    global img
    cdef np.ndarray orientation_centers = np.zeros(bin_size,dtype=DTYPE_FLOAT)
    cdef double[:] orientation_centers_view = orientation_centers
#        print ('here2')
    #cdef list orientation_centers = [0] * self.bin_size
    cdef float gradient_strength=0
    cdef float gradient_angle=0
    cdef int min_angle=0
    cdef int max_angle=0
    cdef float mod=0
    cdef Py_ssize_t i=0
    cdef Py_ssize_t j=0
    cdef Py_ssize_t imax = cell_magnitude.shape[0]
    cdef Py_ssize_t jmax = cell_magnitude.shape[1]
       
    for i in range(imax):
        for j in range(jmax):
            gradient_strength = cell_magnitude[i][j]
            gradient_angle = cell_angle[i][j]            
            
            min_angle, max_angle, mod = get_closest_bins(gradient_angle)
            orientation_centers_view[min_angle] += (gradient_strength * (1 - (mod / angle_unit)))
            orientation_centers_view[max_angle] += (gradient_strength * (mod / angle_unit))
#        print ('here3')
#        print('orien_centers:',orientation_centers)
#        print('cell_grad_func_time:',time.time()-cell_grad_func_time)
    
    return orientation_centers

#@cython.boundscheck(False)  # Deactivate bounds checking
#@cython.wraparound(False)   # Deactivate negative indexing.
cdef get_closest_bins(float gradient_angle):
    global bin_size
    global angle_unit

    cdef int idx = int(gradient_angle / angle_unit)
    cdef float mod = gradient_angle % angle_unit
    if idx == bin_size:
        return idx - 1, (idx) % bin_size, mod
    return idx, (idx + 1) % bin_size, mod

#    def render_gradient(self, image, cell_gradient):
#        cell_width = self.cell_size / 2
#        max_mag = np.array(cell_gradient).max()
#        for x in range(cell_gradient.shape[0]):
#            for y in range(cell_gradient.shape[1]):
#                cell_grad = cell_gradient[x][y]
#                cell_grad /= max_mag
#                angle = 0
#                angle_gap = self.angle_unit
#                for magnitude in cell_grad:
#                    angle_radian = math.radians(angle)
#                    x1 = int(x * self.cell_size + magnitude * cell_width * math.cos(angle_radian))
#                    y1 = int(y * self.cell_size + magnitude * cell_width * math.sin(angle_radian))
#                    x2 = int(x * self.cell_size - magnitude * cell_width * math.cos(angle_radian))
#                    y2 = int(y * self.cell_size - magnitude * cell_width * math.sin(angle_radian))
#                    cv2.line(image, (y1, x1), (y2, x2), int(255 * math.sqrt(magnitude)))
#                    angle += angle_gap
#        return image



