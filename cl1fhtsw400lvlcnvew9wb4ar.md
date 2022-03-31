## Numpy 101

## Introduction

Numpy is a python library that provides functions that are especially useful when you have to work with large arrays and matrices of numeric data, like doing matrix matrix multiplications. Also, Numpy is battle tested and optimized so that it runs fast, much faster than if you were working with Python lists directly.

The array object class is the foundation of Numpy, and Numpy arrays are like lists in Python, except that every thing inside an array must be of the **same type**, like int or float.

Let us quickly have a look at some of the code snippets below to understand how numpy handles arrays, their data types and other functions. 
### Numpy Arrays
```
    array = np.array([1, 4, 5, 8], float) # a 1D array of float type
    print array
OUTPUT: 
    [1. 4. 5. 8.]
```
```
    array = np.array([[1, 2, 3], [4, 5, 6]], float)  # a 2D array/Matrix
    print array
OUTPUT: 
    [[1. 2. 3.]
    [4. 5. 6.]]
```
### Array Slicing
Numpy allows array slicing just like you can do with lists in Python. Array slicing is applicable to two dimensional arrays as well, not shown here though. It follows the same concept as one dimensional, just the second dimension is placed alongside.
```
    array = np.array([1,2,3,4], float)
    print array
    print array[1]
    print array[:2]
    array[1] = 5.0
    print array
OUTPUT:
    [1. 2. 3. 4.]
    2.0
    [1. 2.]
    [1. 5. 3. 4.]
```
### Standard Arithmetic Operations
Here are some arithmetic operations that can be performed on numpy arrays. More functions and operations are also there at an advanced level.
```
    array_1 = np.array([1, 2, 3], int)
    array_2 = np.array([4, 5, 6], int)
    print array_1 + array_2
    print array_1 - array_2
    print array_1 * array_2
OUTPUT:
    [5 7 9]
    [-3 -3 -3]
    [ 4 10 18]
```
The same operations are applicable to two dimensional arrays as well.
```
    array_1 = np.array([[1, 2], [3, 4]], float)
    array_2 = np.array([[5, 6], [7, 8]], float)
    print array_1
    print array_2
    print array_1 + array_2
    print array_1 - array_2
    print array_1 * array_2
OUTPUT:
    [[1. 2.]
     [3. 4.]]
    [[5. 6.]
     [7. 8.]]
    [[ 6.  8.]
     [10. 12.]]
    [[-4. -4.]
     [-4. -4.]]
    [[ 5. 12.]
     [21. 32.]]
```
### Mathematical Operations
In addition to the standard arthimetic operations, Numpy also has a range of other mathematical operations that you can apply to Numpy arrays, such as mean and dot product.
```
    array_1 = np.array([1, 2, 3], float)
    array_2 = np.array([[6], [7], [8]], float)
    print np.mean(array_1) #mean
    print np.mean(array_2)
    print np.std(array_1) #standard deviation
    print np.dot(array_1, array_2) #scalar dot product
OUTPUT:
    2.0
    7.0
    0.816496580928
    [44.]
```
--------
I hope you learned something from this blog if you followed it carefully. As a reward for my time and hard work feel free to [buy me a beer or coffee](https://www.buymeacoffee.com/amitrajit).