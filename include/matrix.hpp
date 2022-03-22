#ifndef MATRIX_H
#define MATRIX_H

// interface for matrix data type

#include <iostream>
#include <vector>

using namespace std;

template <class T>
class matrix {
private:
  vector<vector<T>> data;
  int row, col;
  bool init;

public:
  matrix();                         // default constructor
  matrix(int, int);                 // constructing matrix with given rows and cols (zero-initialization)
  matrix(int, int, T);              // constructing matrix with given rows and cols and default value for elements
  matrix(const vector<T>&);         // constructing a matrix for 1-D vector
  matrix(const vector<vector<T>>&); // constructing a matrix for 2-D vector

  friend istream& operator>>(istream&, matrix&);          // cin
  friend ostream& operator<<(ostream&, const matrix&);    // cout
  
  friend matrix operator+(const matrix&, const matrix&);  // addition b/w two matrices
  matrix& operator+(const matrix&);
  friend matrix operator-(const matrix&, const matrix&);  // substraction b/w two matrices
  matrix& operator-(const matrix&);
  friend matrix operator*(const matrix&, const matrix&);  // multiplication b/w two matrices
  matrix& operator*(const matrix&);
  friend matrix operator*(const matrix&, T);              // |
  friend matrix operator*(T, const matrix&);              // | scalar multiplcation
  matrix& operator*(T);
  friend matrix operator/(const matrix&, T);              // scalar division
  matrix& operator/(T);

  // only applicable for square-matrices
  friend matrix operator^(const matrix&, int);   // matrix exponentiation
  matrix& operator^(int);
  T det();                                       // matrix determinant
  matrix adjoint();                              // adjoint matrix
  matrix inv();                                  // inverse
};

#endif // MATRIX_H
