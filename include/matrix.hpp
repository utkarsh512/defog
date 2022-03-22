// interface for matrix data type

#ifndef MATRIX_H
#define MATRIX_H

#include <bits/stdc++.h>
using namespace std; 

template <class T>
class matrix {
private:
  bool init;
  vector<vector<T>> data;

public:
  int row, col;

  matrix();                         // default constructor
  matrix(int, int);                 // constructing matrix with given rows and cols (identity for square matrices)
  matrix(int, int, T);              // constructing matrix with given rows and cols and default value for elements
  matrix(const vector<T>&);         // constructing a matrix for 1-D vector
  matrix(const vector<vector<T>>&); // constructing a matrix for 2-D vector

  inline T& operator()(int x, int y)  { return data[x][y]; }
  inline T operator()(int x, int y) const { return data[x][y]; }

  template <class U> friend istream& operator>>(istream&, matrix<U>&);          // cin
  template <class U> friend ostream& operator<<(ostream&, const matrix<U>&);    // cout

  template <class U> friend matrix operator+(const matrix<U>&, const matrix<U>&); 
  matrix& operator+=(const matrix&);                      // addition b/w two matrices

  template <class U> friend matrix operator-(const matrix<U>&, const matrix<U>&); 
  matrix& operator-=(const matrix&);                      // substraction b/w two matrices

  matrix operator-();                                     // negation

  template <class U> friend matrix operator*(const matrix<U>&, const matrix<U>&); 
  matrix& operator*=(const matrix&);                      // multiplication b/w two matrices

  template <class U> friend matrix operator*(const matrix<U>&, U);          
  template <class U> friend matrix operator*(U, const matrix<U>&);            
  matrix& operator*=(T);                                  // scalar multiplcation

  template <class U> matrix operator/(const matrix<U>&, T);        
  matrix& operator/=(T);                                  // scalar division

  // following methods valid for square-matrices only

  template <class U> friend matrix operator^(const matrix<U>&, int);           
  matrix& operator^=(int);                                // matrix exponentiation

  T det() const;                                          // matrix determinant
  matrix adjoint() const;                                 // adjoint matrix
  matrix inv() const;                                     // inverse
};

#include "matrix.ipp"

#endif // MATRIX_H
