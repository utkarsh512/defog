// interface for matrix data type

#ifndef MATRIX_H
#define MATRIX_H

#ifndef BITS_H
#define BITS_H

#include <bits/stdc++.h>
using namespace std;

#endif // BITS_H

#ifndef RANDOM_NUM
#define RANDOM_NUM

mt19937_64 rng(chrono::high_resolution_clock::now().time_since_epoch().count());

template <class T>
inline T rnd(T l = numeric_limits<T>::min(), T r = numeric_limits<T>::max()) {
  if constexpr(is_integral_v<T>) {
    uniform_int_distribution<T> dis(l, r);
    return dis(rng);
  } else if (is_floating_point_v<T>) {
    uniform_real_distribution<T> dis(l, r);
    return dis(rng);
  }
}

#endif // RANDOM_NUM

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

  T& operator()(int, int);          // |
  T operator()(int, int) const;     // | allows matrix access as m(i, j)

  matrix T() const;                 // transpose
  void setRandom();
  void setIdentity();

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
  T det() const;                                       // matrix determinant
  matrix adjoint() const;                              // adjoint matrix
  matrix inv() const;                                  // inverse
};

#endif // MATRIX_H
