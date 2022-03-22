// implementation of matrix data type

#include "matrix.hpp"

template <class T>
matrix<T>::matrix() : init(false) {}

template <class T>
matrix<T>::matrix(int row_, int col_) : init(true), data(vector<vector<T>>(row_, vector<T>(col_))), row(row_), col(col_) {
  if (row == col) {
    for (int i = 0; i < row; i++) {
      (*this)(i, i) = 1;
    }
  }
}

template <class T>
matrix<T>::matrix(int row_, int col_, T val) : init(true), data(vector<vector<T>>(row_, vector<T>(col_, val))), row(row_), col(col_) {}

template <class T>
matrix<T>::matrix(const vector<T> &v) : init(true) {
  row = static_cast<int>(v.size());
  col = 1;
  data = vector<vector<T>>(row, vector<T>(col));
  for (int i = 0; i < row; i++) {
    data[i][0] = v[i];
  }
}

template <class T>
matrix<T>::matrix(const vector<vector<T>>& v) : init(true), data(v), row((int) v.size()), col((int) v[0].size()) {}

template <class T>
istream& operator>>(istream& is, matrix<T>& m) {
  if (!m.init) {
    m.init = true;
    is >> m.row >> m.col;
    m.data = vector<vector<T>>(m.row, vector<T>(m.col));
  }
  for (int i = 0; i < m.row; i++) {
    for (int j = 0; j < m.col; j++) {
      is >> m(i, j);
    }
  }
  return is;
}

template <class T>
ostream& operator<<(ostream& os, const matrix<T>& m) {
  for (int i = 0; i < m.row; i++) {
    cout << "{ ";
    for (int j = 0; j < m.col; j++) {
      cout << m(i, j) << ", ";
    }
    cout << "}\n";
  }
  return os;
}

template <class T>
matrix<T> operator+(const matrix<T>& a, const matrix<T>& b) {
  assert(a.row == b.row && a.col == b.col);
  matrix<T> res(a.row, a.col, static_cast<T>(0));
  for (int i = 0; i < a.row; i++) {
    for (int j = 0; j < a.col; j++) {
      res(i, j) = a(i, j) + b(i, j);
    }
  }
  return res;
}

template <class T>
matrix<T>& matrix<T>::operator+=(const matrix<T>& m) {
  return *this = (*this) + m;
}

template <class T>
matrix<T> operator-(const matrix<T>& a, const matrix<T>& b) {
  assert(a.row == b.row && a.col == b.col);
  matrix<T> res(a.row, a.col, static_cast<T>(0));
  for (int i = 0; i < a.row; i++) {
    for (int j = 0; j < a.col; j++) {
      res(i, j) = a(i, j) - b(i, j);
    }
  }
}

template <class T>
matrix<T>& matrix<T>::operator-=(const matrix<T>& m) {
  return *this = (*this) - m;
}

template <class T>
matrix<T> matrix<T>::operator-() {
  matrix<T> res = (*this);
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      res(i, j) *= -1;
    }
  }
  return res;
}

template <class T>
matrix<T> operator*(const matrix<T>& a, const matrix<T>& b) {
  assert(a.col == b.row);
  matrix<T> res(a.row, b.col, static_cast<T>(0));
  for (int i = 0; i < a.row; i++) {
    for (int j = 0; j < b.col; j++) {
      for (int k = 0; k < a.col; k++) {
        res(i, j) += a(i, k) * b(k, j);
      }
    }
  }
  return res;
}

template <class T>
matrix<T>& matrix<T>::operator*=(const matrix<T>& m) {
  return *this = (*this) * m;
}

template <class T>
matrix<T> operator*(const matrix<T>& m, T x) {
  matrix<T> res = *this;
  for (int i = 0; i < m.row; i++) {
    for (int j = 0; j < m.col; j++) {
      res(i, j) *= x;
    }
  }
  return res;
}

template <class T>
matrix<T> operator*(T x, const matrix<T>& m) {
  return m * x;
}

template <class T>
matrix<T>& matrix<T>::operator*=(T x) {
  return *this = (*this) * x;
}

template <class T>
matrix<T> operator/(const matrix<T>& m, T x) {
  assert(T);
  matrix<T> res = *this;
  for (int i = 0; i < m.row; i++) {
    for (int j = 0; j < m.col; j++) {
      res(i, j) /= x;
    }
  }
  return res;
}

template <class T>
matrix<T> matrix<T>::operator/=(T x) {
  return *this = (*this) / x;
}
