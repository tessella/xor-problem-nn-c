#define NN_IMPLEMENTATION
#include "neural_net.h"
#include <mach/mach_time.h>


typedef struct { // Here we can easily add layers into our model
  Mat a0, a1, a2;
  Mat w1, b1; // Layer 1
  Mat w2, b2; // Layer 2
  int owns_matrices; // Indicates if the struct owns the matrices
} Xor;

float td[] = {
  0, 0, 0,
  0, 1, 1,
  1, 0, 1,
  1, 1, 0
};

Xor xor_alloc(void)
{
  Xor m;

  // Initial conditions
  m.a0 = mat_alloc(1, 2);

  // Layer 1:
  m.w1 = mat_alloc(2, 2);
  m.b1 = mat_alloc(1, 2);
  m.a1 = mat_alloc(1, 2); // Destination matrix

  // Layer 2:
  m.w2 = mat_alloc(2, 1);
  m.b2 = mat_alloc(1, 1);
  m.a2 = mat_alloc(1, 1); // Again, destination matrix

  return m;
}

float forward_xor(Xor m)
{
  // Passing through layer 1:
  mat_dot(m.a1, m.a0, m.w1);
  mat_sum(m.a1, m.b1);
  mat_sig(m.a1);

  // Passing through layer 2:
  mat_dot(m.a2, m.a1, m.w2);
  mat_sum(m.a2, m.b2);
  mat_sig(m.a2);
}

float cost(Xor m, Mat ti, Mat to) { // training input, training output
  assert(ti.rows == to.rows);   // Checks if training data was split correctly
  assert(to.cols == m.a2.cols); // Output from network matches expected output in training data
  assert(ti.cols == m.a0.cols); // Number of input features matches input layer of network
  float n = ti.rows;
  float c = 0;

  for (size_t i = 0; i < n; ++i) {
    Mat x = mat_row(ti, i);
    Mat y = mat_row(to, i);
    mat_copy(m.a0, x);
    forward_xor(m);

    size_t q = to.cols;
    for (size_t j = 0; j < q; ++j) {
      float d = MAT_AT(m.a2, 0, j) - MAT_AT(y, 0, j);
      c += d*d;
    }
  }
  return c/n;
}

void finite_diff(Xor m, Xor g, Mat ti, Mat to, float eps)
{
  float saved;
  float c = cost(m, ti, to);

  for (size_t i = 0; i < m.w1.cols; ++i) {
    for (size_t j = 0; j < m.w1.rows; ++j) {
      saved = MAT_AT(m.w1, i, j);
      MAT_AT(m.w1, i, j) += eps;
      MAT_AT(g.w1, i, j) = (cost(m, ti, to) - c) / eps;
      MAT_AT(m.w1, i, j) = saved;
    }
  }

  for (size_t i = 0; i < m.b1.cols; ++i) {
    for (size_t j = 0; j < m.b1.rows; ++j) {
      saved = MAT_AT(m.b1, i, j);
      MAT_AT(m.b1, i, j) += eps;
      MAT_AT(g.b1, i, j) = (cost(m, ti, to) - c) / eps;
      MAT_AT(m.b1, i, j) = saved;
    }
  }

  for (size_t i = 0; i < m.w2.cols; ++i) {
    for (size_t j = 0; j < m.w2.rows; ++j) {
      saved = MAT_AT(m.w2, i, j);
      MAT_AT(m.w2, i, j) += eps;
      MAT_AT(g.w2, i, j) = (cost(m, ti, to) - c) / eps;
      MAT_AT(m.w2, i, j) = saved;
    }
  }

  for (size_t i = 0; i < m.b2.cols; ++i) {
    for (size_t j = 0; j < m.b2.rows; ++j) {
      saved = MAT_AT(m.b2, i, j);
      MAT_AT(m.b2, i, j) += eps;
      MAT_AT(g.b2, i, j) = (cost(m, ti, to) - c) / eps;
      MAT_AT(m.b2, i, j) = saved;
    }
  }

}

float xor_learn(Xor m, Xor g, float rate)
{
  for (size_t i = 0; i < m.w1.cols; ++i) {
    for (size_t j = 0; j < m.w1.rows; ++j) {
      MAT_AT(m.w1, i, j) -= rate * MAT_AT(g.w1, i, j);
    }
  }

  for (size_t i = 0; i < m.b1.cols; ++i) {
    for (size_t j = 0; j < m.b1.rows; ++j) {
      MAT_AT(m.b1, i, j) -= rate * MAT_AT(g.b1, i, j);
    }
  }

  for (size_t i = 0; i < m.w2.cols; ++i) {
    for (size_t j = 0; j < m.w2.rows; ++j) {
      MAT_AT(m.w2, i, j) -= rate * MAT_AT(g.w2, i, j);
    }
  }

  for (size_t i = 0; i < m.b2.cols; ++i) {
    for (size_t j = 0; j < m.b2.rows; ++j) {
      MAT_AT(m.b2, i, j) -= rate * MAT_AT(g.b2, i, j);
    }
  }

}

int main(void) {
  uint64_t seed = mach_absolute_time();
  srand((unsigned int) seed);

  size_t stride = 3;
  size_t n = sizeof(td) / sizeof(td[0])/stride;

  // Below we split training data into input and output:

  Mat ti = { // Input data
    .rows = n,
    .cols = 2,
    .stride = stride,
    .es = td
  };

  Mat to = { // Output data
    .rows = n,
    .cols = 1,
    .stride = stride,
    .es = td + 2
  };

  Xor m = xor_alloc();
  Xor g = xor_alloc();

  // Initialise matrices: (Xavier initialisation reduced variance when training)
  mat_xavier_init(m.w1, m.a0.cols, m.a1.cols); //mat_rand(m.w1, -0.5, 0.5);
  mat_rand(m.b1, -0.5, 0.5);
  mat_xavier_init(m.w2, m.a1.cols, m.a2.cols); //mat_rand(m.w2, -0.5, 0.5);
  mat_rand(m.b2, -0.5, 0.5);

  float eps = 1e-1;
  float rate = 1e-1;

  printf("cost: %f\n", cost(m, ti, to));   // Compute old cost
  for (size_t i = 0; i < 100*1000; ++i) {
    finite_diff(m, g, ti, to, eps);        // Compute finite difference ("grad")
    xor_learn(m, g, rate);                 // Apply gradient
    cost(m, ti, to); // Compute new cost
  }
  printf("cost: %f\n", cost(m, ti, to));

  printf("---------------------------\n");

  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 2; ++j) {
      MAT_AT(m.a0, 0, 0) = i;
      MAT_AT(m.a0, 0, 1) = j;
      forward_xor(m);
      float y = *m.a2.es;
      printf("%zu ^ %zu = %f\n", i, j, y);
    }
  }

  return 0;
}
