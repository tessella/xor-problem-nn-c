#ifndef NN_H_ // Header section begins
#define  NN_H_

#include <stddef.h>
#include <stdio.h>
#include <math.h>

#ifndef NN_MALLOC
#include <stdlib.h>
#define  NN_MALLOC malloc
#endif // NN_MALLOC

#ifndef NN_ASSERT
#include <assert.h>
#define  NN_ASSERT assert
#endif // NN_ASSERT

//very lightweight as its 3 64 bit ints
typedef struct {
    size_t rows;
    size_t cols;
    size_t stride;
    float *es;
} Mat;

#define MAT_AT(m, i, j) (m).es[(i)*(m).stride + (j)]

float rand_float(void);
float xavier_init(size_t inputs, size_t outputs);
float sigmoidf(float x);

Mat mat_alloc(size_t rows, size_t cols);
void mat_free(Mat *m);
void mat_fill(Mat m, float x);
void mat_rand(Mat m, float low, float high);
void mat_xavier_init(Mat m, size_t inputs, size_t outputs);
Mat mat_row(Mat m, size_t row);
void mat_sub(Mat m, size_t);
void mat_copy(Mat dst, Mat src);
void mat_dot(Mat dst, Mat a, Mat b); //preallocate memory for the three matrices
void mat_sum(Mat dst, Mat a);
void mat_sig(Mat m);
void mat_print(Mat m, const char *name);
#define MAT_PRINT(m) mat_print(m, #m)




#endif // NN_H_ Header section ends

#ifdef  NN_IMPLEMENTATION // C Implementation begins

inline float sigmoidf(float x)
{
    return 1.f / (1.f + expf(-x));
}

inline float rand_float(void)
{
    return (float) rand() / (float) RAND_MAX;
}

inline float xavier_init(size_t inputs, size_t outputs)
{
    float limit = sqrt(6.0 / (inputs + outputs));
    return rand_float() * 2 * limit - limit;
}

inline Mat mat_alloc(size_t rows, size_t cols) // allocate memory for the matrix struct
{
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.stride = cols;
    m.es = NN_MALLOC(sizeof(*m.es) * rows * cols);
    NN_ASSERT(m.es != NULL);
    return m;
}

inline void mat_free(Mat *m)
{
    if (m->es) {
        free(m->es);
        m->es = NULL;
    }
    m->rows = m->cols = m->stride = 0;
}

inline void mat_fill(Mat m, float x)
{
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            MAT_AT(m, i, j) = x;
        }
    }
}

inline void mat_sum(Mat dst, Mat a)
{
    NN_ASSERT(dst.rows == a.rows);
    NN_ASSERT(dst.cols == a.cols);
    for (size_t i = 0; i < dst.rows; ++i){
        for (size_t j = 0; j < dst.cols; ++j) {
            MAT_AT(dst, i, j) += MAT_AT(a, i, j);
        }
    }
}

inline void mat_sig(Mat m)
{
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            MAT_AT(m, i, j) = sigmoidf(MAT_AT(m, i, j));
        }
    }
}

inline void mat_dot(Mat dst, Mat a, Mat b)
{
    NN_ASSERT(a.cols == b.rows);
    size_t n = a.cols;
    NN_ASSERT(dst.rows == a.rows);
    NN_ASSERT(dst.cols == b.cols);

    for (size_t i = 0; i < dst.rows; ++i) {
        for (size_t j = 0; j < dst.cols; ++j) {
            for (size_t k = 0; k < n; ++k) {
                MAT_AT(dst, i, j) += MAT_AT(a, i, k) * MAT_AT(b, k, j);
            }
        }
    }
}

inline void mat_print(Mat m, const char *name)
{
    printf("%s = [\n", name);
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            printf("    %f ", MAT_AT(m, i, j));
        }
        printf("\n");
    }
    printf("]\n");
}

inline void mat_rand(Mat m, float low, float high)
{
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            MAT_AT(m, i, j) = rand_float() * (high - low) + low;
        }
    }
}

inline void mat_xavier_init(Mat m, size_t inputs, size_t outputs)
{
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            MAT_AT(m, i, j) = xavier_init(inputs, outputs);
        }
    }
}

inline Mat mat_row(Mat m, size_t row)
{
    return (Mat){
        .rows = 1,
        .cols = m.cols,
        .stride = m.stride,
        .es = &MAT_AT(m, row, 0)
    };
}


inline void mat_copy(Mat dst, Mat src)
{
    NN_ASSERT(dst.rows == src.rows);
    NN_ASSERT(dst.cols == src.cols);
    for (size_t i = 0; i < dst.rows; ++i) {
        for (size_t j = 0; j < dst.cols; ++j) {
            MAT_AT(dst, i, j) = MAT_AT(src, i, j);
        }
    }
}

#endif // NN_IMPLEMENTATION C implementation ends
