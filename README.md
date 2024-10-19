# XOR Neural Network in C

This project builds a simple two-layer neural network in C that solves the XOR problem. It includes both memory-safe and unsafe versions of the implementation, demonstrating different approaches to C programming and memory management.

Implementing a XOR neural network from scratch was tedious but served as a great way of understanding the underlying structures and mathematical tools used in Machine Learning. It explicitely motivates the use of "tensors", multilinear maps which in this case encapsulate the layers of the overall neural network in a single, easily manipulable structure. Despite having a mathematical background, I didn't fully understand where tensors were used until I built the XOR structure displayed in this project.

## Key Components

1. **Matrix Operations Framework**:
   - A header file (`neural_net.h`) containing a lightweight framework for matrix operations.

2. **Dual Versions**:
   - Unsafe version (`xor_unsafe.c`): A straightforward but less robust approach, leading to 256 bytes of leaked memory:
  ```powershell
  ❯ leaks --atExit -- ./Machine_Learning_C
  cost: 0.375066
  cost: 0.000509
  ---------------------------
  0 ^ 0 = 0.024777
  0 ^ 1 = 0.973803
  1 ^ 0 = 0.980775
  1 ^ 1 = 0.019101
  
  Process:         Machine_Learning_C [4217]
  
  Physical footprint:         3729K
  Physical footprint (peak):  3729K
  
  leaks Report Version: 4.0
  Process 4217: 203 nodes malloced for 15 KB
  Process 4217: 14 leaks for 256 total leaked bytes.
  ```  
  
   - Memory-safe version (`xor_pointers.c`): Uses pointers for proper memory management, removing memory leaks:
  ```powershell
  ❯ leaks --atExit -- ./Machine_Learning_C
  cost: 0.252240
  cost: 0.001008
  ---------------------------
  0 ^ 0 = 0.039103
  0 ^ 1 = 0.960554
  1 ^ 0 = 0.976133
  1 ^ 1 = 0.019461
  
  Process:         Machine_Learning_C [4160]
  
  Physical footprint:         3681K
  Physical footprint (peak):  3697K
  
  leaks Report Version: 4.0
  Process 4160: 189 nodes malloced for 14 KB
  Process 4160: 0 leaks for 0 total leaked bytes.
  ```

3. **Neural Network Structure**:
   - The network is represented by a custom `Xor` structure, which can be thought of as a "tensor" storing multiple matrices.
   - It includes matrices for input (`a0`), weights (`w1`, `w2`), biases (`b1`, `b2`), and activation outputs (`a1`, `a2`) for each layer.

   ```c
   void xor_alloc(Xor *m)
   {
      // Inputs
      m->a0 = mat_alloc(1, 2);
    
      // Layer 1:
      m->w1 = mat_alloc(2, 2);
      m->b1 = mat_alloc(1, 2);
      m->a1 = mat_alloc(1, 2); // Destination matrix
    
      // Layer 2:
      m->w2 = mat_alloc(2, 1);
      m->b2 = mat_alloc(1, 1);
      m->a2 = mat_alloc(1, 1); // Again, destination matrix
    
      m->owns_matrices = 1; // Set the flag
   }
   ```

4. **Neural Network Implementation**:

   1. **Network Initialisation**:
      - The `xor_alloc` function initialises the `Xor` structure, allocating memory for all matrices.
      - This sets up the network architecture: 2 input nodes, 2 hidden nodes, and 1 output node.

   2. **Forward Propagation**: 
      - Explicitly defined for both layers of the network.
      - Process: matrix multiplication (input * weights), bias addition, and sigmoid activation.
      - Results are stored in the `a1` and `a2` matrices of the `Xor` structure.

      ```c
      void forward_xor(const Xor *m)
      {
        // Passing through layer 1:
        mat_dot(m->a1, m->a0, m->w1);
        mat_sum(m->a1, m->b1);
        mat_sig(m->a1);
      
        // Passing through layer 2:
        mat_dot(m->a2, m->a1, m->w2);
        mat_sum(m->a2, m->b2);
        mat_sig(m->a2);
      }
      ```  

   3. **Cost Function Calculation**: 
      - Forwards training inputs through the initialised XOR layers.
      - Computes the mean squared error between the final layer's output (`a2`) and the expected result (`to`).

      ```c
      float cost(const Xor *m, Mat ti, Mat to) {
        NN_ASSERT(ti.rows == to.rows);   
        NN_ASSERT(to.cols == m->a2.cols); 
        NN_ASSERT(ti.cols == m->a0.cols); 
        size_t n = ti.rows;
        float c = 0;
      
        // Forward training input through NN layers
        for (size_t i = 0; i < n; ++i) {
          Mat x = mat_row(ti, i);
          Mat y = mat_row(to, i);
          mat_copy(m->a0, x);
          forward_xor(m);
      
          // Compute squared difference between NN output and intended output
          size_t q = to.cols;
          for (size_t j = 0; j < q; ++j) {
            float d = MAT_AT(m->a2, 0, j) - MAT_AT(y, 0, j);
            c += d*d;
          }
        }
        return c/(float)n;
      }
      ```

   4. **Gradient Approximation**: 
      - Uses a finite difference method based on the derivative definition.
      - Computes gradients individually for weight and bias matrices at both layers.
      - Gradient information is stored in a separate `Xor` structure (`g`), mirroring the main network structure.

      ```c
      void finite_diff(const Xor *m, const Xor *g, Mat ti, Mat to, float eps)
      {
        float saved;
        float c = cost(m, ti, to);
        
        for (size_t i = 0; i < m->w1.cols; ++i) {
          for (size_t j = 0; j < m->w1.rows; ++j) {
            saved = MAT_AT(m->w1, i, j);
            MAT_AT(m->w1, i, j) += eps;
            MAT_AT(g->w1, i, j) = (cost(m, ti, to) - c) / eps;
            MAT_AT(m->w1, i, j) = saved;
          }
        }
       // Repeat for every matrix in Xor m.
      }
      ```

   5. **Learning Algorithm**: 
      - Applies the computed gradient (stored in `g`) to update weight and bias matrices in the main `Xor` structure (`m`).
      - Updates parameters using the formula: parameter -= learning_rate * gradient

      ```c
      void xor_learn(const Xor *m, const Xor *g, float rate)
      {
        for (size_t i = 0; i < m->w1.cols; ++i) {
          for (size_t j = 0; j < m->w1.rows; ++j) {
            MAT_AT(m->w1, i, j) -= rate * MAT_AT(g->w1, i, j);
          }
        }
       // Repeat for every matrix in Xor m.
      }
      ```
