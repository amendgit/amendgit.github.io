---
title: "cs224n assignment 1"
date: 2017-07-12T19:43:09+08:00
mathjax: true
mathjaxEnableSingleDollar: true
---

cs224n（nlp with dl）的assigment 1，纯属个人学习和爱好。

<!--more-->

## 1 Softmax (10 points)

**(a)** (5 points) Prove that softmax is invariant to constant offsets in the input, that is, for any input vector x and any constant c,

$$softmax(x) = softmax(x + c)$$

where x + c means adding the constant c to every dimension of x. Remember that

$$softmax(x)\_i = \frac {e^{x\_i}}{\sum\_je^{x\_j}}$$ 

Note: In practice, we make use of this property and choose c = − maxi xi when computing softmax probabilities for numerical stability (i.e., subtracting its maximum element from all elements of x).

**Solution**

数学中，softmax函数（或者叫归一化指数函数），是逻辑函数sigmoid的推广。它将由实数元素组成K维的向量z，压缩到由属于区间[0,1]的实数组成的K维向量σ(z)且各项元素之和为1。

概率论中，softmax的函数输出通常被用于表示分类分布（categorical distribution，一种由K种可能的输出组成的离散型概率分布）。

$$
\begin{align}
softmax(x - c)\_i =& \frac {e^{x\_i - c}}{\sum\_je^{x\_j - c}} \\
    =& \frac {e^{x\_i} e^{-c}}{\sum_je^{x\_j} e^{-c}} \\
    =& \frac {e^{x\_i} e^{-c}}{(\sum\_je^{x\_j}) e^{-c}} \\
    =& \frac {e^{x\_i}}{\sum\_je^{x\_j}}
\end{align}
$$ 

**(b)** (5 points) Given an input matrix of N rows and D columns, compute the softmax prediction for each row using the optimization in part (a). Write your implementation in q1\_softmax.py. You may test by executing python q1\_softmax.py.
Note: The provided tests are not exhaustive. Later parts of the assignment will reference this code so it is important to have a correct implementation. Your implementation should also be efficient and vectorized whenever possible (i.e., use numpy matrix operations rather than for loops). A non-vectorized implementation will not receive full credit!

**Solution**

```python
def softmax(x):
    """Compute the softmax function for each row of the input x.

    It is crucial that this function is optimized for speed because
    it will be used frequently in later code. You might find numpy
    functions np.exp, np.sum, np.reshape, np.max, and numpy
    broadcasting useful for this task.

    Numpy broadcasting documentation:
    http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html

    You should also make sure that your code works for a single
    N-dimensional vector (treat the vector as a single row) and
    for M x N matrices. This may be useful for testing later. Also,
    make sure that the dimensions of the output match the input.

    You must implement the optimization in problem 1(a) of the
    written assignment!

    Arguments:
    x -- A N dimensional vector or M x N dimensional numpy matrix.

    Return:
    x -- You are allowed to modify x in-place
    """
    orig_shape = x.shape
    ndim = x.ndim
    max = np.max(x, axis=ndim-1, keepdims=True)   # max value of array of each row. (M x 1)
    exp = np.exp(x - max)                         # exp of each element. (M x N)
    sum = np.sum(exp, axis=ndim-1, keepdims=True) # sum of each row. (M x 1)
    x = exp / sum                                 # softmax. (M x N)
    assert x.shape == orig_shape
    return x
```

## 2 Neural Network Basics (30 points)
**(a)** (3 points) Derive the gradients of the sigmoid function and show that it can be rewritten as a function of the function value (i.e., in some expression where only σ(x), but not x, is present). Assume that the input x is a scalar for this question. Recall, the sigmoid function is

$$σ(x) = \frac{1}{1+e^{−x}}$$
   
**Solution**

应用链式求导法则

$$
\begin{align}
\frac {\partial{σ}} {\partial{x}} =& \frac {\partial \frac{1}{1+e^{−x}}}{\partial x} \\
    =& \frac {\partial \frac{1}{1+e^{−x}}}{\partial (1+e^{−x})} \cdot \frac {\partial (1+e^{−x})}{\partial e^{−x}} \cdot \frac{\partial e^{−x}}{\partial (-x)} \cdot \frac{\partial (-x)}{\partial x} \\
    =& \frac{-1}{(1+e^{-x})^2} \cdot 1 \cdot e^{-x} \cdot -1 \\
    =& \frac{e^{-x}}{(1+e^{-x})^2} \\
    =& \frac{1 + e^{-x} - 1} {(1+e^{-x})^2} \\
    =& \frac{1}{1+e^{−x}} - \frac{1} {(1+e^{-x})^2} \\
    =& σ - σ^2 \\
    =& σ(1 - σ) \\
\end{align}
$$

**(b)** (3 points) Derive the gradient with regard to the inputs of a softmax function when cross entropy loss is used for evaluation, i.e., find the gradients with respect to the softmax input vector θ, when the prediction is made by $\hat{y} = softmax(θ)$. Remember the cross entropy function is

$$CE(y,\hat{y}) = − \sum\_i y\_i \log(\hat{y}\_i)$$

where y is the one-hot label vector, and yˆ is the predicted probability vector for all classes. (Hint: you might want to consider the fact many elements of y are zeros, and assume that only the k-th dimension of y is one.)

**Solution**
已知$\hat{y}\_i = \frac{e^{\theta_i}}{\sum_j{e^{\theta_j}}}$且y的为one-hot向量。设y的第k个元素为1，其他元素为0，可以得到：
$$
\begin{align}
CE(\theta) =& − \sum_i y_i \log(\hat{y}\_i) \\
    =& - (0 + y_k\log(\hat{y}\_k) + 0) \\
    =& - \log(\hat{y}\_k) \\
    =& - \log(\frac{e^{\theta_k}}{\sum_j{e^{\theta_j}}}) \\
    =& - \theta_k + \log(\sum_j{e^{\theta_j}})
\end{align}
$$

$$\frac{\partial{CE}}{\partial{\theta}}=- \frac{\theta_k}{\partial{\theta}} + \frac{\partial{\log(\sum_j{e^{\theta_j}})}}{\partial{\theta}}$$

对于第一部分，对向量求导是先对向量中的元素分别求导，再将结果组成向量，维度保持不变。其中， $\frac{\partial{\theta_k}}{\partial{\theta_k}}=1$ 且当 $j \neq k$时$\frac{\partial{\theta_k}}{\partial{\theta_j}}=0$ , 可以得到：

$$\frac{\partial{\theta_k}}{\partial{\theta}}=y$$

对于第二部分，应用链式求导法则，对单个元素进行求导：
$$
\begin{align}
\frac{\partial{\log(\sum_j{e^{\theta_j})}}}{\partial{\theta_i}} 
    &= \frac{\partial{\log(\sum_j{e^{\theta_j})}}}{\sum_j{e^{\theta_j}}} \cdot \frac{\sum_j{e^{\theta_j}}}{\partial{\theta_i}} \\
    &= \frac{1}{\sum_j{e^{\theta_j}}} \cdot e^{\theta_i} \\
    &= \frac{e^{\theta_i}}{\sum_j{e^{\theta_j}}} \\
    &= \hat{y}\_i \\
\end{align}
$$

将第一部分和第二部分组合在一起：
$$\frac{\partial{CE}}{\partial{\theta}}=- \frac{\theta_k}{\partial{\theta}} + \frac{\partial{\log(\sum_j{e^{\theta_j}})}}{\partial{\theta}}=-y + \hat{y}$$

**(c)** (6 points) Derive the gradients with respect to the *inputs* x to an one-hidden-layer neural network (that is,find $\frac{\partial{J}}{\partial{x}}$ where $J=CE(y,yˆ)$ is the cost function for the neural network).The neural network employs sigmoid activation function for the hidden layer, and softmax for the output layer. Assume the one-hot label vector is y, and cross entropy cost is used. (Feel free to use σ′(x) as the shorthand for sigmoid gradient, and feel free to define any variables whenever you see fit.)

![one-hidden-layer-nn](/cs224n/one-hidden-layer-nn.png)

   Recall that the forward propagation is as follows 
   
$$h = sigmoid(xW_1 + b_1)$$
$$\hat{y} = softmax(hW_2 + b_2) $$
   
Note that here we’re assuming that the input vector (thus the hidden variables and output probabilities) is a row vector to be consistent with the programming assignment. When we apply the sigmoid function to a vector, we are applying it to each of the elements of that vector. **$W_i$** and **$b_i$** (i = 1, 2) are the weights and biases, respectively, of the two layers.

**Solution**
设    
$$z_1 = xW_1 + b_1$$
$$z_2 = hW_2 + b_2$$

已知
$$h = sigmoid(xW_1 + b_1) = sigmoid(z_1)$$
$$\hat{y} = softmax(hW_2 + b_2) = softmax(z_2)$$
$$CE(y,\hat{y}) = − \sum_i y_i \log(\hat{y}\_i)$$

可以得到：
$$
\begin{align}
\frac{\partial{J}}{\partial{x}} =& \frac{\partial{CE}}{\partial{x}} \\
    =& \frac{\partial{CE}}{\partial{z_2}} \cdot \frac{\partial{z_2}}{\partial{h}} \cdot \frac{\partial{h}}{\partial{z_1}} \cdot \frac{\partial{z_1}}{\partial{x}} \\
    =& (\hat{y} - y) \cdot W_2 \cdot (z_1 - z_1^2) \cdot W_1
\end{align}
$$

**(d)** (2 points) How many parameters are there in this neural network, assuming the input is Dx-dimensional, the output is Dy-dimensional, and there are H hidden units?

**Solution**

$$SUM = H \cdot (D_x + 1) + D_y \cdot (H + 1)$$

**(e)** (4 points) Fill in the implementation for the sigmoid activation function and its gradient in `q2_sigmoid.py`. Test your implementation using `python q2_sigmoid.py`. *Again, thoroughly test your code as the provided tests may not be exhaustive.*

**Solution**

```python
def sigmoid(x):
    """
    Compute the sigmoid function for the input here.

    Arguments:
    x -- A scalar or numpy array.

    Return:
    s -- sigmoid(x)
    """
    s = 1 / (1 + np.exp(-x))
    return s


def sigmoid_grad(s):
    """
    Compute the gradient for the sigmoid function here. Note that
    for this implementation, the input s should be the sigmoid
    function value of your original input x.

    Arguments:
    s -- A scalar or numpy array.

    Return:
    ds -- Your computed gradient.
    """
    ds = s * (1 - s)
    return ds
```

**(f)** (4 points) To make debugging easier, we will now implement a gradient checker. Fill in the implementation for gradcheck naive in `q2_gradcheck.py`. Test your code using `python q2_gradcheck.py`.

**Solution**

```python
# First implement a gradient checker by filling in the following functions
def gradcheck_naive(f, x):
    """ Gradient check for a function f.

    Arguments:
    f -- a function that takes a single argument and outputs the
         cost and its gradients
    x -- the point (numpy array) to check the gradient at
    """

    rndstate = random.getstate()
    random.setstate(rndstate)
    fx, grad = f(x) # Evaluate function value at original point
    h = 1e-4        # Do not change this!

    # Iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        # Try modifying x[ix] with h defined above to compute
        # numerical gradients. Make sure you call random.setstate(rndstate)
        # before calling f(x) each time. This will make it possible
        # to test cost functions with built in randomness later.
        store = x[ix]
        x[ix] = store + h
        random.setstate(rndstate)
        a = f(x)[0]
        x[ix] = store - h
        random.setstate(rndstate)
        b = f(x)[0]
        x[ix] = store
        numgrad = (a - b) / (2 * h)

        # Compare gradients
        reldiff = abs(numgrad - grad[ix]) / max(1, abs(numgrad), abs(grad[ix]))
        if reldiff > 1e-5:
            print "Gradient check failed."
            print "First gradient error found at index %s" % str(ix)
            print "Your gradient: %f \t Numerical gradient: %f" % (
                grad[ix], numgrad)
            return

        it.iternext() # Step to next dimension

    print "Gradient check passed!"
```
**(g)** (8 points) Now, implement the forward and backward passes for a neural network with one sigmoid hidden layer. Fill in your implementation in `q2_neural.py`. Sanity check your implementation with `python q2_neural.py`.

**Solution**

已知
$$CE(y, \hat{y}) = \sum_i{y_i\log{\hat{y}\_i}}$$

$$\hat{y} = a\_2 = softmax(z\_2) = \frac{exp(z2)}{\sum\_i{exp(z_{2i})}}$$

$$z_2=a_1W_2 + b_2$$

$$a_1 = sigmoid(z_1)$$

$$z_1 = a_0W_1 + b_1$$

$$a_0 = x$$

$$\frac{\partial{CE}}{\partial{z2}} = gradz2 = -y + \hat{y}$$

对W2求偏导，gradW2:
$$
\begin{align}
\frac{\partial{CE}}{\partial{W_2}} =& \frac{\partial{CE}}{\partial{z_2}} \cdot \frac{\partial{z_2}}{\partial{W_2}} \\
    =& a_1^T \cdot gradz2
\end{align}
$$

对b2求导，gradb2:
$$
\begin{align}
\frac{\partial{CE}}{\partial{b_2}} =& \frac{\partial{CE}}{\partial{z_2}} \cdot \frac{\partial{z_2}}{\partial{b_2}} \\
    =& gradz2
\end{align}
$$

对a1求导，grada1:
$$
\begin{align}
\frac{\partial{CE}}{\partial{a_1}} =& \frac{\partial{CE}}{\partial{z_2}} \cdot \frac{\partial{z_2}}{\partial{a_1}} \\
    =& W_2^T \cdot gradz2
\end{align}
$$

对z1求导，gradz1:
$$
\begin{align}
\frac{\partial{CE}}{\partial{z_1}} =& \frac{\partial{CE}}{\partial{a_1}} \cdot \frac{\partial{a_1}}{\partial{z_1}} \\
    =& sigmoid'(a_1) \cdot grada1 \\
\end{align}
$$

对W1求导，gradW1:
$$
\begin{align}
\frac{\partial{CE}}{\partial{W_1}} =& \frac{\partial{CE}}{\partial{z_1}} \cdot \frac{\partial{z_1}}{\partial{W_1}} \\
    =& a_0^T \cdot gradz1 \\
    =& x^T \cdot gradz1 \\
\end{align}
$$

对b1求导，gradb1:
$$
\begin{align}
\frac{\partial{CE}}{\partial{b_1}} =& \frac{\partial{CE}}{\partial{z_1}} \cdot \frac{\partial{z_1}}{\partial{b_1}} \\
    =& gradz1 
\end{align}
$$

```python
def forward_backward_prop(data, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.

    Arguments:
    data -- M x Dx matrix, where each row is a training example.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs + Dx * H], (Dx, H))      # Dx * H
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))            # 1 * H
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))      # H * Dy
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    x = data                                                # M * Dx
    y = labels                                              # M * Dy
    M = x.shape[0]

    ### forward propagation
    z1 = np.dot(x, W1) + b1
    a1 = sigmoid(z1)

    z2 = np.dot(a1, W2) + b2
    y_hat = a2 = softmax(z2)

    cost = -np.sum(np.log(a2[np.arange(M), np.argmax(y, axis=1)])) # Cross Entropy

    ### backward propagation
    gradz2 = y_hat - y
    gradW2 = np.dot(a1.T, gradz2)
    gradb2 = np.sum(gradz2, axis=0)
    grada2 = np.dot(gradz2, W2.T)
    gradz1 = sigmoid_grad(a1) * grada2
    gradW1 = np.dot(x.T, gradz1)
    gradb1 = np.sum(gradz1, axis=0)

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))

    return cost, grad
```

## 3 word2vec (40 points + 2 bonus)
**(a)** (3 points) Assume you are given a predicted word vector vc corresponding to the center word c for skipgram, and word prediction is made with the softmax function found in word2vec models

$$\hat{y}\_o=p(o \mathbin{\vert} c)=\frac{e^{u\_o^T v\_c}}{\sum_{w=1}^W{e^{u\_w^T v\_c}}}$$

where w denotes the w-th word and $u_w$ (w = 1, . . . , W ) are the “output” word vectors for all words in the vocabulary. Assume cross entropy cost is applied to this prediction and word o is the expected word (the o-th element of the one-hot label vector is one), derive the gradients with respect to $v_c$.

*Hint: It will be helpful to use notation from question 2. For instance, letting y_hat be the vector of softmax predictions for every word, y as the expected word vector, and the loss function* 

$$J_{softmax−CE}(o,vc,U) = CE(y,yˆ)$$

where $U = [u_1 , u_2 , · · · , u_W]$ is the matrix of all the output vectors. Make sure you state the orientation of your vectors and matrices.

**Solution 1:**

设$ z_i = u_i^T v_c $，则：

$$ z = U^Tv_c $$

$$
\begin{align}
J =& - \sum\_{i=1}^W{y\_i\log(\frac{exp(u\_i^T v\_c)}{\sum\_{w=1}^{W}{exp(u\_w^T v\_c)}})} \\
    =& - \sum\_{i=1}^W{y\_i\log(\frac{exp(z\_i)}{\sum_{w=1}^{W}{exp(z\_w)}})}
\end{align}
$$

可以得到:

$$
\begin{align}
\frac{\partial{J}}{\partial{v_c}} =& \frac{\partial{CE}}{\partial{z}}  \cdot \frac{\partial{z}}{\partial{v_c}} \\
    =&\frac{\partial{CE}}{\partial{z}} \cdot \frac{\partial{U^Tv_c}}{\partial{v_c}} \\
    =& U^T \cdot (\hat{y} - y)
\end{align}
$$

**Solution 2:**

1. |W|是词汇表中的单词数量
2. y和$\hat{y}$是|W| x 1的列向量
3. $u_i$和$v_j$是D x 1的列向量(D输入输出的向量的维度)
4. y是|W| x 1的one-hot编码列向量 
5. $\hat{y}$是|W| x 1的softmax输出的列向量 
6. y hat: $\hat{y\_i} = P(i|c) = \frac {exp(u\_i^T v\_c)} { \sum\_{w=1}^W{exp(u\_w^T v\_c)} }$
7. 交叉熵损失函数: $J = -\sum_{i=1}^Wy_ilog({\hat{y_i}})$
8. $U = [u_1, u_2, ...,u_k, ...u_W]$是由$u_k$列向量组成的矩阵

损失函数，

$$J = - \sum\_{i=1}^W y\_i log(\frac{exp(u\_i^Tv\_c)}{\sum\_{w=1}^Wexp(u\_w^Tv\_c)})$$

简化,

$$J = - \sum\_{i=1}^Wy\_i[u\_i^Tv\_c - log(\sum\_{w=1}^Wexp(u\_w^Tv\_c))]$$

由于y是one-hot编码的列向量，除了第k个元素之外的其他元素都是0。也就是说，上面的求和公式中，只有第k个是非0的，其他的元素均为0。所以，损失函数可以写为：

$$J = -y\_k[u\_k^Tv\_c - log(\sum\_{w=1}^Wexp(u\_w^Tv\_c))]$$

注意, 其中$y_k$为1。求解$\frac{\partial{J}}{\partial{v_c}}$:

$$\frac{\partial J}{\partial v\_c} = -[u\_k - \frac{\sum\_{w=1}^Wexp(u\_w^Tv\_c)u\_w}{\sum\_{x=1}^Wexp(u\_x^Tv\_c)}]$$

使用定义(5)，我们可以重写上述公式：

$$\frac{\partial J}{\partial v\_c} = \sum_{w=1}^W (\hat{y}\_w u\_w) - u\_k$$

现在我们将公式使用矩阵形式重写：
1. $u\_k$可以被写为矩阵向量相乘: $U \cdot y$
2. 且$\sum\_{w=1}^W (\hat{y}\_w u\_w)$是矩阵U中的向量$u\_w$按照比例$\hat{y}\_w$缩放的的一个线性转换。

所以，公式整体可以写成：$U[\hat{y} - y]$

**(b)** (3 points) As in the previous part, derive gradients for the “output” word vectors $u_w$’s (including $u_o$).

**Solution:**

$$
\begin{align}
\frac{\partial{J}}{\partial{U}} =& \frac{\partial{CE}}{\partial{z}} \cdot \frac{\partial{z}}{\partial{U}} \\
    =& v\_c \cdot (\hat{y} - y)^T
\end{align}
$$

**(c)** (6 points) Repeat part (a) and (b) assuming we are using the negative sampling loss for the predicted vector vc, and the expected output word is o. Assume that K negative samples (words) are drawn, and they are 1, · · · , K, respectively for simplicity of notation ($o \notin \{1, . . . , K\}$). Again, for a given word, o, denote its output vector as $u_o$. The negative sampling loss function in this case is K 

$$J\_{neg−sample}(o,v\_c,U)=−\log(σ(u^⊤\_o v\_c)) − \sum\_{k=1}^{K}\log(σ(−u^⊤\_k v\_c))$$

where σ(·) is the sigmoid function.
After you’ve done this, describe with one sentence why this cost function is much more efficient to compute than the softmax-CE loss (you could provide a speed-up ratio, i.e., the runtime of the softmax- CE loss divided by the runtime of the negative sampling loss).
*Note: the cost function here is the negative of what Mikolov et al had in their original paper, because we are doing a minimization instead of maximization in our code.*

**Solution:**

对vc求偏导:
$$
\begin{align}
\frac{\partial{J}}{\partial{v\_c}} =& - \frac{1}{σ(u\_o^T v\_c)} \cdot σ(u\_o^T v\_c)(1-σ(u\_o^T v\_c)) \cdot u\_o - \sum\_{k=1}^{K}{\frac{1}{σ(-u\_k^T v\_c)} \cdot σ(-u\_k^T v\_c)(1-σ(-u\_k^T v\_c)) \cdot -u\_k} \\
    =& - (1 - σ(u\_o^T v\_c)) \cdot u\_o - \sum\_{k=1}^{K}{(1 - σ(-u\_k^T v\_c)) \cdot -u\_k} \\
    =& (σ(u\_o^T v\_c) - 1) \cdot u\_o - \sum\_{k=1}^{K}{(σ(-u\_k^T v\_c) - 1) \cdot u\_k}
\end{align}
$$

对uo求偏导:
$$
\begin{align}
\frac{\partial{J}}{\partial{u_o}} =& - \frac{1}{σ(u_o^T v_c)} \cdot σ(u_o^T v_c)(1-σ(u_o^T v_c)) \cdot v_c - 0 \\
    =& (σ(u_o^T v_c) - 1) \cdot v_c
\end{align}
$$

对uk求偏导:
$$
\begin{align}
\frac{\partial{J}}{\partial{u_k}} =& - 0 - \frac{1}{σ(-u_k^T v_c)} \cdot σ(-u_k^T v_c)(1-σ(-u_k^T v_c)) \cdot - v_c \\
    =& - (σ(-u_k^T v_c) - 1) \cdot v_c
\end{align}
$$

**(d)** (8 points) Derive gradients for all of the word vectors for skip-gram and CBOW given the previous parts and given a set of context words $$ word\_{c-m}, ... , word\_{c-1}, word\_{c}, word\_{c+1}, ... word\_{c+m} $$, where m is the context size. Denote the “input” and “output” word vectors for $word_k$ as $v_k$ and $u_k$ respectively.

*Hint: feel free to use $$F(o, v\_c)$$ (where **o** is the expected word) as a placeholder for the $$J_{softmax−CE}(o, v_c , ...)$$ or $$J\_{neg−sample}(o,v\_c,...)$$ cost functions in this part — you’ll see that this is a useful abstraction for the coding part. That is, your solution may contain terms of the form $$\frac{\partial{F(o,v_c)}}{\partial{...}}$$*

Recall that for skip-gram, the cost for a context centered around c is
$$J\_{skip-gram}(word\_{c−m...c+m}) = \sum\_{-m \le j \le m, j \ne 0}F(w\_{c+j},v\_c)$$
where $w\_{c+j}$ refers to the word at the j-th index from the center.
CBOW is slightly different. Instead of using $v\_c$ as the predicted vector, we use $\hat{v}$ defined below. For (a simpler variant of) CBOW, we sum up the input word vectors in the context
$$\hat{v} = \sum\_{-m \le j \le m, j \ne 0}{v\_{c+j}}$$
then the CBOW cost is
$$J\_{CBOW}(word\_{c−m...c+m}) = F(w\_c, \hat{v})$$
*Note: To be consistent with the $\hat{v}$ notation such as for the code portion, for skip-gram $\hat{v} = v_c$.*

**Solution:**

假设U是由词汇表中所有单词的输出向量组成的矩阵. 

已知$\frac{\partial{F(w_j, \hat{v})}}{\partial{U}}$和$\frac{\partial{F(w_j, \hat{v}}}{\partial{\hat{v}}}$

对于skip-gram模型进行求偏导，对U求偏导:
$$
\frac{\partial{J\_{skip-gram}(word\_{c-m...c+m})}}{\partial{U}} = \sum\_{-m \le j \le m, j \ne 0}\frac{\partial{F(w\_{c+j}, \hat{v})}}{\partial{U}}
$$

对vc求偏导:
$$
\frac{\partial{J\_{skip-gram}(word\_{c-m...c+m})}}{\partial{v\_c}} = \sum\_{-m \le j \le m, j \ne 0}\frac{\partial{F(w\_{c+j}, \hat{v})}}{\partial{v\_c}}
$$

对vj求偏导:
$$
\frac{\partial{J\_{skip-gram}(word\_{c-m...c+m})}}{\partial{v\_j}} = 0,\ for\ all\ j \ne c
$$

对CBOW模型进行求偏导，对U求偏导:
$$
\frac{\partial{J\_{CBOW}(word\_{c-m...c+m})}}{\partial{U}} = \frac{\partial{F(w, \hat{v})}}{\partial{U}}
$$

对v hat求偏导:
$$
\frac{\partial{J\_{CBOW}(word\_{c-m...c+m})}}{\partial{v\_j}} = \frac{\partial{F(w, \hat{v})}}{\partial{\hat{v}}}, for\ all\ j \in \{c-m, ... , c+m\}
$$

对vj求偏导:
$$
\frac{\partial{J\_{CBOW}(word\_{c-m...c+m})}}{\partial{v\_j}} = 0, for\ all\ j \notin \{c-m, ... , c+m\}
$$

**(e)** (12 points) In this part you will implement the word2vec models and train your own word vectors with stochastic gradient descent (SGD). First, write a helper function to normalize rows of a matrix in q3_word2vec.py. In the same file, fill in the implementation for the softmax and negative sampling cost and gradient functions. Then, fill in the implementation of the cost and gradient functions for the skip-gram model. When you are done, test your implementation by running python q3_word2vec.py. *Note: If you choose not to implement CBOW (part h), simply remove the NotImplementedError so that your tests will complete.*

**Solution:**

代码: [q3_word2vec.py](https://github.com/amendgit/cs224n/blob/master/assignment1/code/q3_word2vec.py)

normlize:

```python
def normalizeRows(x):
    """ Row normalization function

    Implement a function that normalizes each row of a matrix to have
    unit length.
    """

    y = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / y

    return x
```

softmax:

```python
def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """ Softmax cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, assuming the softmax prediction function and cross
    entropy loss.

    Arguments:
    predicted -- numpy ndarray, predicted word vector (\hat{v} in
                 the written component)
    target -- integer, the index of the target word
    outputVectors -- "output" vectors (as rows) for all tokens
    dataset -- needed for negative sampling, unused here.

    Return:
    cost -- cross entropy cost for the softmax word prediction
    gradPred -- the gradient with respect to the predicted word
           vector
    grad -- the gradient with respect to all the other word
           vectors

    We will not provide starter code for this function, but feel
    free to reference the code you previously wrote for this
    assignment!
    """
    grad = np.zeros_like(outputVectors)
    gradPred = np.zeros_like(predicted)

    prob = softmax(np.dot(predicted, outputVectors.T))
    cost = - np.log(prob[target])

    gradZ = prob
    gradZ[target] -= 1 # y_hat - y

    N, D = outputVectors.shape

    grad     = np.dot(gradZ.reshape(N, 1), predicted.reshape(1, D))
    gradPred = (np.dot(gradZ.reshape(1, N), outputVectors)).flatten()

    return cost, gradPred, grad
```

negative sampling:

```python    
def negSamplingCostAndGradient(predicted, target, outputVectors, dataset,
                               K=10):
    """ Negative sampling cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, using the negative sampling technique. K is the sample
    size.

    Note: See test_word2vec below for dataset's initialization.

    Arguments/Return Specifications: same as softmaxCostAndGradient
    """

    # Sampling of indices is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    indices = [target]
    indices.extend(getNegativeSamples(target, dataset, K))

    U = outputVectors
    vc = predicted

    grad = np.zeros_like(U)
    gradPred = np.zeros_like(vc)

    N, D = U.shape

    labels = np.array([1] + [-1 for k in range(K)])
    u = U[indices] # uo, uk1, uk2, ... , uK

    z = np.dot(u, vc) * labels
    probs = sigmoid(z)
    cost = - np.sum(np.log(probs))

    gradZ    = labels * (probs - 1)
    gradPred = gradZ.reshape(1, K+1).dot(u).flatten()
    gradu    = gradZ.reshape(K+1, 1).dot(vc.reshape(1, vc.shape[0]))

    for k in range(K+1):
        grad[indices[k]] += gradu[k,:]

    return cost, gradPred, grad
```

skip-gram

```python
def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
             dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """ Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    currrentWord -- a string of the current center word
    C -- integer, context size
    contextWords -- list of no more than 2*C strings, the context words
    tokens -- a dictionary that maps words to their indices in
              the word vector list
    inputVectors -- "input" word vectors (as rows) for all tokens
    outputVectors -- "output" word vectors (as rows) for all tokens
    word2vecCostAndGradient -- the cost and gradient function for
                               a prediction vector given the target
                               word vectors, could be one of the two
                               cost functions you implemented above.

    Return:
    cost -- the cost function value for the skip-gram model
    grad -- the gradient with respect to the word vectors
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    cIndex = tokens[currentWord]
    predicted = inputVectors[cIndex, :]

    for contextWord in contextWords:
        target = tokens[contextWord]
        cCost, cGradPred, cGrad = word2vecCostAndGradient(predicted, target, outputVectors, dataset)
        cost += cCost
        gradIn[cIndex,:] += cGradPred
        gradOut += cGrad

    return cost, gradIn, gradOut
```

**(f)** (4 points) Complete the implementation for your SGD optimizer in q3_sgd.py. Test your implementation by running python q3_sgd.py.

**Solution:**

完整代码: [q3_sgd.py](https://github.com/amendgit/cs224n/blob/master/assignment1/code/q3_sgd.py)

```python
def sgd(f, x0, step, iterations, postprocessing=None, useSaved=False,
        PRINT_EVERY=10):
    """ Stochastic Gradient Descent

    Implement the stochastic gradient descent method in this function.

    Arguments:
    f -- the function to optimize, it should take a single
         argument and yield two outputs, a cost and the gradient
         with respect to the arguments
    x0 -- the initial point to start SGD from
    step -- the step size for SGD
    iterations -- total iterations to run SGD for
    postprocessing -- postprocessing function for the parameters
                      if necessary. In the case of word2vec we will need to
                      normalize the word vectors to have unit length.
    PRINT_EVERY -- specifies how many iterations to output loss

    Return:
    x -- the parameter value after SGD finishes
    """

    # Anneal learning rate every several iterations
    ANNEAL_EVERY = 20000

    if useSaved:
        start_iter, oldx, state = load_saved_params()
        if start_iter > 0:
            x0 = oldx
            step *= 0.5 ** (start_iter / ANNEAL_EVERY)

        if state:
            random.setstate(state)
    else:
        start_iter = 0

    x = x0

    if not postprocessing:
        postprocessing = lambda x: x

    expcost = None

    for iter in xrange(start_iter + 1, iterations + 1):
        # Don't forget to apply the postprocessing after every iteration!
        # You might want to print the progress every few iterations.

        cost, grad = f(x)
        x = x - step * grad
        x = postprocessing(x)

        if iter % PRINT_EVERY == 0:
            if not expcost:
                expcost = cost
            else:
                expcost = .95 * expcost + .05 * cost
            print "iter %d: %f" % (iter, expcost)

        if iter % SAVE_PARAMS_EVERY == 0 and useSaved:
            save_params(iter, x)

        if iter % ANNEAL_EVERY == 0:
            step *= 0.5

    return x
```

**(g)** (4 points) Show time! Now we are going to load some real data and train word vectors with everything you just implemented! We are going to use the Stanford Sentiment Treebank (SST) dataset to train word vectors, and later apply them to a simple sentiment analysis task. You will need to fetch the datasets first. To do this, run sh get datasets.sh. There is no additional code to write for this part; just run python q3_run.py.
*Note: The training process may take a long time depending on the efficiency of your implementation (an efficient implementation takes approximately an hour). Plan accordingly!
When the script finishes, a visualization for your word vectors will appear. It will also be saved as q3_word_vectors.png in your project directory. Include the plot in your homework write up. Briefly explain in at most three sentences what you see in the plot.*

**Solution:**

![q3_word_vectors](/cs224n/q3_word_vectors-2.png)

**(h)** (Extra credit: 2 points) Implement the CBOW model in q3_word2vec.py. Note: This part is optional but the gradient derivations for CBOW in part (d) are not!.

**Solution:**

continue bag of words:

```python
def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
         dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """CBOW model in word2vec

    Implement the continuous bag-of-words model in this function.

    Arguments/Return specifications: same as the skip-gram model

    Extra credit: Implementing CBOW is optional, but the gradient
    derivations are not. If you decide not to implement CBOW, remove
    the NotImplementedError.
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    cIndex = tokens[currentWord]
    predicted = inputVectors[cIndex, :]

    for contextWord in contextWords:
        target = tokens[contextWord]
        cCost, cGradPred, cGrad = word2vecCostAndGradient(predicted, target, outputVectors, dataset)
        cost += cCost
        gradIn[cIndex,:] += cGradPred
        gradOut += cGrad

    return cost, gradIn, gradOut
```

## 4 Sentiment Analysis (20 points)
Now, with the word vectors you trained, we are going to perform a simple sentiment analysis. For each sentence in the Stanford Sentiment Treebank dataset, we are going to use the average of all the word vectors in that sentence as its feature, and try to predict the sentiment level of the said sentence. The sentiment level of the phrases are represented as real values in the original dataset, here we’ll just use five classes:
“very negative (−−)”, “negative (−)”, “neutral”, “positive (+)”, “very positive (++)”
which are represented by 0 to 4 in the code, respectively. For this part, you will learn to train a softmax classifier, and perform train/dev validation to improve generalization.

**(a)** (2 points) Implement a sentence featurizer. A simple way of representing a sentence is taking the average of the vectors of the words in the sentence. Fill in the implementation in  q4_sentiment.py.

**Solution:**

```python
def getSentenceFeatures(tokens, wordVectors, sentence):
    """
    Obtain the sentence feature for sentiment analysis by averaging its
    word vectors
    """

    # Implement computation for the sentence features given a sentence.

    # Inputs:
    # tokens -- a dictionary that maps words to their indices in
    #           the word vector list
    # wordVectors -- word vectors (each row) for all tokens
    # sentence -- a list of words in the sentence of interest

    # Output:
    # - sentVector: feature vector for the sentence

    sentVector = np.zeros((wordVectors.shape[1],))

    listOfInd = [tokens[w] for w in sentence]
    for i in listOfInd:
        sentVector += wordVectors[i]
    sentVector /= len(listOfInd)

    assert sentVector.shape == (wordVectors.shape[1],)
    return sentVector
```

regularization: 

```python
def getRegularizationValues():
    """Try different regularizations

    Return a sorted list of values to try.
    """
    values = None   # Assign a list of floats in the block below
    values = np.logspace(-4, 2, num=100, base=10)
    return sorted(values)
```

**(b)** (1 points) Explain in at most two sentences why we want to introduce regularization when doing classi-fication (in fact, most machine learning tasks).

**Solution:**

避免在训练集上overfitting，而在测试集上效果过差。

**(c)** (2 points) Fill in the hyperparameter selection code in q4_sentiment.py to search for the “optimal” regularization parameter. Attach your code for chooseBestModel to your written write-up. You should be able to attain at least 36.5% accuracy on the dev and test sets using the pretrained vectors in part (d).

**Solution:**

```python
def chooseBestModel(results):
    """Choose the best model based on parameter tuning on the dev set

    Arguments:
    results -- A list of python dictionaries of the following format:
        {
            "reg": regularization,
            "clf": classifier,
            "train": trainAccuracy,
            "dev": devAccuracy,
            "test": testAccuracy
        }

    Returns:
    Your chosen result dictionary.
    """
    bestResult = None
    bestResult = max(results, key=lambda model: model["dev"])
    return bestResult
```

**(d)** (3 points) Run python q4_sentiment.py --yourvectors to train a model using your word vectors from q3. Now, run python q4_sentiment.py --pretrained to train a model using pretrained GloVe vectors (on Wikipedia data). Compare and report the best train, dev, and test accuracies. Why do you think the pretrained vectors did better? Be specific and justify with 3 distinct reasons.

**Solution:**
Higher dimensional word vectors may encode more infomation. GloVe vectors were trained on a much larger corpus. GloVe vs Word2Vec.

**(e)** (4 points) Plot the classification accuracy on the train and dev set with respect to the regularization value for the pretrained GloVe vectors, using a logarithmic scale on the x-axis. This should have been done automatically. Include q4_reg_acc.png in your homework write up. Briefly explain in at most three sentences what you see in the plot.

**Solution:**
![q4_reg_v_ac](/cs224n/q4_reg_v_acc.png)

**(f)** (4 points) We will now analyze errors that the model makes (with pretrained GloVe vectors). When you ran python q4_sentiment.py --pretrained, two files should have been generated. Take a look at q4_dev_conf.png and include it in your homework writeup. Interpret the confusion matrix in at most three sentences.

**Solution:**
![q4_dev_conf](/cs224n/q4_dev_conf.png)

**(g)** (4 points) Next, take a look at q4_dev_pred.txt. Choose 3 examples where your classifier made errors and briefly explain the error and what features the classifier would need to classify the example correctly (1 sentence per example). Try to pick examples with different reasons.

**Solution:**


