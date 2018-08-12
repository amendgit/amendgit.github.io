---
title: "cs224n assignment 2"
date: 2017-09-12T22:52:06+08:00
---

## 1 Tensorflow Softmax

本题主要是实现一个线性分类器，以交叉熵作为损失函数。

(a) 使用tensorflow实现softmax

```python
def softmax(x):
    """
    Compute the softmax function in tensorflow.

    You might find the tensorflow functions tf.exp, tf.reduce_max,
    tf.reduce_sum, tf.expand_dims useful. (Many solutions are possible, so you may
    not need to use all of these functions). Recall also that many common
    tensorflow operations are sugared (e.g. x * y does a tensor multiplication
    if x and y are both tensors). Make sure to implement the numerical stability
    fixes as in the previous homework!

    Args:
        x:   tf.Tensor with shape (n_samples, n_features). Note feature vectors are
                  represented by row-vectors. (For simplicity, no need to handle 1-d
                  input as in the previous homework)
    Returns:
        out: tf.Tensor with shape (n_sample, n_features). You need to construct this
                  tensor in this problem.
    """

    exp = tf.exp(x - tf.reduce_min(x, axis=1, keep_dims=True))
    out = exp / tf.reduce_sum(exp, axis=1, keep_dims=True)
    
    return out
```

(b) 使用tensorflow实现cross entropy

```python
def cross_entropy_loss(y, yhat):
    """
    Compute the cross entropy loss in tensorflow.
    The loss should be summed over the current minibatch.

    y is a one-hot tensor of shape (n_samples, n_classes) and yhat is a tensor
    of shape (n_samples, n_classes). y should be of dtype tf.int32, and yhat should
    be of dtype tf.float32.

    The functions tf.to_float, tf.reduce_sum, and tf.log might prove useful. (Many
    solutions are possible, so you may not need to use all of these functions).

    Note: You are NOT allowed to use the tensorflow built-in cross-entropy
                functions.

    Args:
        y:    tf.Tensor with shape (n_samples, n_classes). One-hot encoded.
        yhat: tf.Tensor with shape (n_samples, n_classes). Each row encodes a
                    probability distribution and should sum to 1.
    Returns:
        out:  tf.Tensor with shape (1,) (Scalar output). You need to construct this
                    tensor in the problem.
    """

    out = - tf.reduce_sum(tf.to_float(y) * tf.log(yhat))
    
    return out
```

(c) 阅读 model.py，解释 tensorflow的 placeholders 和 feed dictionaries，完成 q1_classifier.py 的 add_placeholders 和 create_feed_dict 方法

**placeholder**：tensorflow使用placeholder表示计算图中数据被插入的位置，placehodlers将会作为构建模型其他部分的输入，并且会在训练模型的过程中被填充数据。

**feeddictionaries**：feed_dictionaries是在训练的过程中，作为填充placeholders的数据。dictionry的key应该是placeholders的子集，value是填充到对应的placeholder的数据。

```python
class SoftmaxModel(Model):
    """Implements a Softmax classifier with cross-entropy loss."""

    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors.

        These placeholders are used as inputs by the rest of the model building
        and will be fed data during training.

        Adds following nodes to the computational graph

        input_placeholder: Input placeholder tensor of shape
                                              (batch_size, n_features), type tf.float32
        labels_placeholder: Labels placeholder tensor of shape
                                              (batch_size, n_classes), type tf.int32

        Add these placeholders to self as the instance variables
            self.input_placeholder
            self.labels_placeholder
        """
        self.input_placeholder = tf.placeholder(tf.float32, shape=(self.config.batch_size, self.config.n_features))
        self.labels_placeholder = tf.placeholder(tf.int32, shape=(self.config.batch_size, self.config.n_classes))
        
    def create_feed_dict(self, inputs_batch, labels_batch=None):
        """Creates the feed_dict for training the given step.

        A feed_dict takes the form of:
        feed_dict = {
                <placeholder>: <tensor of values to be passed for placeholder>,
                ....
        }

        If label_batch is None, then no labels are added to feed_dict.

        Hint: The keys for the feed_dict should be the placeholder
                tensors created in add_placeholders.

        Args:
            inputs_batch: A batch of input data.
            labels_batch: A batch of label data.
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        feed_dict = {
            self.input_placeholder: inputs_batch,
            self.labels_placeholder: labels_batch,
        }
        return feed_dict
```

(d) 完成q1_classifier.py里的add_prediction_op方法

```python
def add_prediction_op(self):
   """Adds the core transformation for this model which transforms a batch of input
   data into a batch of predictions. In this case, the transformation is a linear layer plus a
   softmax transformation:

   y = softmax(Wx + b)

   Hint: Make sure to create tf.Variable as needed.
   Hint: For this simple use-case, it's sufficient to initialize both weights W
               and biases b with zeros.

   Args:
       input_data: A tensor of shape (batch_size, n_features).
   Returns:
       pred: A tensor of shape (batch_size, n_classes)
   """
   W = tf.Variable(tf.zeros([self.config.n_features, self.config.n_classes]))
   b = tf.Variable(tf.zeros([self.config.batch_size, self.config.n_classes]))
   pred = softmax(tf.matmul(self.input_placeholder, W) + b)
   return pred
```

(e) 完成q1_classifier.py里的add_training_op方法

```python
def add_training_op(self, loss):
   """Sets up the training Ops.

   Creates an optimizer and applies the gradients to all trainable variables.
   The Op returned by this function is what must be passed to the
   `sess.run()` call to cause the model to train. See

   https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

   for more information.

   Hint: Use tf.train.GradientDescentOptimizer to get an optimizer object.
               Calling optimizer.minimize() will return a train_op object.

   Args:
       loss: Loss tensor, from cross_entropy_loss.
   Returns:
       train_op: The Op for training.
   """
   optimizer = tf.train.GradientDescentOptimizer(self.config.lr)
   train_op = optimizer.minimize(loss)
   return train_op
```

## 2 Neural Transition-Based Dependency Parsing

实现一个基于神经网络的依赖解析器

(a) 


| stack                          | buffer                                 | new dependency      | transition          |
| ------------------------------ | -------------------------------------- | ------------------- | ------------------- |
| [ROOT]                         | [I, parsed, this, sentence, correctly] |                     | Initial Configation |
| [ROOT, I]                      | [parsed, this, sentence, correctly]    |                     | SHIFT               |
| [ROOT, I, parsed]              | [this, sentence, correctly]            |                     | SHIFT               |
| [ROOT, parsed]                 | [this, sentence, correctly]            | parsed -> I         | LEFT-ARC            |
| [ROOT, parsed, this]           | [sentence, correctly]                  |                     | SHIFT               |
| [ROOT, parsed, this, sentence] | [correctly]                            |                     | SHIFT               |
| [ROOT, parsed, sentence]       | [correctly]                            | sentence -> parsed  | LEFT-ARC            |
| [ROOT, parsed]                 | [correctly]                            | parsed -> sentence  | RIGHT-ARC           |
| [ROOT, parsed, correctly]      | []                                     |                     | SHIFT               |
| [ROOT, parsed]                 | []                                     | parsed -> correctly | SHIFT               |
| [ROOT]                         | []                                     | ROOT -> parsed      | RIGHT-SHIFT         |


## (b)

