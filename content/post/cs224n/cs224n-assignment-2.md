---
title: "cs224n assignment 2"
date: 2017-09-12T22:52:06+08:00
tags: ["cs224n"]
mathjax: true
mathjaxEnableSingleDollar: true
---

自学cs224n（nlp with dl）的assigment 2，纯属个人学习和爱好。

<!--more-->

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

(a) 手动为"I parsed this sentence correctly"执行dependency parse. 


| stack                          | buffer                                 | new dependency      | transition          |
| ------------------------------ | -------------------------------------- | ------------------- | ------------------- |
| [ROOT]                         | [I, parsed, this, sentence, correctly] |                     | Initial Configation |
| [ROOT, I]                      | [parsed, this, sentence, correctly]    |                     | SHIFT               |
| [ROOT, I, parsed]              | [this, sentence, correctly]            |                     | SHIFT               |
| [ROOT, parsed]                 | [this, sentence, correctly]            | parsed -> I         | LEFT-ARC            |
| [ROOT, parsed, this]           | [sentence, correctly]                  |                     | SHIFT               |
| [ROOT, parsed, this, sentence] | [correctly]                            |                     | SHIFT               |
| [ROOT, parsed, sentence]       | [correctly]                            | sentence -> this    | LEFT-ARC            |
| [ROOT, parsed]                 | [correctly]                            | parsed -> sentence  | RIGHT-ARC           |
| [ROOT, parsed, correctly]      | []                                     |                     | SHIFT               |
| [ROOT, parsed]                 | []                                     | parsed -> correctly | SHIFT               |
| [ROOT]                         | []                                     | ROOT -> parsed      | RIGHT-SHIFT         |


(b) 包含n个words的sentence，进行dependency parse共需要多少步？

2n步，每个word入栈一步和每个word出栈一步，共2n步。

(c) 补充完整`q2_parser_transitions.py`的`__init__`和`parse_step`方法，实现dependency parse.

方法`PartialParse.__init__`的实现如下

```python
def __init__(self, sentence):
    """Initializes this partial parse.

    Your code should initialize the following fields:
        self.stack: The current stack represented as a list with the top of the stack as the
                    last element of the list.
        self.buffer: The current buffer represented as a list with the first item on the
                     buffer as the first item of the list
        self.dependencies: The list of dependencies produced so far. Represented as a list of
                tuples where each tuple is of the form (head, dependent).
                Order for this list doesn't matter.

    The root token should be represented with the string "ROOT"

    Args:
        sentence: The sentence to be parsed as a list of words.
                  Your code should not modify the sentence.
    """
    # The sentence being parsed is kept for bookkeeping purposes. Do not use it in your code.
    self.sentence = sentence
    self.stack = ['ROOT']
    self.buffer = self.sentence[:]
    self.dependencies = []
```

方法`PartialParse.parse_step`方法的实现如下

```python
def parse_step(self, transition):
    """Performs a single parse step by applying the given transition to this partial parse

    Args:
        transition: A string that equals "S", "LA", or "RA" representing the shift, left-arc,
                    and right-arc transitions.
    """
    if transition == 'S':
        self.stack.append(self.buffer[0])
        del self.buffer[0]
    elif transition == 'LA':
        self.dependencies.append((self.stack[-1], self.stack[-2]))
        del self.stack[-2]
    elif transition == 'RA':
        self.dependencies.append((self.stack[-2], self.stack[-1]))
        del self.stack[-1]

```

运行结果

```sh
 2018-08-18 22:10:26 ⌚  |2.2.2| erxiangbo in ~/courses/others/amendgit/assignment2/code
± |master {1} U:1 ?:1 ✗| → python q2_parser_transitions.py
SHIFT test passed!
LEFT-ARC test passed!
RIGHT-ARC test passed!
```

(d) 神经网络对批量数据做处理总体效率会更高，实现`minibatch_parse`方法。

```python
def minibatch_parse(sentences, model, batch_size):
    """Parses a list of sentences in minibatches using a model.

    Args:
        sentences: A list of sentences to be parsed (each sentence is a list of words)
        model: The model that makes parsing decisions. It is assumed to have a function
               model.predict(partial_parses) that takes in a list of PartialParses as input and
               returns a list of transitions predicted for each parse. That is, after calling
                   transitions = model.predict(partial_parses)
               transitions[i] will be the next transition to apply to partial_parses[i].
        batch_size: The number of PartialParses to include in each minibatch
    Returns:
        dependencies: A list where each element is the dependencies list for a parsed sentence.
                      Ordering should be the same as in sentences (i.e., dependencies[i] should
                      contain the parse for sentences[i]).
    """
    partial_parses = [PartialParse(sentence) for sentence in sentences]
    unfinished_parses = partial_parses[:]
    while len(unfinished_parses) > 0:
        batch_parses = unfinished_parses[0:batch_size]
        transitions = model.predict(batch_parses)
        for i, parse in enumerate(batch_parses):
            parse.parse_step(transitions[i])
            if len(parse.stack) <= 1 or len(parse.stack) <= 0:
                unfinished_parses.remove(parse)
    return [parse.dependencies for parse in partial_parses]
```

运行结果：

```sh
 2018-08-19 01:42:20 ⌚  |2.2.2| erxiangbo in ~/courses/others/amendgit/assignment2/code
± |master {1} U:1 ?:1 ✗| → python q2_parser_transitions.py
SHIFT test passed!
LEFT-ARC test passed!
RIGHT-ARC test passed!
parse test passed!
minibatch_parse test passed!
```

现在我们来构建一个神经网络来根据stack、buffer和dependencies，来predict下一步的transition。首先，model导出一个feature vector来表示当前的状态。我们将用最初的神经依赖分析的论文《A Fast and Accurate Dependecing Parser using Neural Networking》中的feature set。这个导出特征的的方法已经预先在`parser_utils`中实现好了

(e) 为了避免神经元之间corelation，实现Xavier Initialization。完善`q2_initialization.py`的`_xavier_initializer`方法。

```python
def _xavier_initializer(shape, **kwargs):
    """Defines an initializer for the Xavier distribution.
    Specifically, the output should be sampled uniformly from [-epsilon, epsilon] where
        epsilon = sqrt(6) / <sum of the sizes of shape's dimensions>
    e.g., if shape = (2, 3), epsilon = sqrt(6 / (2 + 3))

    This function will be used as a variable initializer.

    Args:
        shape: Tuple or 1-d array that species the dimensions of the requested tensor.
    Returns:
        out: tf.Tensor of specified shape sampled from the Xavier distribution.
    """
    epsilon = np.sqrt(6 / np.sum(shape))
    out = tf.Variable(tf.random_uniform(shape=shape, minval=-epsilon, maxval=epsilon))
    return out
# Returns defined initializer function.
return _xavier_initializer
```

运行结果：

```sh
 2018-08-19 10:59:22 ⌚  |2.2.2| erxiangbo in ~/courses/others/amendgit/assignment2/code
± |master {1} U:2 ?:1 ✗| → python q2_initialization.py
Running basic tests...
Basic (non-exhaustive) Xavier initialization tests pass
```

(f) 使用Dropout对神经网络进行正则化 (题目详见pdf)

$$
E\_{P\_{drop}}[h\_{drop}]\_i = E\_{P\_{drop}}[\gamma d\_i h\_i] = P\_{drop}(0) + (1 - P\_{drop}) \gamma h_i = (1 - P\_{drop}) \gamma h_i = h_i
$$

所以，常数gamma的值为 $1 / (1 - P\_{drop}) $
