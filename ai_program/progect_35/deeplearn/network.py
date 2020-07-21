# %load network.py

"""
network.py
~~~~~~~~~~
IT WORKS

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np

class Network(object): 

    def __init__(self, sizes):  
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)                            #输入[784,30,10],3层。
        self.sizes = sizes                     
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]] #隐藏层和输出层的每个神经元都有一个偏置。[[30,1],[10,1]]，random是生成正态分布的(y,1)维度的随机数。
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]  #隐藏层和输出层的每个神经元的权重与前一层相关 [[30,784],[10,30]]

    def feedforward(self, a): #a是输入的数据，返回经过神经网络的结果。
        """Return the output of the network if ``a`` is input."""
        
        #输出测试
        #print('feedforward==',sigmoid(np.dot(self.weights[0],a)+self.biases[0]))
        #o1 = sigmoid(np.dot(self.weights[0],a)+self.biases[0])
        #o2 = sigmoid(np.dot(self.weights[1],o1)+self.biases[1])
        #print (o2)
        
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,  #迭代次数，小批量数量的大小，n
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""

        training_data = list(training_data) #把zip函数转换为列表，列表的长度是50000，每个元素都是元组，元组是数据和答案。
        n = len(training_data)

        if test_data:  #如果输入测试数据的话。
            test_data = list(test_data)
            n_test = len(test_data)
        

        for j in range(epochs): #每一代
            random.shuffle(training_data)  #先把里面的数据顺序打乱。
            mini_batches = [               #不要被它的分行搞糊涂了，实际上就是初始化list,将训练数据初始化成5000个小的列表
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]  #k是0,10,20,30.....n(训练数据总长度。)
            for mini_batch in mini_batches: #对每一个batch的训练数据进行更新，共计跟新5000次。
                self.update_mini_batch(mini_batch, eta)
                
            #当迭代到第29次时，将其结果保存下来。
            if j==2:
                np.savez('/home/zhangzhipeng/software/github/2020/ai_program/save_weight/weight.npz',out_w=self.weights)
                np.savez('/home/zhangzhipeng/software/github/2020/ai_program/save_weight/biases.npz',out_b=self.biases)
                
            #到此为止，每一次迭代就算是结束了，可以把其训练后的w和b取出来。
            if test_data:
                print("Epoch {} : {} / {}".format(j,self.evaluate(test_data),n_test));
            else:
                print("Epoch {} complete".format(j))
            #print ('j==2',self.biases)

    def update_mini_batch(self, mini_batch, eta):#对每一组的训练数据进行更新？
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]  #与权重和偏置大小一样，只是都初始化为0了。
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch: #x y分别是输入的数据和答案。求一个batch中每一个数据的梯度，然后求和。
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)] #nable_w相当于求batch(10)个数据的梯度和。
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw #
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb  #这里面的for循环是指每个层的循环，更新每层的b值.
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):  #求单独一个数据的梯度，最后返回的是与self.weight和self.b相同的梯度。
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        
        #下面这个for循环使得最后的activations包含每一次的输入(输入、隐藏层输出xigema、最终输出)#是np格式的
        #zs存储2个，隐藏层的非西格玛输出、最终结果的非西格玛输出
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b #每一层的输出(非西格玛函数)
            zs.append(z)
            activation = sigmoid(z)     #每一层的输出(西格玛函数)
            activations.append(activation)
        # backward pass #最后一层的实际输出减去答案然后乘以其西格玛函数的导数。注意是10维的。
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose()) #转置
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers): #一共有3层，最后一层的偏导再上面，这个for循环是从倒数第二层开始的偏导。
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp #注意最后已经是正常的乘法了，不是矩阵乘法。
            nabla_b[-l] = delta #这个是l,不是数字1,注意数字是紫色的.
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y) #np返回输入的x的实际结果，整个test返回实际y和理论y
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results) #如果实际结果于理论结果相同，则等于1，这是个元组，里面是很过个1，求和。

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z): #将w和b的结果变成西格玛函数
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z): #西格玛函数求导
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
