import numpy as np
import random
import tensorflow as tf
import datetime

#read data
text = open('GOT.txt').read()
print('text length in number of characters:', len(text))

print('head of text:')
print(text[:1000])
chars = sorted(list(set(text)))
char_size = len(chars)

char2id = dict((c, i) for i, c in enumerate(chars))
id2char = dict((i, c) for i, c in enumerate(chars))

print('number of characters:', char_size)
print(chars)

def sample(prediction):
    r = random.uniform(0,1)
    s = 0
    char_id = len(prediction) - 1
    for i in range(len(prediction)):
        s += prediction[i]
        if s >= r:
            char_id = i
            break
    char_one_hot = np.zeros(shape=[char_size])
    char_one_hot[char_id] = 1.0
    return char_one_hot

#creating training data
len_per_section = 50
skip = 2

sections = []
next_chars = []
for i in range(0, len(text) - len_per_section, skip):
    sections.append(text[i: i + len_per_section])
    next_chars.append(text[i + len_per_section])

#Vectorize input and output
X = np.zeros((len(sections), len_per_section, char_size))
y = np.zeros((len(sections), char_size))
for i, section in enumerate(sections):
    for j, char in enumerate(section):
        X[i, j, char2id[char]] = 1
    y[i, char2id[next_chars[i]]] = 1


batch_size = 512
max_steps = 144001
log_every = 3000
test_every = 6000
hidden_nodes = 2048
test_start = 'Winter is coming'
checkpoint_directory = 'ckpt'

#Create a checkpoint directory
if tf.gfile.Exists(checkpoint_directory):
    tf.gfile.DeleteRecursively(checkpoint_directory)
tf.gfile.MakeDirs(checkpoint_directory)
print('training data size:', len(X))
print('approximate steps per epoch:', int(len(X)/batch_size))

#create the model
graph = tf.Graph()
with graph.as_default():
    
    #Variables and placeholders
    global_step = tf.Variable(0)
    
    data = tf.placeholder(tf.float32, [batch_size, len_per_section, char_size])
    labels = tf.placeholder(tf.float32, [batch_size, char_size])
    
    #Prep LSTM Operation
    #Input gate: weights for input, weights for previous output, and bias
    w_ii = tf.Variable(tf.truncated_normal([char_size, hidden_nodes], -0.1, 0.1))
    w_io = tf.Variable(tf.truncated_normal([hidden_nodes, hidden_nodes], -0.1, 0.1))
    b_i = tf.Variable(tf.zeros([1, hidden_nodes]))
    #Forget gate: weights for input, weights for previous output, and bias
    w_fi = tf.Variable(tf.truncated_normal([char_size, hidden_nodes], -0.1, 0.1))
    w_fo = tf.Variable(tf.truncated_normal([hidden_nodes, hidden_nodes], -0.1, 0.1))
    b_f = tf.Variable(tf.zeros([1, hidden_nodes]))
    #Output gate: weights for input, weights for previous output, and bias
    w_oi = tf.Variable(tf.truncated_normal([char_size, hidden_nodes], -0.1, 0.1))
    w_oo = tf.Variable(tf.truncated_normal([hidden_nodes, hidden_nodes], -0.1, 0.1))
    b_o = tf.Variable(tf.zeros([1, hidden_nodes]))
    #Memory cell: weights for input, weights for previous output, and bias
    w_ci = tf.Variable(tf.truncated_normal([char_size, hidden_nodes], -0.1, 0.1))
    w_co = tf.Variable(tf.truncated_normal([hidden_nodes, hidden_nodes], -0.1, 0.1))
    b_c = tf.Variable(tf.zeros([1, hidden_nodes]))
    #LCTM Cell
    def lstm(i, o, state):
        input_gate = tf.sigmoid(tf.matmul(i, w_ii) + tf.matmul(o, w_io) + b_i)
        forget_gate = tf.sigmoid(tf.matmul(i, w_fi) + tf.matmul(o, w_fo) + b_f)
        output_gate = tf.sigmoid(tf.matmul(i, w_oi) + tf.matmul(o, w_oo) + b_o)
        memory_cell = tf.sigmoid(tf.matmul(i, w_ci) + tf.matmul(o, w_co) + b_c)
        state = forget_gate * state + input_gate * memory_cell
        output = output_gate * tf.tanh(state)
        return output, state
    
    
    #LSTM
    output = tf.zeros([batch_size, hidden_nodes])
    state = tf.zeros([batch_size, hidden_nodes])

    for i in range(len_per_section):
        output, state = lstm(data[:, i, :], output, state)
        if i == 0:
            outputs_all_i = output
            labels_all_i = data[:, i+1, :]
        elif i != len_per_section - 1:
            outputs_all_i = tf.concat( [outputs_all_i, output],0)
            labels_all_i = tf.concat([labels_all_i, data[:, i+1, :]],0)
        else:
            outputs_all_i = tf.concat( [outputs_all_i, output],0)
            labels_all_i = tf.concat( [labels_all_i, labels],0)
        
    #Classifier
    w = tf.Variable(tf.truncated_normal([hidden_nodes, char_size], -0.1, 0.1))
    b = tf.Variable(tf.zeros([char_size]))
    logits = tf.matmul(outputs_all_i, w) + b
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels= labels_all_i))

    #Optimizer
    optimizer = tf.train.GradientDescentOptimizer(10.).minimize(loss, global_step=global_step)
    
    
    #Test
    
    test_data = tf.placeholder(tf.float32, shape=[1, char_size])
    test_output = tf.Variable(tf.zeros([1, hidden_nodes]))
    test_state = tf.Variable(tf.zeros([1, hidden_nodes]))
    
    #Reset at the beginning of each test
    reset_test_state = tf.group(test_output.assign(tf.zeros([1, hidden_nodes])), 
                                test_state.assign(tf.zeros([1, hidden_nodes])))

    #LSTM
    test_output, test_state = lstm(test_data, test_output, test_state)
    test_prediction = tf.nn.softmax(tf.matmul(test_output, w) + b)

#training the data
with tf.Session(graph=graph) as sess:
    tf.global_variables_initializer().run()
    offset = 0
    saver = tf.train.Saver()
    
    for step in range(max_steps):
        offset = offset % len(X)
        if offset <= (len(X) - batch_size):
            batch_data = X[offset: offset + batch_size]
            batch_labels = y[offset: offset + batch_size]
            offset += batch_size
        else:
            to_add = batch_size - (len(X) - offset)
            batch_data = np.concatenate((X[offset: len(X)], X[0: to_add]))
            batch_labels = np.concatenate((y[offset: len(X)], y[0: to_add]))
            offset = to_add
        _, training_loss = sess.run([optimizer, loss], feed_dict={data: batch_data, labels: batch_labels})
        
        if step % log_every == 0:
            print('training loss at step %d: %.2f (%s)' % (step, training_loss, datetime.datetime.now()))

            if step % test_every == 0:
                reset_test_state.run()
                test_generated = test_start
                
                for i in range(len(test_start) - 1):
                    test_X = np.zeros((1, char_size))
                    test_X[0, char2id[test_start[i]]] = 1.
                    _ = sess.run(test_prediction, feed_dict={test_data: test_X})
                
                test_X = np.zeros((1, char_size))
                test_X[0, char2id[test_start[-1]]] = 1.
                
                for i in range(500):
                    prediction = test_prediction.eval({test_data: test_X})[0]
                    next_char_one_hot = sample(prediction)
                    next_char = id2char[np.argmax(next_char_one_hot)]
                    test_generated += next_char
                    test_X = next_char_one_hot.reshape((1, char_size))
                    
                print('=' * 80)
                print(test_generated)
                print('=' * 80)
                
                saver.save(sess, checkpoint_directory + '/model', global_step=step)

#generate new text
test_start = 'You know nothing. Jon snow'

with tf.Session(graph=graph) as sess:
    tf.global_variables_initializer().run()
    model = tf.train.latest_checkpoint(checkpoint_directory)
    saver = tf.train.Saver()
    saver.restore(sess, model)

    reset_test_state.run()
    test_generated = test_start

    for i in range(len(test_start) - 1):
        test_X = np.zeros((1, char_size))
        test_X[0, char2id[test_start[i]]] = 1.
        _ = sess.run(test_prediction, feed_dict={test_data: test_X})

    test_X = np.zeros((1, char_size))
    test_X[0, char2id[test_start[-1]]] = 1.

    for i in range(1000):
        prediction = test_prediction.eval({test_data: test_X})[0]
        next_char_one_hot = sample(prediction)
        next_char = id2char[np.argmax(next_char_one_hot)]
        test_generated += next_char
        test_X = next_char_one_hot.reshape((1, char_size))

    print(test_generated)
