


import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt





sess=tf.Session()
norm=tf.random_normal([100],mean=1, stddev=2)
np_A=np.array(sess.run(norm))



plt.scatter(range(len(np_A)), np_A, color='r')
plt.show()


avg_a=np.mean(np_A)
std_a=np.std(np_A)
print(avg_a, std_a)





with tf.name_scope('Input_placeholder'):
    a = tf.placeholder(tf.float32, shape=None, name="input_a")
   
    
with tf.name_scope("Middle_Section"):
        b=tf.reduce_prod(a, name="b_prod")
        c=tf.reduce_mean(a, name="c_mean")
        d=tf.reduce_sum(a, name="d_Sum")
        e = tf.add(b, c, name='e_add')
        
with tf.name_scope('Final_node'):
    f = tf.add(d, e, name='f')   
         
        
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


print(sess.run(f, {a: np_A}))              
   
    
writer=tf.summary.FileWriter('./graph_HW2', graph=tf.get_default_graph())
writer.close()

