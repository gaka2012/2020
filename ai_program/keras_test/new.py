#!/usr/bin/python
# -*- coding:UTF-8 -*-

import tensorflow as tf


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #忽略弹出的警告。

price = {'price': [[1.], [2.], [3.], [4.]]}  # 4行样本

column = tf.feature_column.numeric_column('price', normalizer_fn=lambda x:x+2)
tensor = tf.feature_column.input_layer(price,[column])

with tf.Session() as session:
    print(session.run([tensor]))
