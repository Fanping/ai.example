import tensorflow as tf

if __name__ == '__main__':
    # 1. Define model.
    tf.reset_default_graph()
    W = tf.get_variable('w', initializer=tf.constant(2.0), dtype=tf.float32)
    x = tf.placeholder(tf.float32, name='x')
    y = tf.multiply(W, x, name='y')

    # 2. Save model.
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    builder = tf.saved_model.builder.SavedModelBuilder(
        "./ai.example.tensorflow.java/model")
    builder.add_meta_graph_and_variables(
        sess,
        [tf.saved_model.tag_constants.SERVING]
    )
    builder.save()
    # INFO:If you want to convert any model to java, please see 01.Basic-Classifier/export_model_to_java_app.py
