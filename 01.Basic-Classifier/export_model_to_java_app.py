import tensorflow as tf
from basic_classifier_model import BasicClassifierModel


def _export(python_model_path, java_model_path):
    # 1. Init model.
    model = BasicClassifierModel(1, 2)
    model.create(0.001)

    # 2.Load model.
    variables = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(variables)
        last_ckpt_path = tf.train.latest_checkpoint(python_model_path)
        if last_ckpt_path is not None:
            saver = tf.train.Saver(max_to_keep=None)
            saver.restore(sess, last_ckpt_path)
        else:
            print('Not found the model.')
            return None

        # 3.Convert model.
        builder = tf.saved_model.builder.SavedModelBuilder(java_model_path)

        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={
                    'x_input': tf.saved_model.utils.build_tensor_info(
                        model.input)
                },
                outputs={
                    'y_output': tf.saved_model.utils.build_tensor_info(
                        model.output)
                }))

        print(prediction_signature)
        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    prediction_signature,
            })
        builder.save()
        print ('Finish.')


if __name__ == '__main__':
    _export('./model/', './java_model/')