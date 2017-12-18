package com.example.ai.tensorflow.service;

import org.springframework.stereotype.Service;
import org.tensorflow.Graph;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.TensorFlow;

import java.io.IOException;
import java.nio.FloatBuffer;
import java.nio.file.Paths;

@Service
public class MultiplyService {


    private Session session;

    public MultiplyService() {
        session = SavedModelBundle.load("./model", "serve").session();
    }

    public float[] get(final float[] val) {
        int num_of_predictions = val.length;
        Tensor x = Tensor.create(
                new long[]{num_of_predictions},
                FloatBuffer.wrap(val)
        );
        float[] y = session.runner()
                .feed("x", x)
                .fetch("y")
                .run()
                .get(0)
                .copyTo(new float[num_of_predictions]);
        return y;
    }
}
