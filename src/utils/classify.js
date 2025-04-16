// utils/classify.js
import * as tf from "@tensorflow/tfjs";

/**
 * Preprocesses an image or video frame for classification.
 * @param {HTMLImageElement | HTMLVideoElement} source - The image or video element
 * @param {number[]} inputShape - Model input shape, usually [1, height, width, 3]
 * @returns {tf.Tensor4D}
 */
export const preprocessImage = (source, inputShape) => {
  return tf.tidy(() => {
    return tf.browser.fromPixels(source)
      .resizeBilinear([inputShape[1], inputShape[2]])
      .toFloat()
      .div(255.0)
      .expandDims();
  });
};

/**
 * Runs classification on an input tensor and returns class probabilities.
 * @param {tf.GraphModel} model - The loaded model
 * @param {tf.Tensor4D} tensor - Preprocessed input tensor
 * @returns {Promise<Float32Array>} - Class probability array
 */
export const runPrediction = async (model, tensor) => {
  const predictionTensor = model.predict(tensor);
  const predictionArray = await predictionTensor.data();
  tf.dispose([tensor, predictionTensor]);
  return predictionArray;
};

/**
 * Utility to interpret and label a prediction
 * @param {Float32Array} probs - The prediction probability array
 * @param {string[]} classLabels - Array of class names (e.g., ['Wildfire', 'No Wildfire'])
 * @returns {string} - Label of predicted class
 */
export const getPredictedLabel = (probs, classLabels) => {
  const maxIndex = probs.indexOf(Math.max(...probs));
  return classLabels[maxIndex];
};
