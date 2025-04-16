import React, { useState, useEffect, useRef } from "react";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl"; // set backend to webgl
import Loader from "./components/loader";
import ButtonHandler from "./components/btn-handler";
import {
  preprocessImage,
  runPrediction,
  getPredictedLabel
} from './utils/classify';
import "./style/App.css";

const App = () => {
  const [loading, setLoading] = useState({ loading: true, progress: 0 }); // loading state
  const [model, setModel] = useState({
    net: null,
    inputShape: [1, 0, 0, 3],
  }); // init model & input shape

  const [prediction, setPrediction] = useState(null);
  const [probabilities, setProbabilities] = useState([]);

  // references
  const imageRef = useRef(null);
  const cameraRef = useRef(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  // model configs
  const modelName = "best";

  useEffect(() => {
    tf.ready().then(async () => {
      const yolov8 = await tf.loadGraphModel(
        //IMPORTANT NOTE: This  --> ${window.location.href} --> was originally where the "./" is. I changed it to "./" because for the hugging face deployment, the path was wrong. It needs to directly go into best_web_model folder, use relative path. 
        `./${modelName}_web_model/model.json`,
        {
          onProgress: (fractions) => {
            setLoading({ loading: true, progress: fractions }); // set loading fractions
          },
        }
      ); // load model

      // warming up model
      const dummyInput = tf.ones(yolov8.inputs[0].shape);
      const warmupResults = yolov8.execute(dummyInput);

      setLoading({ loading: false, progress: 1 });
      setModel({
        net: yolov8,
        inputShape: yolov8.inputs[0].shape,
      }); // set model & input shape

      tf.dispose([warmupResults, dummyInput]); // cleanup memory
    });
  }, []);

  const handleImagePrediction = async () => {
    if (!model.net || !imageRef.current) return;
  
    const tensor = preprocessImage(imageRef.current, model.inputShape);
    const probs = await runPrediction(model.net, tensor);
    const label = getPredictedLabel(probs, ["No Wildfire", "Wildfire"]);
  
    setPrediction(label);
    setProbabilities(probs);
  };

  const handleVideoPrediction = async () => {
    if (!model.net || !videoRef.current) return;
  
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
  
    const predictLoop = async () => {
      const tensor = preprocessImage(videoRef.current, model.inputShape);
      const probs = await runPrediction(model.net, tensor);
      const label = getPredictedLabel(probs, ["No Wildfire", "Wildfire"]);
  
      setPrediction(label);
      setProbabilities(probs);
    };

    predictLoop();
  };
    

  return (
    <div className="App">
      {loading.loading && (
        <Loader>Loading model... {(loading.progress * 100).toFixed(2)}%</Loader>
      )}
      <div className="header">
        <h1>Wildfire detection of Satellite Images</h1>
      </div>

      <div className="content">
        <img
          src="#"
          ref={imageRef}
          onLoad= {handleImagePrediction}
        />
        <video
   ref={videoRef}
   autoPlay
   playsInline
   muted
   width={model.inputShape[1]}
   height={model.inputShape[2]}
   onPlay={handleVideoPrediction}
        />
        <canvas
          width={model.inputShape[1]}
          height={model.inputShape[2]}
          ref={canvasRef}
        />

{prediction && (
        <div className="prediction-display">
          <h2>Prediction: {prediction}</h2>
          <ul>
            <li>No Wildfire: {(probabilities[0] * 100).toFixed(2)}%</li>
            <li>Wildfire: {(probabilities[1] * 100).toFixed(2)}%</li>
          </ul>
        </div>
      )}

      </div>

      

      <ButtonHandler
        imageRef={imageRef}
        cameraRef={cameraRef}
        videoRef={videoRef}
      />
    </div>
  );
};

export default App;
