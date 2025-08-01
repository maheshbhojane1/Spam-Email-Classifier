import React, { useEffect, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import { buildVocabulary, textToVector } from "../nlp";

const Classifier = () => {
  const [model, setModel] = useState(null);
  const [vocab, setVocab] = useState([]);
  const [emailText, setEmailText] = useState("");
  const [result, setResult] = useState("");

  useEffect(() => {
    trainModel();
  }, []);

  const trainModel = async () => {
    const response = await fetch("/emails.json");
    const emails = await response.json();

    const vocabList = buildVocabulary(emails);
    setVocab(vocabList);

    const xs = tf.tensor2d(emails.map((e) => textToVector(e.text, vocabList)));
    const ys = tf.tensor2d(emails.map((e) => [e.label]));

    const m = tf.sequential();
    m.add(tf.layers.dense({ units: 16, activation: "relu", inputShape: [vocabList.length] }));
    m.add(tf.layers.dropout({ rate: 0.2 }));
    m.add(tf.layers.dense({ units: 8, activation: "relu" }));
    m.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));

    m.compile({ optimizer: "adam", loss: "binaryCrossentropy", metrics: ["accuracy"] });

    await m.fit(xs, ys, { epochs: 30 });
    console.log("Model trained!");
    setModel(m);
  };

  const handlePredict = async () => {
    if (!model) return;
    const inputVector = tf.tensor2d([textToVector(emailText, vocab)]);
    const prediction = await model.predict(inputVector).data();
    setResult(prediction[0] > 0.5 ? "🚨 Spam Email!" : "✅ Not Spam");
  };

  return (
    <div className="classifier-container">
      <h2>📧 Spam Email Classifier (AI/ML)</h2>
      <textarea
        placeholder="Paste email text here..."
        value={emailText}
        onChange={(e) => setEmailText(e.target.value)}
      />
      <button onClick={handlePredict}>Check Email</button>
      <p className="result">{result}</p>
    </div>
  );
};

export default Classifier;
