/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

export interface NeuralLayer {
  name: string;
  type: string; // e.g. "Conv2D", "MaxPooling2D", "Flatten", "Dense", "Dropout"
  params: string; // e.g. "filters: 32, kernelSize: 3, activation: relu"
}

export interface TrainingParams {
  learningRate: number;
  epochs: number;
  batchSize: number;
  optimizer: string; // "Adam" | "SGD" | "RMSprop"
  lossFunction: string; // "categoricalCrossentropy" | "binaryCrossentropy"
}

export interface VideoAnalysisResult {
  videoSubject: string;
  suggestedClasses: string[];
  insights: string[];
  modelPy: string;
  tfjsLayerConfig: NeuralLayer[];
  trainingParameters: TrainingParams;
}

export interface UploadedVideo {
  id: string;
  name: string;
  size: string;
  url: string; // Object URL for preview
  type: string;
  duration: number; // in seconds
  frames: string[]; // base64 representation of extracted visual frames
  thumbnail: string; // thumbnail base64
}

export interface EpochLog {
  epoch: number;
  loss: number;
  acc: number;
  valLoss: number;
  valAcc: number;
}
