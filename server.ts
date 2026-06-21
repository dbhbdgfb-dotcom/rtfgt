import express from "express";
import path from "path";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI, Type } from "@google/genai";
import dotenv from "dotenv";

dotenv.config();

const app = express();
const PORT = 3000;

// Set high limits for uploaded base64 video frame packets
app.use(express.json({ limit: "50mb" }));
app.use(express.urlencoded({ limit: "50mb", extended: true }));

// Lazy-initialize Gemini SDK
let aiClient: GoogleGenAI | null = null;
function getAIClient(): GoogleGenAI {
  if (!aiClient) {
    const key = process.env.GEMINI_API_KEY;
    if (!key) {
      throw new Error("GEMINI_API_KEY is missing in your environment variables. Please add it via AI Studio settings.");
    }
    aiClient = new GoogleGenAI({
      apiKey: key,
      httpOptions: {
        headers: {
          'User-Agent': 'aistudio-build',
        }
      }
    });
  }
  return aiClient;
}

// Strip data URL prefixes for Gemini inlineData
function cleanBase64(dataUrl: string): { data: string; mimeType: string } {
  const matches = dataUrl.match(/^data:([^;]+);base64,(.+)$/);
  if (matches && matches.length === 3) {
    return {
      mimeType: matches[1],
      data: matches[2],
    };
  }
  return {
    mimeType: "image/jpeg",
    data: dataUrl,
  };
}

// API endpoint to analyze extracted video frames and generate customized AI model architecture
app.post("/api/analyze-frames", async (req, res) => {
  try {
    const { frames, prompt: userPrompt } = req.body;

    if (!frames || !Array.isArray(frames) || frames.length === 0) {
      return res.status(400).json({ error: "No video frames supplied for analysis" });
    }

    const ai = getAIClient();

    // Prepare vision parts
    const parts: any[] = [];
    frames.slice(0, 5).forEach((frameUrl, index) => {
      try {
        const cleaned = cleanBase64(frameUrl);
        parts.push({
          inlineData: {
            mimeType: cleaned.mimeType,
            data: cleaned.data,
          }
        });
      } catch (err) {
        console.error(`Error processing frame at index ${index}:`, err);
      }
    });

    // Instructions on how to build a highly optimized deep learning video trainer model
    const systemInstructions = `You are an expert Computer Vision scientist and Deep Learning teacher.
Analyze the uploaded video material (supplied as a chronological sequence of video frames) and auto-generate a targeted deep learning model configuration.
Your output must be a valid JSON object matching the requested schema.

The user aims to train a classifier on the action, subject, or objects in this video. 
Assess the frames:
1. Identify the core video category/domain (e.g. human motion, vehicle tracking, background event, cooking, sporting).
2. Generate suggested labels (at least 2 to 5 distinct labels) to feed into the training pipeline.
3. Choose an ideal deep learning architecture (e.g., CNN, ResNet-backbone + LSTM/GRU sequence model, simple MobileNet feature extractor with dense layers).
4. Provide customized code: 
   - A PyTorch model definition 'modelPy' containing a complete, executable Python module with self-contained dataset loader and training code tailored to this input topic/classes! Include comments explaining the hyperparameter selection.
   - A sequence of layers 'tfjsLayerConfig' depicting standard CNN neural layers (with input shapes, filter sizes, activation functions, and layer types) to display in the frontend's visual training designer and to render standard models.
5. Provide scientific model hyperparameters appropriate for this video type.`;

    const modelPrompt = `For this video dataset ${userPrompt ? `with goal: "${userPrompt}"` : 'of captured footage'}:
Analyze these frames, design an architectural network layout, prescribe label classifications, suggest hyperparameter details, and write PyTorch & JSON configs. Ensure all Python code is complete, annotated, and fully executable.`;

    parts.push({ text: modelPrompt });

    const response = await ai.models.generateContent({
      model: "gemini-3.5-flash",
      contents: { parts },
      config: {
        systemInstruction: systemInstructions,
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          required: ["videoSubject", "suggestedClasses", "insights", "modelPy", "tfjsLayerConfig", "trainingParameters"],
          properties: {
            videoSubject: {
              type: Type.STRING,
              description: "A summary classification or subject description of the video, e.g. Basketball Shot Practice."
            },
            suggestedClasses: {
              type: Type.ARRAY,
              items: { type: Type.STRING },
              description: "Array of detected target categories or actions to train a classification/detection model on. Minimum 2."
            },
            insights: {
              type: Type.ARRAY,
              items: { type: Type.STRING },
              description: "High-level scientific observations about the frame series (e.g. lighting conditions, motion speed, recommended input sizes)."
            },
            modelPy: {
              type: Type.STRING,
              description: "Complete, continuous, outstanding PyTorch training code with a custom neural net, validation step, and inference loop in raw python."
            },
            tfjsLayerConfig: {
              type: Type.ARRAY,
              description: "Structural sequence of layers depicting a sequential convolutional network.",
              items: {
                type: Type.OBJECT,
                required: ["name", "type", "params"],
                properties: {
                  name: { type: Type.STRING, description: "Display name, e.g. Conv2D_Layer_1" },
                  type: { type: Type.STRING, description: "Type, e.g., Conv2D, MaxPooling2D, Flatten, Dense, Dropout" },
                  params: { type: Type.STRING, description: "Concise string describing values, e.g. 'filters: 32, kernelSize: 3, activation: relu'" }
                }
              }
            },
            trainingParameters: {
              type: Type.OBJECT,
              required: ["learningRate", "epochs", "batchSize", "optimizer", "lossFunction"],
              properties: {
                learningRate: { type: Type.NUMBER, description: "Suggested alpha rate, e.g. 0.001" },
                epochs: { type: Type.INTEGER, description: "Suggested training epoch loops, e.g. 15" },
                batchSize: { type: Type.INTEGER, description: "Optimal training batch size, e.g. 8" },
                optimizer: { type: Type.STRING, description: "Suggested optimizer parameter, e.g., 'Adam', 'SGD', 'RMSprop'" },
                lossFunction: { type: Type.STRING, description: "Suggested loss metric, e.g., 'categoricalCrossentropy', 'binaryCrossentropy'" }
              }
            }
          }
        }
      }
    });

    const parsedData = JSON.parse(response.text || "{}");
    res.json(parsedData);
  } catch (error: any) {
    console.error("Error during frame analysis:", error);
    res.status(500).json({ error: error.message || "An unexpected error occurred during analysis." });
  }
});

// Configure Vite middleware in development, or serve statically in production
async function startServer() {
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`AI Video Trainer server active on http://0.0.0.0:${PORT}`);
  });
}

startServer();
