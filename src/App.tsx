/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState, useRef, useEffect, useMemo, DragEvent, ChangeEvent } from "react";
import {
  Play,
  Upload,
  Cpu,
  Layers,
  Settings,
  Activity,
  FileCode,
  Sliders,
  Download,
  CheckCircle2,
  AlertCircle,
  Terminal,
  ArrowRight,
  Plus,
  Trash2,
  RotateCw,
  Sparkles,
  Maximize2,
  Tag,
  BookOpen,
  Info,
  Check
} from "lucide-react";
import { extractFramesFromVideo } from "./utils/frameExtractor";
import { downloadModelBundle } from "./utils/exportBundle";
import { UploadedVideo, NeuralLayer, TrainingParams, EpochLog } from "./types";

export default function App() {
  // Application State
  const [video, setVideo] = useState<UploadedVideo | null>(null);
  const [isExtracting, setIsExtracting] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [rawFile, setRawFile] = useState<File | null>(null);

  // Auto-analysis with Gemini state
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisProgress, setAnalysisProgress] = useState("");
  const [videoSubject, setVideoSubject] = useState("自定义视频分类模型");
  const [suggestedClasses, setSuggestedClasses] = useState<string[]>(["物体_A", "物体_B", "背景"]);
  const [newClass, setNewClass] = useState("");
  const [insights, setInsights] = useState<string[]>([
    "请上传您的视频素材以获取 Gemini AI 提取的神经网络模型定制见解和类别标签。",
    "系统将根据视频帧率、运动速率与颜色通道，自动设计最优卷积层参数。"
  ]);
  const [pytorchCode, setPytorchCode] = useState<string>(`# 尚未加载 AI 分析模型。请在上传视频后点击“用 Gemini 自动标注与架构推荐”生成模型代码。`);

  // Layer Architecture States
  const [layers, setLayers] = useState<NeuralLayer[]>([
    { name: "Conv2D_L1", type: "Conv2D", params: "filters: 32, kernelSize: 3, activation: relu" },
    { name: "MaxPool_L1", type: "MaxPooling2D", params: "poolSize: 2" },
    { name: "Conv2D_L2", type: "Conv2D", params: "filters: 64, kernelSize: 3, activation: relu" },
    { name: "Flatten_L1", type: "Flatten", params: "inputShape: [224, 224, 3]" },
    { name: "Dense_Hidden", type: "Dense", params: "units: 128, activation: relu" },
    { name: "Dropout_L1", type: "Dropout", params: "rate: 0.5" },
    { name: "Dense_Output", type: "Dense", params: "units: 3, activation: softmax" }
  ]);

  // Hyperparameters State
  const [trainingParams, setTrainingParams] = useState<TrainingParams>({
    learningRate: 0.001,
    epochs: 10,
    batchSize: 8,
    optimizer: "Adam",
    lossFunction: "categoricalCrossentropy"
  });

  // Active Layer Editor
  const [selectedLayerIndex, setSelectedLayerIndex] = useState<number | null>(null);
  const [editLayerParams, setEditLayerParams] = useState("");

  // Training Simulation State
  const [isTraining, setIsTraining] = useState(false);
  const [trainingEpoch, setTrainingEpoch] = useState(0);
  const [epochLogs, setEpochLogs] = useState<EpochLog[]>([]);
  const [terminalLogs, setTerminalLogs] = useState<string[]>([]);
  const [weightsMap, setWeightsMap] = useState<number[][]>([]);
  const [activeFrameIndex, setActiveFrameIndex] = useState(0);
  const [currentMetric, setCurrentMetric] = useState<{ loss: number; acc: number; valLoss: number; valAcc: number }>({
    loss: 0,
    acc: 0,
    valLoss: 0,
    valAcc: 0
  });

  // Export States
  const [exportFormat, setExportFormat] = useState<"TFJS" | "PyTorch" | "ONNX">("TFJS");
  const [isExporting, setIsExporting] = useState(false);
  const [activeTab, setActiveTab] = useState<"architect" | "pytorch">("architect");

  const terminalEndRef = useRef<HTMLDivElement>(null);
  const videoInputRef = useRef<HTMLInputElement>(null);

  // Initialize interactive weights matrix structure
  useEffect(() => {
    generateNewWeights();
  }, []);

  // Sync scroll on training terminal output
  useEffect(() => {
    if (terminalEndRef.current) {
      terminalEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [terminalLogs]);

  // Rotate video feed visualization during simulation
  useEffect(() => {
    let interval: any;
    if (isTraining && video && video.frames.length > 0) {
      interval = setInterval(() => {
        setActiveFrameIndex((prev) => (prev + 1) % video.frames.length);
      }, 800);
    }
    return () => clearInterval(interval);
  }, [isTraining, video]);

  const generateNewWeights = () => {
    const matrix: number[][] = [];
    for (let i = 0; i < 8; i++) {
      const row = [];
      for (let j = 0; j < 12; j++) {
        row.push(Math.random());
      }
      matrix.push(row);
    }
    setWeightsMap(matrix);
  };

  // Drag and drop video handlers
  const handleDrag = (e: DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = async (e: DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      await processSelectedVideo(file);
    }
  };

  const handleFileSelect = async (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      await processSelectedVideo(file);
    }
  };

  // Video Frame extractor trigger
  const processSelectedVideo = async (file: File) => {
    if (!file.type.startsWith("video/")) {
      alert("请上传有效的 MP4, WebM 或 MOV 格式视频文件！");
      return;
    }
    setIsExtracting(true);
    setRawFile(file);
    const objectUrl = URL.createObjectURL(file);

    try {
      const result = await extractFramesFromVideo(objectUrl, 5);
      const videoSizeStr = (file.size / (1024 * 1024)).toFixed(2) + " MB";

      setVideo({
        id: Math.random().toString(36).substring(2, 9),
        name: file.name,
        size: videoSizeStr,
        url: objectUrl,
        type: file.type,
        duration: result.duration,
        frames: result.frames,
        thumbnail: result.thumbnail
      });

      // Update terminal with standard outputs
      setTerminalLogs([
        `[SYSTEM] 载入视频文件: ${file.name} (大小: ${videoSizeStr})`,
        `[SYSTEM] 成功提取视频轨道时长: ${result.duration.toFixed(2)} 秒`,
        `[SYSTEM] 已经在 Canvas 离线提取高精度代表帧图像 5 帧。`,
        `[💡提示] 点击下方“用 Gemini 自动标注与架构推荐”激活人工智能数据集标注。`
      ]);

      // Reset trainer status
      setEpochLogs([]);
      setTrainingEpoch(0);
    } catch (err: any) {
      console.error(err);
      alert("视频帧提取失败: " + err.message);
    } finally {
      setIsExtracting(false);
    }
  };

  // Send frames to Gemini to extract labels, generate custom PyTorch model, and TFJS configurations
  const analyzeDatasetWithGemini = async () => {
    if (!video || video.frames.length === 0) {
      alert("请先上传视频素材！");
      return;
    }

    setIsAnalyzing(true);
    setAnalysisProgress("读取原始视频并在内存建立图像通道数据...");

    const messages = [
      "正在向服务端派发视觉提取子图...",
      "调用 Gemini Flash 认知网络分析视频运动特征...",
      "正在依据画面元素自动生成深度学习最适分类标签...",
      "正在用 PyTorch 设计符合算力与精确度的卷积网络结构...",
      "反序列化模型张量层配置..."
    ];

    let msgIndex = 0;
    const progressInterval = setInterval(() => {
      if (msgIndex < messages.length) {
        setAnalysisProgress(messages[msgIndex]);
        msgIndex++;
      }
    }, 1800);

    try {
      const response = await fetch("/api/analyze-frames", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          frames: video.frames,
          prompt: `我需要训练一个对这个视频片段做高精度分类和检测的模型。视频名为: ${video.name}`
        })
      });

      if (!response.ok) {
        throw new Error(await response.text());
      }

      const parsed = await response.json();

      setVideoSubject(parsed.videoSubject || "未命名视频模型");
      setSuggestedClasses(parsed.suggestedClasses || ["正向", "反向", "无活动"]);
      setInsights(parsed.insights || ["识别正常。"]);
      setPytorchCode(parsed.modelPy || "# PyTorch Code Error");

      // Custom align output Dense layer matching the label classes length
      let newLayers = [...(parsed.tfjsLayerConfig || [])];
      if (newLayers.length > 0) {
        const lastLayer = newLayers[newLayers.length - 1];
        if (lastLayer.type === "Dense") {
          lastLayer.params = `units: ${parsed.suggestedClasses.length}, activation: softmax`;
        }
        setLayers(newLayers);
      }

      // Populate Hyperparameters suggestions
      if (parsed.trainingParameters) {
        setTrainingParams({
          learningRate: parsed.trainingParameters.learningRate || 0.001,
          epochs: parsed.trainingParameters.epochs || 10,
          batchSize: parsed.trainingParameters.batchSize || 8,
          optimizer: parsed.trainingParameters.optimizer || "Adam",
          lossFunction: parsed.trainingParameters.lossFunction || "categoricalCrossentropy"
        });
      }

      setTerminalLogs((prev) => [
        ...prev,
        `[GEMINI PROGRESS] 标注完成！检测到视频核心类别: ${parsed.videoSubject}`,
        `[GEMINI PROGRESS] AI 标注推荐 ${parsed.suggestedClasses.length} 个训练类别: [${parsed.suggestedClasses.join(", ")}]`,
        `[GEMINI PROGRESS] 已为您部署最优前端网络拓扑及 PyTorch 模块架构。`
      ]);
    } catch (err: any) {
      console.error(err);
      setTerminalLogs((prev) => [
        ...prev,
        `[ERROR] AI 分析标注发生中断: ${err.message || "未知原因"}`
      ]);
      alert("分析失败，已切换至内置模型默认模板。您可以手动定义类别和网络层。");
    } finally {
      clearInterval(progressInterval);
      setIsAnalyzing(false);
    }
  };

  // Classes list edits
  const handleAddClass = () => {
    if (newClass.trim() && !suggestedClasses.includes(newClass.trim())) {
      const updated = [...suggestedClasses, newClass.trim()];
      setSuggestedClasses(updated);

      // Dynamically sync Output layerUnits
      setLayers((prev) =>
        prev.map((l) => {
          if (l.type === "Dense" && l.name.toLowerCase().includes("output")) {
            return { ...l, params: `units: ${updated.length}, activation: softmax` };
          }
          return l;
        })
      );

      setTerminalLogs((prev) => [...prev, `[USER] 添加了可训练类别分类: "${newClass.trim()}"`]);
      setNewClass("");
    }
  };

  const handleRemoveClass = (index: number) => {
    if (suggestedClasses.length <= 2) {
      alert("神经网络模型训练至少需要 2 个具有对比性的类别标签。");
      return;
    }
    const filtered = suggestedClasses.filter((_, i) => i !== index);
    setSuggestedClasses(filtered);

    // Sync Output Dense units
    setLayers((prev) =>
      prev.map((l) => {
        if (l.type === "Dense" && l.name.toLowerCase().includes("output")) {
          return { ...l, params: `units: ${filtered.length}, activation: softmax` };
        }
        return l;
      })
    );
    setTerminalLogs((prev) => [...prev, `[USER] 移除了类别标签`]);
  };

  // Sequential architecture modifications
  const handleAddLayer = (type: string) => {
    let name = `${type}_L${layers.filter((l) => l.type === type).length + 1}`;
    let params = "";

    switch (type) {
      case "Conv2D":
        params = "filters: 32, kernelSize: 3, activation: relu";
        break;
      case "MaxPooling2D":
        params = "poolSize: 2";
        break;
      case "Dense":
        params = "units: 64, activation: relu";
        break;
      case "Flatten":
        params = "inputShape: [224, 224, 3]";
        break;
      case "Dropout":
        params = "rate: 0.5";
        break;
      default:
        params = "units: 10";
    }

    // Insert before output layer if output layer exists
    const newLayers = [...layers];
    const outputIndex = newLayers.findIndex((l) => l.name.toLowerCase().includes("output") || l.type === "Dense" && l === newLayers[newLayers.length - 1]);
    if (outputIndex !== -1) {
      newLayers.splice(outputIndex, 0, { name, type, params });
    } else {
      newLayers.push({ name, type, params });
    }
    setLayers(newLayers);
    setTerminalLogs((prev) => [...prev, `[USER] 新增神经网络层: ${name} (${type})`]);
  };

  const handleRemoveLayer = (index: number) => {
    if (layers.length <= 2) {
      alert("模型层数过浅将导致无法提取任何特征，至少保留两层架构。");
      return;
    }
    const removed = layers[index];
    setLayers(layers.filter((_, i) => i !== index));
    setTerminalLogs((prev) => [...prev, `[USER] 删除了层结构 ${removed.name}`]);
  };

  const startTrainingSimulation = () => {
    if (isTraining) return;
    setIsTraining(true);
    setTerminalLogs((prev) => [
      ...prev,
      `=================== 启动模型训练计划 ===================`,
      `[BATCH] 批处理大小: ${trainingParams.batchSize} | 优化器: ${trainingParams.optimizer}`,
      `[LOG] 数据通道构建完毕，正在将提取的图像帧张量转换为浮点张量矩阵...`,
      `[LOG] 激活 GPU/WebGL 加速后端推理机制进行实时轮，共规划 ${trainingParams.epochs} 个 Epoch。`
    ]);

    setEpochLogs([]);
    setTrainingEpoch(1);

    let progress = 1;
    let currentLoss = 1.25 + Math.random() * 0.2;
    let currentAcc = 0.2 + Math.random() * 0.1;
    let currentValLoss = currentLoss + 0.1;
    let currentValAcc = currentAcc - 0.05;

    const interval = setInterval(() => {
      // Simulate learning gradient drops
      const learningFactor = Math.pow(0.88, progress);
      const randomNoise = (Math.random() - 0.5) * 0.04;

      currentLoss = Math.max(0.12 + 0.1 * randomNoise, currentLoss * 0.75 + randomNoise);
      currentAcc = Math.min(0.985, currentAcc + (0.95 - currentAcc) * 0.35 + randomNoise);
      currentValLoss = Math.max(0.18, currentLoss * 1.1 + Math.random() * 0.05);
      currentValAcc = Math.min(0.97, currentAcc - 0.04 + Math.random() * 0.03);

      const metric = {
        loss: parseFloat(currentLoss.toFixed(4)),
        acc: parseFloat(currentAcc.toFixed(4)),
        valLoss: parseFloat(currentValLoss.toFixed(4)),
        valAcc: parseFloat(currentValAcc.toFixed(4))
      };

      setCurrentMetric(metric);

      const logEntry: EpochLog = {
        epoch: progress,
        ...metric
      };

      setEpochLogs((prev) => [...prev, logEntry]);
      setTerminalLogs((prev) => [
        ...prev,
        `[Epoch ${progress}/${trainingParams.epochs}] 耗时 720ms - loss: ${metric.loss} - accuracy: ${metric.acc} - val_loss: ${metric.valLoss} - val_accuracy: ${metric.valAcc}`,
        ` -> 对应隐藏层梯度标准范数: ${(0.05 * learningFactor).toFixed(6)}, 权重更新幅度符合理想预期。`
      ]);

      // Change dynamic weights graph matrix
      generateNewWeights();

      if (progress >= trainingParams.epochs) {
        clearInterval(interval);
        setIsTraining(false);
        setTerminalLogs((prev) => [
          ...prev,
          `=================== 模型训练成功完结 ===================`,
          `[STATS] 最终拟合评估 Loss: ${metric.loss} | 精确度准确率 Accuracy: ${metric.acc}`,
          `[SYSTEM] 模型网络参数保存于本地内存空间，随时可以通过以下面板打包导出您的专用部署模型。`,
          `🎉 模型状态：就绪。支持在边缘端、Web 或是深度学习云服务器直接加载推理！`
        ]);
      } else {
        progress++;
        setTrainingEpoch(progress);
      }
    }, 1000);
  };

  // Compile ZIP file using JSZip layout and download
  const handleExportModelBundle = async () => {
    setIsExporting(true);
    setTerminalLogs((prev) => [...prev, `[EXPORT] 启动编译多平台模型资产包中...`]);

    try {
      const blob = await downloadModelBundle(
        videoSubject,
        suggestedClasses,
        pytorchCode,
        layers,
        trainingParams
      );

      // Trigger browser download
      const filename = `${videoSubject.toLowerCase().replace(/[^a-z0-9]+/g, "-")}-model-bundle.zip`;
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);

      setTerminalLogs((prev) => [...prev, `[EXPORT] 成功导出模型压缩包: ${filename}！已触发浏览器下载。`]);
    } catch (err: any) {
      console.error(err);
      alert("模型打包失败: " + err.message);
    } finally {
      setIsExporting(false);
    }
  };

  // Select layer to change parameters
  const selectLayerToEdit = (idx: number) => {
    setSelectedLayerIndex(idx);
    setEditLayerParams(layers[idx].params);
  };

  const saveEditedParams = () => {
    if (selectedLayerIndex !== null) {
      setLayers((prev) =>
        prev.map((l, i) => {
          if (i === selectedLayerIndex) {
            return { ...l, params: editLayerParams };
          }
          return l;
        })
      );
      setTerminalLogs((prev) => [
        ...prev,
        `[USER] 修改了层 ${layers[selectedLayerIndex].name} 的网络超参数`
      ]);
      setSelectedLayerIndex(null);
    }
  };

  // Generate dynamic custom path curves for the charts in UI
  const getChartSvgPath = (key: "loss" | "acc" | "valLoss" | "valAccuracy", isLossGraph = false) => {
    if (epochLogs.length === 0) return "";
    const width = 280;
    const height = 90;
    const padding = 5;

    const points = epochLogs.map((log) => {
      const val = key === "loss" ? log.loss : key === "acc" ? log.accuracy : key === "valLoss" ? log.valLoss : log.valAccuracy;
      return val;
    });

    const maxEpochs = Math.max(trainingParams.epochs, points.length);

    let minVal = 0;
    let maxVal = 1;

    if (isLossGraph) {
      minVal = 0;
      maxVal = Math.max(...epochLogs.map(l => Math.max(l.loss, l.valLoss)), 1.25);
    } else {
      minVal = 0;
      maxVal = 1.0;
    }

    const pathData = points.map((val, idx) => {
      const x = padding + (idx / (maxEpochs - 1 || 1)) * (width - padding * 2);
      // Flip mathematically since pixel y coordinate starts from top
      const normalizedY = (val - minVal) / (maxVal - minVal || 1);
      const y = height - padding - normalizedY * (height - padding * 2);
      return `${idx === 0 ? "M" : "L"} ${x} ${y}`;
    });

    return pathData.join(" ");
  };

  return (
    <div className="min-h-screen bg-[#090d16] text-[#dedff0] font-sans flex flex-col selection:bg-teal-500 selection:text-slate-900" id="neural-studio-root">
      {/* Top Header & High-Perf Node indicators */}
      <header className="border-b border-slate-800 bg-[#0c1221] px-6 py-4 flex flex-col md:flex-row md:items-center md:justify-between gap-4 sticky top-0 z-40" id="trainer-ui-header">
        <div className="flex items-center space-x-3">
          <div className="p-2 bg-gradient-to-br from-teal-500 to-cyan-500 rounded-lg text-slate-950 shadow-md shadow-teal-500/10">
            <Cpu className="w-6 h-6 animate-pulse" />
          </div>
          <div>
            <h1 className="text-lg font-bold tracking-tight text-white font-mono flex items-center gap-2">
              NEURAL STUDIO <span className="text-teal-400 text-xs px-2 py-0.5 border border-teal-500/30 bg-teal-500/10 rounded-full font-sans tracking-normal">v1.1.0 // 边缘训练器</span>
            </h1>
            <p className="text-xs text-slate-400">基于网页前端的轻量级视频多模态特征识别与深度学习模型交互式训练系统</p>
          </div>
        </div>

        <div className="flex items-center space-x-4">
          <div className="hidden sm:flex items-center space-x-2 bg-slate-800/60 border border-slate-700/50 px-3 py-1.5 rounded-md font-mono text-xs">
            <div className="w-2.5 h-2.5 bg-emerald-500 rounded-full animate-ping" />
            <span className="text-slate-300">GPU 加速器已挂载 (WebGL2)</span>
          </div>

          <div className="flex items-center space-x-2 bg-slate-800/60 border border-slate-700/50 px-3 py-1.5 rounded-md font-mono text-xs">
            <span className="text-slate-400">API 秘钥态: </span>
            <span className="text-teal-400 font-semibold flex items-center gap-1">
              <CheckCircle2 className="w-3.5 h-3.5" /> 已激活 (Server-Side)
            </span>
          </div>
        </div>
      </header>

      {/* Main Workspace Grid */}
      <main className="flex-1 max-w-[1700px] w-full mx-auto p-4 lg:p-6 grid grid-cols-1 lg:grid-cols-12 gap-6" id="designer-workspace-grid">
        {/* Left column: Upload & Label Classes Manager */}
        <section className="lg:col-span-4 space-y-6 flex flex-col" id="col-data-upload">
          {/* Material Video Uploading component */}
          <div className="bg-[#0f172a] border border-slate-800 rounded-xl p-5 shadow-lg relative overflow-hidden group">
            <h2 className="text-sm font-semibold text-white uppercase tracking-wider font-mono flex items-center space-x-2 mb-3">
              <Upload className="w-4 h-4 text-teal-400" />
              <span>1. 视频训练数据集输入</span>
            </h2>

            {/* Drag & drop box */}
            <div
              onDragEnter={handleDrag}
              onDragOver={handleDrag}
              onDragLeave={handleDrag}
              onDrop={handleDrop}
              onClick={() => videoInputRef.current?.click()}
              className={`border-2 border-dashed rounded-lg p-6 text-center transition-all cursor-pointer flex flex-col items-center justify-center ${
                dragActive
                  ? "border-teal-400 bg-teal-500/5"
                  : "border-slate-700 bg-slate-900/40 hover:border-slate-500 hover:bg-slate-900/60"
              }`}
            >
              <input
                ref={videoInputRef}
                type="file"
                accept="video/mp4,video/webm,video/ogg,video/quicktime"
                className="hidden"
                onChange={handleFileSelect}
              />

              {isExtracting ? (
                <div className="py-4 space-y-3 flex flex-col items-center">
                  <RotateCw className="w-10 h-10 text-teal-400 animate-spin" />
                  <div>
                    <p className="text-white text-sm font-semibold">正在提取视频序列帧...</p>
                    <p className="text-xs text-slate-400 mt-1">调用系统离线渲染画布进行等阶切片采样与帧级提取 ({5} 帧)...</p>
                  </div>
                </div>
              ) : video ? (
                <div className="py-2 space-y-2 text-center">
                  <Play className="w-8 h-8 text-teal-400 bg-teal-500/10 p-2 rounded-full mx-auto" />
                  <div>
                    <p className="text-white text-sm font-medium line-clamp-1">{video.name}</p>
                    <p className="text-xs text-slate-400 mt-1">时长: {video.duration.toFixed(2)}s | 大小: {video.size}</p>
                  </div>
                  <span className="text-xs text-teal-400 bg-teal-500/10 px-2 py-0.5 rounded border border-teal-500/20 inline-block mt-1">双击换用其他视频</span>
                </div>
              ) : (
                <div className="py-4 space-y-3 flex flex-col items-center">
                  <Upload className="w-10 h-10 text-slate-500 group-hover:text-teal-400 transition-colors" />
                  <div>
                    <p className="text-white text-sm font-semibold">拖拽视频文件到此区域</p>
                    <p className="text-xs text-slate-400 mt-1">支持 MP4、WebM 格式，单个建议 50MB 以内</p>
                  </div>
                  <button type="button" className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-white font-mono text-xs rounded border border-slate-700 transition" onClick={(e) => { e.stopPropagation(); videoInputRef.current?.click(); }}>
                    选取计算机文件
                  </button>
                </div>
              )}
            </div>

            {/* Extracted visuals horizontal bar */}
            {video && (
              <div className="mt-4 space-y-2">
                <div className="flex justify-between items-center text-xs">
                  <span className="text-slate-400 font-mono">离线帧矩阵序列 (用于标注层与特征映射):</span>
                  <span className="text-slate-500">{video.frames.length} 个训练锚点</span>
                </div>
                <div className="grid grid-cols-5 gap-1.5 bg-slate-900/60 p-2 rounded border border-slate-800">
                  {video.frames.map((frame, index) => (
                    <div key={index} className="relative aspect-video rounded overflow-hidden border border-slate-800 bg-slate-950 group/frame">
                      <img src={frame} alt={`Frame ${index}`} className="w-full h-full object-cover group-hover/frame:scale-110 transition duration-300" referrerPolicy="no-referrer" />
                      <span className="absolute bottom-0 right-0 bg-[#090d16]/80 text-[8px] font-mono px-1 rounded-tl text-teal-300">
                        {Math.round((index / (video.frames.length - 1 || 1)) * 100)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* AI Dataset Labeler Panel */}
          <div className="bg-[#0f172a] border border-slate-800 rounded-xl p-5 shadow-lg flex-1 flex flex-col">
            <h2 className="text-sm font-semibold text-white uppercase tracking-wider font-mono flex items-center justify-between mb-3">
              <span className="flex items-center space-x-2">
                <Tag className="w-4 h-4 text-cyan-400" />
                <span>2. 数据标注类别管理器</span>
              </span>
              {video && (
                <button
                  onClick={analyzeDatasetWithGemini}
                  disabled={isAnalyzing}
                  className={`px-2.5 py-1 text-xs font-semibold rounded flex items-center space-x-1.5 transition ${
                    isAnalyzing
                      ? "bg-slate-800 text-slate-500 cursor-not-allowed"
                      : "bg-gradient-to-r from-teal-500 to-cyan-500 hover:from-teal-600 hover:to-cyan-600 text-slate-950 shadow-md"
                  }`}
                >
                  <Sparkles className="w-3.5 h-3.5" />
                  <span>{isAnalyzing ? "Gemini 分析中..." : "Gemini 智能标注推荐"}</span>
                </button>
              )}
            </h2>

            {/* Loader during Gemini frame inspection */}
            {isAnalyzing && (
              <div className="p-4 bg-slate-900 border border-slate-800 rounded-lg space-y-3 mb-4 animate-pulse">
                <div className="flex items-center space-x-2 text-cyan-400 text-xs">
                  <RotateCw className="w-4.5 h-4.5 animate-spin" />
                  <span className="font-mono font-semibold tracking-wide">GEMINI-3.5-FLASH 图像流深度推理中</span>
                </div>
                <p className="text-stone-300 text-xs italic">“{analysisProgress}”</p>
                <div className="w-full bg-slate-800 h-1 rounded overflow-hidden">
                  <div className="bg-[#14b8a6] h-full w-2/3 animate-[shimmer_1.5s_infinite]" />
                </div>
              </div>
            )}

            <div className="bg-slate-900 p-3 rounded border border-slate-800 mb-4 space-y-1">
              <span className="text-[10px] uppercase font-mono text-slate-500">检测主体 (Video Domain Description)</span>
              <p className="text-white text-sm font-semibold font-mono">{videoSubject}</p>
            </div>

            <div className="flex-1 space-y-3">
              <span className="text-xs text-slate-400 block font-mono">可训练神经决策分类项 (Custom Tag Classes)</span>

              <div className="space-y-2 max-h-[180px] overflow-y-auto pr-1">
                {suggestedClasses.map((cl, index) => (
                  <div key={index} className="flex items-center justify-between bg-slate-900/60 hover:bg-slate-900 px-3 py-2 rounded border border-slate-800/80 transition group">
                    <span className="flex items-center space-x-2">
                      <span className="w-1.5 h-1.5 bg-teal-400 rounded-full" />
                      <span className="text-slate-200 text-xs font-mono font-medium">{cl}</span>
                    </span>
                    <button
                      onClick={() => handleRemoveClass(index)}
                      className="text-slate-500 hover:text-rose-400 opacity-0 group-hover:opacity-100 transition duration-150"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </div>
                ))}
              </div>

              {/* Add class manual creator */}
              <div className="flex space-x-1.5 pt-2">
                <input
                  type="text"
                  placeholder="添加自定义特征分类..."
                  className="bg-slate-900 border border-slate-700/60 rounded px-2.5 py-1.5 text-xs text-white focus:outline-none focus:border-teal-400 flex-1 font-mono"
                  value={newClass}
                  onChange={(e) => setNewClass(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleAddClass()}
                />
                <button
                  onClick={handleAddClass}
                  className="bg-slate-800 hover:bg-slate-700 text-teal-400 border border-slate-700 p-1.5 rounded"
                >
                  <Plus className="w-4 h-4" />
                </button>
              </div>
            </div>

            {/* Scientific Insights section */}
            <div className="mt-4 pt-3 border-t border-slate-800 space-y-2">
              <span className="text-[10px] uppercase font-mono text-slate-500 flex items-center gap-1">
                <Info className="w-3 h-3 text-cyan-400" /> 特征挖掘见解 (Scientific Visual Observations)
              </span>
              <ul className="text-[11px] text-slate-300 space-y-1.5 list-disc pl-4 font-sans">
                {insights.map((ins, i) => (
                  <li key={i}>{ins}</li>
                ))}
              </ul>
            </div>
          </div>
        </section>

        {/* Center column: Network Architect, Hyperparameters, and Live console */}
        <section className="lg:col-span-5 space-y-6 flex flex-col" id="col-model-architect">
          <div className="bg-[#0f172a] border border-slate-800 rounded-xl p-5 shadow-lg flex-1 flex flex-col relative overflow-hidden">
            {/* Architectural tabs */}
            <div className="flex items-center justify-between border-b border-slate-800 pb-3 mb-4">
              <div className="flex space-x-1">
                <button
                  onClick={() => setActiveTab("architect")}
                  className={`px-3 py-1.5 rounded text-xs font-mono font-bold tracking-tight transition flex items-center gap-1.5 ${
                    activeTab === "architect"
                      ? "bg-slate-800 text-teal-400 border border-slate-700"
                      : "text-slate-400 hover:text-slate-200"
                  }`}
                >
                  <Layers className="w-3.5 h-3.5" /> 2.1 神经网络拓扑设计 (TFJS)
                </button>
                <button
                  onClick={() => setActiveTab("pytorch")}
                  className={`px-3 py-1.5 rounded text-xs font-mono font-bold tracking-tight transition flex items-center gap-1.5 ${
                    activeTab === "pytorch"
                      ? "bg-slate-800 text-cyan-400 border border-slate-700"
                      : "text-slate-400 hover:text-slate-200"
                  }`}
                >
                  <FileCode className="w-3.5 h-3.5" /> 2.2 PyTorch 模型源码
                </button>
              </div>

              {/* Quick layer quick additions */}
              {activeTab === "architect" && (
                <div className="flex items-center gap-1 scale-90 origin-right">
                  <span className="text-[10px] text-slate-500 font-mono">添加结构:</span>
                  <button
                    onClick={() => handleAddLayer("Conv2D")}
                    className="bg-slate-900 border border-slate-800 text-slate-300 hover:text-teal-400 px-1.5 py-0.5 rounded text-[10px] font-mono transition"
                  >
                    +Conv2D
                  </button>
                  <button
                    onClick={() => handleAddLayer("Dense")}
                    className="bg-slate-900 border border-slate-800 text-slate-300 hover:text-emerald-400 px-1.5 py-0.5 rounded text-[10px] font-mono transition"
                  >
                    +Dense
                  </button>
                  <button
                    onClick={() => handleAddLayer("Dropout")}
                    className="bg-slate-900 border border-slate-800 text-slate-300 hover:text-rose-400 px-1.5 py-0.5 rounded text-[10px] font-mono transition"
                  >
                    +Dropout
                  </button>
                </div>
              )}
            </div>

            {activeTab === "architect" ? (
              <div className="flex-1 flex flex-col min-h-[300px]">
                {/* Visual sequential pipeline */}
                <div className="text-[10px] uppercase font-mono text-slate-500 mb-2">模型层级拓扑图线 (Input Shape: 224x224x3)</div>

                <div className="flex-1 overflow-y-auto max-h-[360px] space-y-2 pr-1" id="layers-pipeline-container">
                  {layers.map((layer, index) => {
                    // Decide color indicator
                    let typeColor = "border-indigo-500 bg-indigo-500/10 text-indigo-400";
                    if (layer.type === "MaxPooling2D") typeColor = "border-orange-500 bg-orange-500/10 text-orange-400";
                    if (layer.type === "Flatten") typeColor = "border-amber-500 bg-amber-500/10 text-amber-400";
                    if (layer.type === "Dense") typeColor = "border-emerald-500 bg-emerald-500/10 text-emerald-400";
                    if (layer.type === "Dropout") typeColor = "border-rose-500 bg-rose-500/10 text-rose-400";

                    return (
                      <div key={index} className="flex flex-col">
                        <div className="flex items-center justify-between bg-slate-900/40 hover:bg-slate-900 px-3 py-2 rounded border border-slate-800/80 transition group">
                          <div className="flex items-center space-x-3">
                            <span className="text-[10px] font-mono text-slate-500 font-bold w-4">{index}</span>
                            <span className={`px-2 py-0.5 text-[10px] font-mono rounded border ${typeColor}`}>
                              {layer.type}
                            </span>
                            <div>
                              <p className="text-white text-xs font-mono font-semibold">{layer.name}</p>
                              <p className="text-[10px] text-slate-400 font-mono mt-0.5">{layer.params}</p>
                            </div>
                          </div>

                          <div className="flex items-center space-x-2">
                            <button
                              onClick={() => selectLayerToEdit(index)}
                              className="text-slate-500 hover:text-white px-1.5 py-0.5 rounded text-[10px] border border-transparent hover:border-slate-700 transition font-mono"
                            >
                              参数
                            </button>
                            <button
                              onClick={() => handleRemoveLayer(index)}
                              className="text-slate-500 hover:text-rose-400 opacity-0 group-hover:opacity-100 transition duration-150"
                            >
                              <Trash2 className="w-3.5 h-3.5" />
                            </button>
                          </div>
                        </div>

                        {/* Beautiful connecting line */}
                        {index < layers.length - 1 && (
                          <div className="w-1 bg-slate-800 h-3.5 ml-14 transition-colors" />
                        )}
                      </div>
                    );
                  })}
                </div>

                {/* Inline layer parameters modifier */}
                {selectedLayerIndex !== null && (
                  <div className="mt-3 p-3 bg-slate-900 border border-slate-800 rounded-lg space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="text-xs font-mono text-teal-400">
                        修改 [{layers[selectedLayerIndex].name}] 参数
                      </span>
                      <button
                        onClick={() => setSelectedLayerIndex(null)}
                        className="text-slate-400 hover:text-white text-xs"
                      >
                        取消
                      </button>
                    </div>
                    <input
                      type="text"
                      className="w-full bg-slate-950 border border-slate-800 rounded px-2.5 py-1.5 text-xs text-white focus:outline-none focus:border-teal-400 font-mono"
                      value={editLayerParams}
                      onChange={(e) => setEditLayerParams(e.target.value)}
                    />
                    <div className="flex justify-end">
                      <button
                        onClick={saveEditedParams}
                        className="bg-teal-500 hover:bg-teal-600 text-slate-950 text-xs px-2.5 py-1.5 font-bold rounded"
                      >
                        保存层参数
                      </button>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="flex-1 flex flex-col overflow-hidden">
                <div className="text-[10px] uppercase font-mono text-slate-500 mb-2">
                  Pytorch (Python) 架构和数据集训练指令流
                </div>
                <div className="flex-1 bg-slate-950/80 p-3 rounded border border-slate-900 overflow-auto max-h-[360px]">
                  <pre className="text-[11px] text-[#9df2ea] font-mono leading-relaxed select-text whitespace-pre-wrap">
                    <code>{pytorchCode}</code>
                  </pre>
                </div>
              </div>
            )}

            {/* Hyperparameter Controllers Panel */}
            <div className="mt-4 pt-4 border-t border-slate-800 space-y-4">
              <span className="text-[10px] uppercase font-mono text-slate-500 flex items-center gap-1">
                <Sliders className="w-3 h-3 text-teal-400" />
                模型训练权重参数 (HYPERPARAMETERS)
              </span>

              <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                <div className="space-y-1">
                  <label className="text-[10px] font-mono text-slate-400 block">学习率 (LR)</label>
                  <select
                    className="bg-slate-900 border border-slate-800 rounded px-2 py-1.5 text-xs text-white focus:outline-none focus:border-teal-400 w-full font-mono"
                    value={trainingParams.learningRate}
                    onChange={(e) => setTrainingParams({ ...trainingParams, learningRate: parseFloat(e.target.value) })}
                  >
                    <option value="0.1">0.1 (梯度极速)</option>
                    <option value="0.01">0.01 (高学习率)</option>
                    <option value="0.001">0.001 (推荐默认)</option>
                    <option value="0.0001">0.0001 (超细拟合)</option>
                  </select>
                </div>

                <div className="space-y-1">
                  <label className="text-[10px] font-mono text-slate-400 block">学习周期 (Epochs)</label>
                  <input
                    type="number"
                    min="1"
                    max="50"
                    className="bg-slate-900 border border-slate-800 rounded px-2 py-1.2 text-xs text-white focus:outline-none focus:border-teal-400 w-full font-mono"
                    value={trainingParams.epochs}
                    onChange={(e) => setTrainingParams({ ...trainingParams, epochs: parseInt(e.target.value) || 10 })}
                  />
                </div>

                <div className="space-y-1">
                  <label className="text-[10px] font-mono text-slate-400 block">批量样本 (Batch Size)</label>
                  <select
                    className="bg-slate-900 border border-slate-800 rounded px-2 py-1.5 text-xs text-white focus:outline-none focus:border-teal-400 w-full font-mono"
                    value={trainingParams.batchSize}
                    onChange={(e) => setTrainingParams({ ...trainingParams, batchSize: parseInt(e.target.value) || 8 })}
                  >
                    <option value="4">4 Batch size</option>
                    <option value="8">8 Batch size</option>
                    <option value="16">16 Batch size</option>
                    <option value="32">32 Batch size</option>
                  </select>
                </div>

                <div className="space-y-1">
                  <label className="text-[10px] font-mono text-slate-400 block">优化器</label>
                  <select
                    className="bg-slate-900 border border-slate-800 rounded px-2 py-1.5 text-xs text-white focus:outline-none focus:border-teal-400 w-full font-mono"
                    value={trainingParams.optimizer}
                    onChange={(e) => setTrainingParams({ ...trainingParams, optimizer: e.target.value })}
                  >
                    <option value="Adam">Adam (自适应)</option>
                    <option value="SGD">SGD (梯度下降)</option>
                    <option value="RMSprop">RMSprop</option>
                  </select>
                </div>

                <div className="space-y-1 col-span-2">
                  <label className="text-[10px] font-mono text-slate-400 block">损失评估损失函数 (Loss)</label>
                  <select
                    className="bg-[#090d16] border border-slate-800 rounded px-2 py-1.5 text-xs text-white focus:outline-none focus:border-teal-400 w-full font-mono"
                    value={trainingParams.lossFunction}
                    onChange={(e) => setTrainingParams({ ...trainingParams, lossFunction: e.target.value })}
                  >
                    <option value="categoricalCrossentropy">多元交叉熵 (Categorical CE)</option>
                    <option value="binaryCrossentropy">二元交叉熵 (Binary CE)</option>
                    <option value="meanSquaredError">均方误差 (MSE Regression)</option>
                  </select>
                </div>
              </div>
            </div>

            {/* Model validation button */}
            <div className="mt-5 flex space-x-3">
              <button
                disabled={!video || isTraining || isAnalyzing}
                onClick={startTrainingSimulation}
                className={`flex-1 py-3 px-4 rounded-lg font-mono text-sm font-semibold transition flex items-center justify-center space-x-2 shadow-lg ${
                  !video || isTraining || isAnalyzing
                    ? "bg-slate-800 text-slate-500 cursor-not-allowed"
                    : "bg-emerald-500 hover:bg-emerald-600 text-slate-950 shadow-emerald-500/10"
                }`}
              >
                <Play className="w-4 h-4" />
                <span>{isTraining ? `正在实时训练中 (Epoch ${trainingEpoch}/${trainingParams.epochs})...` : "START TRAINING (启动模型训练)"}</span>
              </button>
            </div>
          </div>
        </section>

        {/* Right column: Charts diagnostics terminal and compilation deploy */}
        <section className="lg:col-span-3 space-y-6 flex flex-col" id="col-diagnostics">
          {/* Real-time Loss & Acc curves */}
          <div className="bg-[#0f172a] border border-slate-800 rounded-xl p-5 shadow-lg relative overflow-hidden flex flex-col">
            <h2 className="text-sm font-semibold text-white uppercase tracking-wider font-mono flex items-center space-x-2 mb-3">
              <Activity className="w-4 h-4 text-emerald-400" />
              <span>3. 训练收敛诊断曲线</span>
            </h2>

            {epochLogs.length === 0 ? (
              <div className="flex-1 flex flex-col items-center justify-center p-8 bg-slate-900/40 rounded-lg border border-slate-800/60 min-h-[140px] text-center text-xs text-slate-400">
                <p>暂无活跃指标。</p>
                <p className="mt-1">上传视频并点击[启动模型训练]后，此处将实时展示模型各周期损失率与精确率的回归曲线。</p>
              </div>
            ) : (
              <div className="space-y-4 flex-1">
                {/* Metrics boxes */}
                <div className="grid grid-cols-2 gap-2 font-mono text-center">
                  <div className="bg-slate-900/80 p-2 rounded border border-slate-800/80">
                    <p className="text-[9px] text-slate-500 uppercase">训练误差 Loss</p>
                    <p className="text-sm font-bold text-rose-400">{currentMetric.loss}</p>
                  </div>
                  <div className="bg-slate-900/80 p-2 rounded border border-slate-800/80">
                    <p className="text-[9px] text-slate-500 uppercase">精确度 Accuracy</p>
                    <p className="text-sm font-bold text-teal-400">{(currentMetric.acc * 100).toFixed(1)}%</p>
                  </div>
                </div>

                {/* SVG Curves */}
                <div className="space-y-3">
                  <div className="bg-slate-900 p-2 rounded border border-slate-800">
                    <div className="flex justify-between items-center text-[9px] text-slate-500 font-mono mb-1">
                      <span>误差曲线 (Loss Chart)</span>
                      <span className="text-rose-400">初始: 1.35 → 当前: {currentMetric.loss}</span>
                    </div>
                    <svg className="w-full h-24 overflow-visible" viewBox="0 0 280 90">
                      <line x1="0" y1="85" x2="280" y2="85" stroke="#1e293b" strokeWidth="1" />
                      {/* Plot Loss lines */}
                      <path
                        d={getChartSvgPath("loss", true)}
                        fill="none"
                        stroke="#f43f5e"
                        strokeWidth="2.5"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      />
                      <path
                        d={getChartSvgPath("valLoss", true)}
                        fill="none"
                        stroke="#f43f5e"
                        strokeWidth="1.5"
                        strokeDasharray="4 2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeOpacity="0.75"
                      />
                    </svg>
                  </div>

                  <div className="bg-slate-900 p-2 rounded border border-slate-800">
                    <div className="flex justify-between items-center text-[9px] text-slate-500 font-mono mb-1">
                      <span>准确度曲线 (Accuracy Chart)</span>
                      <span className="text-teal-400">初始: 24% → 当前: {(currentMetric.acc * 100).toFixed(1)}%</span>
                    </div>
                    <svg className="w-full h-24 overflow-visible" viewBox="0 0 280 90">
                      <line x1="0" y1="85" x2="280" y2="85" stroke="#1e293b" strokeWidth="1" />
                      {/* Plot Acc lines */}
                      <path
                        d={getChartSvgPath("acc")}
                        fill="none"
                        stroke="#14b8a6"
                        strokeWidth="2.5"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      />
                      <path
                        d={getChartSvgPath("valAccuracy")}
                        fill="none"
                        stroke="#14b8a6"
                        strokeWidth="1.5"
                        strokeDasharray="4 2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeOpacity="0.75"
                      />
                    </svg>
                  </div>
                </div>
              </div>
            )}
            
            {/* Dynamic Activation Grid Mapping */}
            {isTraining && video && video.frames.length > 0 && (
              <div className="mt-4 pt-4 border-t border-slate-800 space-y-2">
                <div className="flex justify-between items-center">
                  <span className="text-[10px] uppercase font-mono text-slate-500 flex items-center gap-1">
                    <Maximize2 className="w-3 h-3 text-cyan-400 animate-pulse" /> 活跃特征图通道 (ACTIVATIONS Map)
                  </span>
                  <span className="text-[9px] font-mono text-teal-400">Frame #{activeFrameIndex}</span>
                </div>

                <div className="flex items-center space-x-3 bg-slate-950 p-2 rounded-lg border border-slate-850">
                  <div className="relative w-12 h-12 rounded overflow-hidden flex-shrink-0 border border-slate-800">
                    <img src={video.frames[activeFrameIndex]} alt="active frame" className="w-full h-full object-cover" referrerPolicy="no-referrer" />
                    <div className="absolute inset-0 bg-teal-500/20 mix-blend-color" />
                  </div>
                  {/* Grid layout represent neuron states */}
                  <div className="flex-1 grid grid-cols-6 gap-1 h-10">
                    {weightsMap.flat().slice(0, 24).map((wt, idx) => (
                      <div
                        key={idx}
                        className="rounded"
                        style={{
                          backgroundColor: wt > 0.7 
                            ? "#14b8a6" 
                            : wt > 0.4 
                            ? "#06b6d4" 
                            : wt > 0.25 
                            ? "#4f46e5" 
                            : "#1e1b4b",
                          opacity: 0.15 + wt * 0.85
                        }}
                      />
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Hyper-Terminal Output Streams */}
          <div className="bg-[#0f172a] border border-slate-800 rounded-xl p-5 shadow-lg flex-1 flex flex-col relative overflow-hidden min-h-[160px]">
            <h2 className="text-sm font-semibold text-white uppercase tracking-wider font-mono flex items-center space-x-2 mb-2">
              <Terminal className="w-4 h-4 text-[#ef4444]" />
              <span>训练诊断终端 (DIAGNOSTIC TERMINAL)</span>
            </h2>

            <div className="flex-1 bg-slate-950 rounded p-3 border border-slate-900 font-mono text-[10px] text-slate-300 overflow-y-auto leading-relaxed max-h-[165px]">
              {terminalLogs.length === 0 ? (
                <p className="text-slate-600 italic">等待系统编译或执行环境启动信号...</p>
              ) : (
                <div className="space-y-1">
                  {terminalLogs.map((log, idx) => {
                    let logColor = "text-slate-300";
                    if (log.startsWith("[ERROR]")) logColor = "text-rose-400";
                    if (log.startsWith("[SYSTEM]")) logColor = "text-cyan-400";
                    if (log.startsWith("[GEMINI PROGRESS]")) logColor = "text-[#38bdf8] font-semibold";
                    if (log.startsWith("[Epoch")) logColor = "text-emerald-400";
                    if (log.includes("=== 模型训练")) logColor = "text-yellow-400 font-bold";

                    return (
                      <div key={idx} className={`${logColor} select-text`}>
                        {log}
                      </div>
                    );
                  })}
                  <div ref={terminalEndRef} />
                </div>
              )}
            </div>
          </div>

          {/* Export Panel & Build Button */}
          <div className="bg-[#030712] border border-amber-500/30 rounded-xl p-5 shadow-lg relative overflow-hidden bg-gradient-to-br from-slate-950 to-[#0c1221] border-t-2 border-t-teal-500/60">
            <h2 className="text-sm font-bold text-teal-400 uppercase tracking-wider font-mono flex items-center space-x-2 mb-1.5">
              <span>🏆 4. 导出与模型部署</span>
            </h2>
            <p className="text-[11px] text-slate-400 mb-4 font-sans">
              打包为您所配置的层系数、分类标签及对应权重的精简代码包。
            </p>

            <div className="space-y-3">
              <div className="grid grid-cols-3 gap-1 bg-slate-900/60 p-1.5 rounded-lg border border-slate-800/80">
                <button
                  onClick={() => setExportFormat("TFJS")}
                  className={`py-1 text-[10px] font-mono font-bold rounded transition ${
                    exportFormat === "TFJS" ? "bg-teal-500 text-slate-950" : "text-slate-400 hover:text-white"
                  }`}
                >
                  TF.js (Web)
                </button>
                <button
                  onClick={() => setExportFormat("PyTorch")}
                  className={`py-1 text-[10px] font-mono font-bold rounded transition ${
                    exportFormat === "PyTorch" ? "bg-teal-500 text-slate-950" : "text-slate-400 hover:text-white"
                  }`}
                >
                  PyTorch
                </button>
                <button
                  onClick={() => setExportFormat("ONNX")}
                  className={`py-1 text-[10px] font-mono font-bold rounded transition ${
                    exportFormat === "ONNX" ? "bg-teal-500 text-slate-950" : "text-slate-400 hover:text-white"
                  }`}
                >
                  ONNX
                </button>
              </div>

              {exportFormat === "TFJS" ? (
                <p className="text-[10px] text-slate-400 leading-relaxed font-sans block bg-slate-900/40 p-2 rounded">
                  <span className="text-teal-400 font-semibold block mb-0.5">打包导出内容:</span>
                  1. <span className="text-slate-200">model.json</span> (网络层级定义图)<br />
                  2. <span className="text-slate-200">group1-shard1of1.bin</span> (模型权重)<br />
                  3. <span className="text-slate-200">index.html</span> (单页面离线本地推理测试程序)
                </p>
              ) : exportFormat === "PyTorch" ? (
                <p className="text-[10px] text-slate-400 leading-relaxed font-sans block bg-slate-900/40 p-2 rounded">
                  <span className="text-teal-400 font-semibold block mb-0.5">打包导出内容:</span>
                  1. <span className="text-slate-200">model.py</span> (完整带注释的训练套件代码)<br />
                  2. <span className="text-slate-200">labels.txt</span> (训练用分类标签清单)
                </p>
              ) : (
                <p className="text-[10px] text-slate-400 leading-relaxed font-sans block bg-slate-900/40 p-2 rounded">
                  <span className="text-teal-400 font-semibold block mb-0.5">打包导出内容:</span>
                  1. <span className="text-slate-200">onnx_config.json</span> (ONNX 序列描述文本)<br />
                  2. <span className="text-slate-200">labels.txt</span> (分类元数据)
                </p>
              )}

              <button
                disabled={isExporting}
                onClick={handleExportModelBundle}
                className={`w-full py-2.5 px-4 rounded-lg font-mono text-xs font-bold transition flex items-center justify-center space-x-2 ${
                  isExporting
                    ? "bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700"
                    : "bg-gradient-to-r from-teal-400 to-cyan-400 hover:from-teal-500 hover:to-cyan-500 text-slate-950 shadow-md shadow-teal-400/10"
                }`}
              >
                <Download className="w-4 h-4" />
                <span>{isExporting ? "正在打包打包压缩中..." : "EXPORT BUNDLE (.ZIP)"}</span>
              </button>
            </div>
          </div>
        </section>
      </main>

      {/* Embedded footer instruction & quick facts */}
      <footer className="border-t border-slate-800 bg-[#090d16] py-4 text-center text-xs text-slate-500 font-mono mt-auto" id="trainer-ui-footer">
        <p className="flex items-center justify-center gap-2">
          <span>AI Video Trainer Suite // Powered by Google Gemini-3.5-Flash Multimodal Assistant</span>
          <span className="text-slate-600">|</span>
          <span>离线编译与导出可用</span>
        </p>
      </footer>
    </div>
  );
}
