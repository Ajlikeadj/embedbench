import React, { useState } from "react";
import { Sliders, Cpu, Zap, Trophy, ShieldCheck, Gauge } from "lucide-react";

interface ModelPreset {
  name: string;
  type: string;
  hit5: number;
  mrr5: number;
  ndcg5: number;
  embedTimeSec: number;
  p95LatencyMs: number;
  memoryMb: number;
  recommendation?: "Speed" | "Accuracy" | "Balanced";
}

const mockPresets: Record<string, ModelPreset[]> = {
  technicalDocs: [
    {
      name: "ollama/bge-m3",
      type: "Dense + Sparse Local",
      hit5: 0.94,
      mrr5: 0.88,
      ndcg5: 0.91,
      embedTimeSec: 8.4,
      p95LatencyMs: 24.1,
      memoryMb: 1140,
      recommendation: "Accuracy"
    },
    {
      name: "all-MiniLM-L6-v2",
      type: "SentenceTransformers HF",
      hit5: 0.86,
      mrr5: 0.79,
      ndcg5: 0.82,
      embedTimeSec: 2.1,
      p95LatencyMs: 6.8,
      memoryMb: 240,
      recommendation: "Speed"
    },
    {
      name: "ollama/nomic-embed-text",
      type: "Local Ollama 8k Context",
      hit5: 0.91,
      mrr5: 0.84,
      ndcg5: 0.87,
      embedTimeSec: 4.8,
      p95LatencyMs: 14.5,
      memoryMb: 580,
      recommendation: "Balanced"
    }
  ],
  supportTickets: [
    {
      name: "ollama/nomic-embed-text",
      type: "Local Ollama 8k Context",
      hit5: 0.93,
      mrr5: 0.86,
      ndcg5: 0.89,
      embedTimeSec: 5.2,
      p95LatencyMs: 15.2,
      memoryMb: 580,
      recommendation: "Balanced"
    },
    {
      name: "bge-small-en-v1.5",
      type: "SentenceTransformers HF",
      hit5: 0.89,
      mrr5: 0.81,
      ndcg5: 0.84,
      embedTimeSec: 2.6,
      p95LatencyMs: 7.9,
      memoryMb: 280,
      recommendation: "Speed"
    },
    {
      name: "ollama/bge-large",
      type: "Local Ollama",
      hit5: 0.95,
      mrr5: 0.89,
      ndcg5: 0.92,
      embedTimeSec: 12.1,
      p95LatencyMs: 38.4,
      memoryMb: 1420,
      recommendation: "Accuracy"
    }
  ]
};

export function InteractiveBenchmarkDemo() {
  const [corpusType, setCorpusType] = useState<"technicalDocs" | "supportTickets">("technicalDocs");
  const [chunkSize, setChunkSize] = useState<number>(500);
  const [topK, setTopK] = useState<number>(5);

  const models = mockPresets[corpusType];

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 sm:p-8 shadow-2xl relative overflow-hidden">
      <div className="absolute top-0 right-0 w-96 h-96 bg-indigo-500/10 rounded-full blur-3xl pointer-events-none -mr-20 -mt-20"></div>

      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 mb-8">
        <div>
          <div className="flex items-center gap-2 text-indigo-400 font-semibold text-sm mb-1 uppercase tracking-wider">
            <Gauge className="w-4 h-4" /> Live Benchmark Playground
          </div>
          <h3 className="text-2xl font-bold text-white">Compare Real RAG Retrieval Profiles</h3>
          <p className="text-slate-400 text-sm">Preview how embedbench analyzes accuracy vs latency vs memory footprint.</p>
        </div>

        <div className="flex items-center gap-2 bg-slate-950 p-1.5 rounded-xl border border-slate-800">
          <button
            onClick={() => setCorpusType("technicalDocs")}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
              corpusType === "technicalDocs"
                ? "bg-indigo-600 text-white shadow-sm"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            Technical Docs
          </button>
          <button
            onClick={() => setCorpusType("supportTickets")}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
              corpusType === "supportTickets"
                ? "bg-indigo-600 text-white shadow-sm"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            Support Tickets
          </button>
        </div>
      </div>

      {/* Control sliders */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-8 p-4 bg-slate-950/70 border border-slate-800/80 rounded-xl">
        <div>
          <div className="flex justify-between text-xs font-medium text-slate-300 mb-2">
            <span>Chunk Size: {chunkSize} chars</span>
            <span className="text-slate-500">Fixed / Recursive</span>
          </div>
          <input
            type="range"
            min="200"
            max="1200"
            step="100"
            value={chunkSize}
            onChange={(e) => setChunkSize(Number(e.target.value))}
            className="w-full accent-indigo-500 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
          />
        </div>

        <div>
          <div className="flex justify-between text-xs font-medium text-slate-300 mb-2">
            <span>Retrieval Top-K: {topK}</span>
            <span className="text-slate-500">FAISS Cosine Index</span>
          </div>
          <input
            type="range"
            min="3"
            max="10"
            step="1"
            value={topK}
            onChange={(e) => setTopK(Number(e.target.value))}
            className="w-full accent-indigo-500 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
          />
        </div>
      </div>

      {/* Models Table / Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {models.map((m) => {
          const badgeColor =
            m.recommendation === "Accuracy"
              ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/30"
              : m.recommendation === "Speed"
              ? "bg-amber-500/10 text-amber-400 border-amber-500/30"
              : "bg-indigo-500/10 text-indigo-400 border-indigo-500/30";

          return (
            <div
              key={m.name}
              className="bg-slate-950/80 border border-slate-800 rounded-xl p-5 hover:border-slate-700 transition-all flex flex-col justify-between"
            >
              <div>
                <div className="flex items-center justify-between mb-3">
                  <span className="text-xs font-mono text-slate-400 truncate max-w-[140px]">{m.name}</span>
                  {m.recommendation && (
                    <span className={`text-[11px] font-semibold px-2 py-0.5 rounded-full border ${badgeColor} flex items-center gap-1`}>
                      <Trophy className="w-3 h-3" />
                      {m.recommendation} Pick
                    </span>
                  )}
                </div>

                <div className="text-xs text-slate-400 mb-4">{m.type}</div>

                <div className="space-y-3">
                  <div>
                    <div className="flex justify-between text-xs mb-1">
                      <span className="text-slate-400">Hit@{topK} (Recall)</span>
                      <span className="font-mono text-slate-200 font-semibold">{(m.hit5 * 100).toFixed(1)}%</span>
                    </div>
                    <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                      <div
                        className="bg-indigo-500 h-full rounded-full transition-all duration-500"
                        style={{ width: `${m.hit5 * 100}%` }}
                      ></div>
                    </div>
                  </div>

                  <div>
                    <div className="flex justify-between text-xs mb-1">
                      <span className="text-slate-400">MRR@{topK}</span>
                      <span className="font-mono text-slate-200 font-semibold">{m.mrr5.toFixed(2)}</span>
                    </div>
                    <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                      <div
                        className="bg-emerald-500 h-full rounded-full transition-all duration-500"
                        style={{ width: `${m.mrr5 * 100}%` }}
                      ></div>
                    </div>
                  </div>

                  <div>
                    <div className="flex justify-between text-xs mb-1">
                      <span className="text-slate-400">NDCG@{topK}</span>
                      <span className="font-mono text-slate-200 font-semibold">{m.ndcg5.toFixed(2)}</span>
                    </div>
                    <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                      <div
                        className="bg-cyan-500 h-full rounded-full transition-all duration-500"
                        style={{ width: `${m.ndcg5 * 100}%` }}
                      ></div>
                    </div>
                  </div>
                </div>
              </div>

              <div className="mt-6 pt-4 border-t border-slate-800/80 grid grid-cols-2 gap-2 text-xs">
                <div>
                  <div className="text-slate-500 flex items-center gap-1">
                    <Zap className="w-3 h-3 text-amber-400" /> p95 Query
                  </div>
                  <div className="font-mono font-medium text-slate-200 mt-0.5">{m.p95LatencyMs} ms</div>
                </div>
                <div>
                  <div className="text-slate-500 flex items-center gap-1">
                    <Cpu className="w-3 h-3 text-purple-400" /> Memory
                  </div>
                  <div className="font-mono font-medium text-slate-200 mt-0.5">{m.memoryMb} MB</div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
