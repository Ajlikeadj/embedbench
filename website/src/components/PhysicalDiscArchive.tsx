"use client";

import React, { useState } from "react";
import { Disc3, Check, Sparkles, ArrowRight, CornerDownRight } from "lucide-react";

export interface ModelRecord {
  id: string;
  catalogNo: string;
  name: string;
  runtime: string;
  dim: number;
  maxTokens: number;
  hit5: number;
  mrr5: number;
  ndcg5: number;
  p95LatencyMs: number;
  embedTimeSec: number;
  ramUsageMb: number;
  license: string;
  verdict: "Pinnacle Recall" | "Ultralight Speed" | "Balanced Workhorse" | "High Capacity";
  verdictQuote: string;
  critic: string;
  vinylColor: string;
  accentColor: string;
}

export const ARCHIVE_MODELS: ModelRecord[] = [
  {
    id: "bge-m3",
    catalogNo: "EB-VOL.01",
    name: "BAAI / BGE-M3",
    runtime: "Ollama / Local PyTorch",
    dim: 1024,
    maxTokens: 8192,
    hit5: 0.942,
    mrr5: 0.884,
    ndcg5: 0.913,
    p95LatencyMs: 22.4,
    embedTimeSec: 7.8,
    ramUsageMb: 1140,
    license: "MIT",
    verdict: "Pinnacle Recall",
    verdictQuote: "Unrivaled multi-lingual & technical document comprehension. Dense retrieval precision that surfaces edge cases other models miss entirely.",
    critic: "Retrieval Journal, Vol. IV",
    vinylColor: "from-stone-900 via-zinc-800 to-black",
    accentColor: "#d97706" // amber
  },
  {
    id: "all-minilm",
    catalogNo: "EB-VOL.02",
    name: "all-MiniLM-L6-v2",
    runtime: "Hugging Face / ONNX",
    dim: 384,
    maxTokens: 512,
    hit5: 0.865,
    mrr5: 0.791,
    ndcg5: 0.824,
    p95LatencyMs: 5.6,
    embedTimeSec: 1.9,
    ramUsageMb: 240,
    license: "Apache 2.0",
    verdict: "Ultralight Speed",
    verdictQuote: "Featherweight champion. 4x faster indexing with near-instant CPU execution. Perfect for low-resource laptops and high-throughput pipelines.",
    critic: "Benchmark Digest",
    vinylColor: "from-neutral-900 via-stone-800 to-zinc-950",
    accentColor: "#10b981" // emerald
  },
  {
    id: "nomic-embed",
    catalogNo: "EB-VOL.03",
    name: "Nomic Embed Text v1.5",
    runtime: "Ollama / GGUF",
    dim: 768,
    maxTokens: 8192,
    hit5: 0.918,
    mrr5: 0.843,
    ndcg5: 0.879,
    p95LatencyMs: 13.8,
    embedTimeSec: 4.2,
    ramUsageMb: 580,
    license: "Apache 2.0",
    verdict: "Balanced Workhorse",
    verdictQuote: "The sweet spot of modern RAG. Extended 8k token context window, disciplined memory footprint, and consistent ranking authority.",
    critic: "Local AI Monograph",
    vinylColor: "from-zinc-900 via-neutral-800 to-black",
    accentColor: "#3b82f6" // blue
  },
  {
    id: "bge-large",
    catalogNo: "EB-VOL.04",
    name: "BAAI / BGE-Large-EN v1.5",
    runtime: "Ollama / PyTorch",
    dim: 1024,
    maxTokens: 512,
    hit5: 0.951,
    mrr5: 0.899,
    ndcg5: 0.928,
    p95LatencyMs: 34.2,
    embedTimeSec: 11.5,
    ramUsageMb: 1380,
    license: "MIT",
    verdict: "High Capacity",
    verdictQuote: "Heavyweight semantic discrimination. Demands capable hardware but returns near-human relevancy scores on dense academic texts.",
    critic: "ArXiv Empirical Review",
    vinylColor: "from-stone-900 via-zinc-900 to-black",
    accentColor: "#ec4899" // pink
  }
];

export function PhysicalDiscArchive() {
  const [selectedIndex, setSelectedIndex] = useState(0);
  const active = ARCHIVE_MODELS[selectedIndex];

  return (
    <div className="w-full">
      {/* Editorial Section Header */}
      <div className="flex flex-col md:flex-row md:items-end justify-between border-b border-archival pb-6 mb-10 gap-4">
        <div>
          <span className="text-xs uppercase tracking-[0.25em] text-[#a09e94] font-mono-code block mb-2">
            Archive Catalogue // Index 01—04
          </span>
          <h2 className="text-3xl md:text-5xl font-serif-editorial italic font-normal tracking-tight text-[#f2f1eb]">
            Curated Local Embedding Editions
          </h2>
        </div>
        <p className="text-xs md:text-sm text-[#8c8a80] max-w-md font-mono-code leading-relaxed">
          Select a title to inspect physical disc profiles, token geometries, FAISS cosine rankings, and verified laboratory metrics.
        </p>
      </div>

      {/* Horizontal Disc / Physical Media Carousel */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-6 mb-12">
        {ARCHIVE_MODELS.map((item, idx) => {
          const isSelected = selectedIndex === idx;
          return (
            <button
              key={item.id}
              onClick={() => setSelectedIndex(idx)}
              className={`group relative text-left p-4 sm:p-5 rounded-none border transition-all duration-300 flex flex-col justify-between h-[300px] sm:h-[340px] overflow-hidden ${
                isSelected
                  ? "bg-[#14151a] border-[#e5e4de] shadow-[0_0_35px_rgba(255,255,255,0.06)]"
                  : "bg-[#0f1013] border-archival hover:border-[#62615a] hover:bg-[#121317]"
              }`}
            >
              {/* Top metadata */}
              <div className="flex items-center justify-between text-[11px] font-mono-code text-[#73726b] w-full">
                <span>{item.catalogNo}</span>
                <span className="uppercase text-[10px] tracking-widest text-[#a8a69b]">
                  {item.dim}D // {item.maxTokens}T
                </span>
              </div>

              {/* Physical Vinyl / Disc visualizer */}
              <div className="relative my-auto flex items-center justify-center py-4">
                <div
                  className={`relative w-28 h-28 sm:w-36 sm:h-36 rounded-full bg-gradient-to-tr ${item.vinylColor} border border-[#383a42] shadow-2xl flex items-center justify-center transition-transform duration-700 ${
                    isSelected ? "spin-slow scale-105" : "group-hover:scale-95 opacity-80"
                  }`}
                  style={{
                    boxShadow: isSelected
                      ? `0 0 30px ${item.accentColor}25, inset 0 0 15px rgba(0,0,0,0.8)`
                      : "inset 0 0 15px rgba(0,0,0,0.8)",
                  }}
                >
                  {/* Concentric disc grooves */}
                  <div className="absolute inset-2 rounded-full border border-white/5 pointer-events-none" />
                  <div className="absolute inset-5 rounded-full border border-white/5 pointer-events-none" />
                  <div className="absolute inset-8 rounded-full border border-white/10 pointer-events-none" />

                  {/* Center Label */}
                  <div
                    className="w-10 h-10 sm:w-12 sm:h-12 rounded-full flex flex-col items-center justify-center text-center border shadow-inner relative z-10"
                    style={{
                      backgroundColor: isSelected ? item.accentColor : "#202228",
                      borderColor: "#4a4d58",
                    }}
                  >
                    <div className="w-2.5 h-2.5 rounded-full bg-[#0c0d10] border border-white/20" />
                  </div>
                </div>

                {/* Hand-drawn editorial circle badge on selected */}
                {isSelected && (
                  <div className="absolute -top-1 -right-1 text-[10px] font-mono-code px-2 py-0.5 border border-[#e5e4de] bg-[#0c0d10] text-[#f4f3ec] uppercase tracking-wider">
                    Selected
                  </div>
                )}
              </div>

              {/* Title & Category */}
              <div className="pt-2 border-t border-archival">
                <div className="text-xs uppercase tracking-wider text-[#a09e94] font-mono-code mb-1">
                  {item.verdict}
                </div>
                <div className="text-sm sm:text-base font-serif-editorial italic font-medium text-[#f5f4ee] truncate">
                  {item.name}
                </div>
              </div>
            </button>
          );
        })}
      </div>

      {/* Selected Title Editorial Pull-Out (A24 / Arthouse Style) */}
      <div className="border border-archival bg-[#111216] p-6 sm:p-10 lg:p-12 relative overflow-hidden">
        {/* Subtle background glow */}
        <div
          className="absolute top-0 right-0 w-96 h-96 opacity-10 blur-3xl pointer-events-none"
          style={{ backgroundColor: active.accentColor }}
        />

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 lg:gap-12 items-start">
          {/* Left Column: Review & Monograph */}
          <div className="lg:col-span-7 space-y-6">
            <div className="flex items-center gap-3 text-xs font-mono-code text-[#8e8d83]">
              <span className="px-2 py-0.5 border border-archival uppercase text-[10px] tracking-widest text-[#f5f4ee]">
                {active.catalogNo}
              </span>
              <span>•</span>
              <span className="text-[#c7c5ba]">{active.runtime}</span>
              <span>•</span>
              <span>License: {active.license}</span>
            </div>

            <h3 className="text-3xl sm:text-4xl md:text-5xl font-serif-editorial italic font-normal text-[#faf9f5] leading-tight">
              &ldquo;{active.verdictQuote}&rdquo;
            </h3>

            <div className="flex items-center gap-2 text-xs font-mono-code text-[#a09e94]">
              <span className="text-[#e28743]">★ ★ ★ ★ ★</span>
              <span>— {active.critic}</span>
            </div>

            <div className="pt-4 flex flex-wrap gap-4 text-xs font-mono-code text-[#8e8d83]">
              <div className="border border-archival p-3 bg-[#0c0d10]">
                <div className="text-[#6d6b63] text-[10px] uppercase tracking-wider">Embed Dimensions</div>
                <div className="text-[#f5f4ee] text-sm font-semibold mt-0.5">{active.dim} dimensions</div>
              </div>
              <div className="border border-archival p-3 bg-[#0c0d10]">
                <div className="text-[#6d6b63] text-[10px] uppercase tracking-wider">Max Context Window</div>
                <div className="text-[#f5f4ee] text-sm font-semibold mt-0.5">{active.maxTokens.toLocaleString()} tokens</div>
              </div>
              <div className="border border-archival p-3 bg-[#0c0d10]">
                <div className="text-[#6d6b63] text-[10px] uppercase tracking-wider">In-Memory RAM Peak</div>
                <div className="text-[#f5f4ee] text-sm font-semibold mt-0.5">{active.ramUsageMb} MB</div>
              </div>
            </div>
          </div>

          {/* Right Column: Exact Empirical Metrics Gauge */}
          <div className="lg:col-span-5 bg-[#0c0d10] border border-archival p-6 sm:p-8 space-y-6">
            <div className="flex items-center justify-between border-b border-archival pb-3">
              <span className="text-xs font-mono-code uppercase tracking-widest text-[#8e8d83]">
                Empirical Evaluation (k=5)
              </span>
              <span className="text-[11px] font-mono-code text-[#e28743] uppercase">
                FAISS Cosine
              </span>
            </div>

            {/* Metrics */}
            <div className="space-y-4">
              <div>
                <div className="flex justify-between text-xs font-mono-code mb-1">
                  <span className="text-[#a09e94]">Hit@5 (Recall Authority)</span>
                  <span className="text-[#f5f4ee] font-bold">{(active.hit5 * 100).toFixed(1)}%</span>
                </div>
                <div className="h-1 bg-[#1a1c22] w-full overflow-hidden">
                  <div
                    className="h-full bg-[#f5f4ee] transition-all duration-700"
                    style={{ width: `${active.hit5 * 100}%` }}
                  />
                </div>
              </div>

              <div>
                <div className="flex justify-between text-xs font-mono-code mb-1">
                  <span className="text-[#a09e94]">MRR@5 (Mean Reciprocal Rank)</span>
                  <span className="text-[#f5f4ee] font-bold">{active.mrr5.toFixed(3)}</span>
                </div>
                <div className="h-1 bg-[#1a1c22] w-full overflow-hidden">
                  <div
                    className="h-full bg-[#e28743] transition-all duration-700"
                    style={{ width: `${active.mrr5 * 100}%` }}
                  />
                </div>
              </div>

              <div>
                <div className="flex justify-between text-xs font-mono-code mb-1">
                  <span className="text-[#a09e94]">NDCG@5 (Graded Relevance)</span>
                  <span className="text-[#f5f4ee] font-bold">{active.ndcg5.toFixed(3)}</span>
                </div>
                <div className="h-1 bg-[#1a1c22] w-full overflow-hidden">
                  <div
                    className="h-full bg-[#a09e94] transition-all duration-700"
                    style={{ width: `${active.ndcg5 * 100}%` }}
                  />
                </div>
              </div>
            </div>

            {/* Speed & Bench stats */}
            <div className="grid grid-cols-2 gap-3 pt-3 border-t border-archival text-xs font-mono-code">
              <div>
                <span className="text-[#6d6b63] block text-[10px] uppercase">Query Latency p95</span>
                <span className="text-[#f5f4ee] text-base font-medium">{active.p95LatencyMs} ms</span>
              </div>
              <div>
                <span className="text-[#6d6b63] block text-[10px] uppercase">Corpus Ingest Time</span>
                <span className="text-[#f5f4ee] text-base font-medium">{active.embedTimeSec}s</span>
              </div>
            </div>

            {/* Run with CLI snippet */}
            <div className="pt-2">
              <span className="text-[10px] uppercase tracking-wider text-[#6d6b63] font-mono-code block mb-1.5">
                Execute This Model locally
              </span>
              <pre className="p-2.5 bg-[#14151a] border border-archival text-[11px] font-mono-code text-[#c7c5ba] overflow-x-auto">
                <code>embedbench run ./docs --models {active.name.toLowerCase().replace(/\s+/g, '-')}</code>
              </pre>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
