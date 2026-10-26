"use client";

import React, { useState } from "react";
import { Copy, Check, Terminal, CornerDownRight } from "lucide-react";

interface ArchivalStep {
  idx: string;
  category: string;
  heading: string;
  lead: string;
  snippet: string;
  notes: string;
}

const STEPS: ArchivalStep[] = [
  {
    idx: "01",
    category: "INSTALLATION",
    heading: "Bootstrap Local Binary",
    lead: "Install the standalone Python CLI environment with optional Hugging Face extras for local PyTorch execution.",
    snippet: "pip install embedbench\n# or with local Hugging Face SentenceTransformers:\npip install \"embedbench[hf]\"",
    notes: "Requires Python 3.10+. Zero cloud vendor accounts or API credits required."
  },
  {
    idx: "02",
    category: "INGESTION & BENCHMARK",
    heading: "Point at Target Corpus",
    lead: "Point directly at an uncurated directory of PDFs, Markdown notes, Word files, or technical manuals.",
    snippet: "embedbench run ./documentation \\\n  --models ollama/bge-m3,ollama/nomic-embed-text,all-MiniLM-L6-v2 \\\n  --generate 30",
    notes: "--generate 30 prompts your local Ollama LLM to formulate 30 grounded questions with quality deduplication."
  },
  {
    idx: "03",
    category: "EVALUATION SPECIFICATION",
    heading: "Ground-Truth Eval Set",
    lead: "Provide deterministic query-to-chunk target pairs in JSONL format to compute exact recall metrics.",
    snippet: "embedbench run ./docs \\\n  --models ollama/bge-m3,all-MiniLM-L6-v2 \\\n  --eval-set ./evals/ground_truth.jsonl \\\n  --top-k 1,3,5,10",
    notes: "Computes Hit@k, MRR@k, and NDCG@k strictly through per-model FAISS cosine indexes."
  },
  {
    idx: "04",
    category: "REPORT PUBLICATION",
    heading: "Generate Archival Report",
    lead: "Produce self-contained HTML monograph reports complete with latency percentiles, memory footprints, and recommendation badges.",
    snippet: "embedbench run ./docs \\\n  --config embedbench.yaml \\\n  --report-html ./reports/retrieval_study.html",
    notes: "Outputs can be viewed offline, shared as static artifacts, or piped directly into CI/CD pipelines."
  }
];

export function EditorialQuickstart() {
  const [activeStep, setActiveStep] = useState(0);
  const [copied, setCopied] = useState(false);

  const step = STEPS[activeStep];

  const handleCopy = () => {
    navigator.clipboard.writeText(step.snippet);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="w-full border border-archival bg-[#0f1013]">
      {/* Navigation tabs header */}
      <div className="grid grid-cols-2 md:grid-cols-4 border-b border-archival">
        {STEPS.map((s, i) => {
          const isSelected = activeStep === i;
          return (
            <button
              key={s.idx}
              onClick={() => {
                setActiveStep(i);
                setCopied(false);
              }}
              className={`p-4 sm:p-5 text-left border-r border-archival last:border-r-0 transition-colors ${
                isSelected
                  ? "bg-[#16181f] text-[#f5f4ee]"
                  : "bg-transparent text-[#7e7d74] hover:bg-[#121317] hover:text-[#c7c5ba]"
              }`}
            >
              <div className="flex items-center justify-between text-[11px] font-mono-code mb-1">
                <span>{s.idx}</span>
                <span className="text-[9px] uppercase tracking-widest">{s.category}</span>
              </div>
              <div className="font-serif-editorial italic text-sm sm:text-base font-normal truncate">
                {s.heading}
              </div>
            </button>
          );
        })}
      </div>

      {/* Main Step Detail */}
      <div className="p-6 sm:p-10 space-y-6">
        <div className="flex flex-col sm:flex-row sm:items-baseline justify-between gap-2 border-b border-archival pb-4">
          <div>
            <span className="text-[10px] uppercase font-mono-code tracking-[0.2em] text-[#e28743] block mb-1">
              Phase {step.idx} // {step.category}
            </span>
            <h3 className="text-2xl sm:text-3xl font-serif-editorial italic text-[#f5f4ee]">
              {step.heading}
            </h3>
          </div>
          <span className="text-xs font-mono-code text-[#73726b]">
            Standard Terminal Interface
          </span>
        </div>

        <p className="text-sm sm:text-base text-[#b0ae9f] leading-relaxed max-w-3xl">
          {step.lead}
        </p>

        {/* Code block */}
        <div className="relative border border-archival bg-[#08090b] p-4 sm:p-6 group">
          <div className="flex items-center justify-between mb-3 text-xs font-mono-code text-[#62615a] border-b border-[#1c1d24] pb-2">
            <span className="flex items-center gap-2">
              <Terminal className="w-3.5 h-3.5 text-[#e28743]" />
              bash command
            </span>
            <button
              onClick={handleCopy}
              className="flex items-center gap-1.5 px-2.5 py-1 border border-archival hover:border-[#8e8d83] bg-[#121317] text-[#c7c5ba] hover:text-white transition-colors"
            >
              {copied ? (
                <>
                  <Check className="w-3 h-3 text-[#10b981]" />
                  <span className="text-[#10b981]">Copied to Clipboard</span>
                </>
              ) : (
                <>
                  <Copy className="w-3 h-3" />
                  <span>Copy Shell Command</span>
                </>
              )}
            </button>
          </div>

          <pre className="font-mono-code text-xs sm:text-sm text-[#e8e7df] overflow-x-auto leading-relaxed py-1">
            <code>{step.snippet}</code>
          </pre>
        </div>

        {/* Footnote notes */}
        <div className="flex items-center gap-2 text-xs font-mono-code text-[#828076] pt-1">
          <CornerDownRight className="w-3.5 h-3.5 text-[#e28743] shrink-0" />
          <span>Note: {step.notes}</span>
        </div>
      </div>
    </div>
  );
}
