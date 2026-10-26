import React, { useState } from "react";
import { Copy, Check, Terminal, Play, CheckCircle2 } from "lucide-react";

interface Step {
  id: string;
  title: string;
  description: string;
  command: string;
  notes?: string;
}

const steps: Step[] = [
  {
    id: "install",
    title: "1. Install embedbench",
    description: "Install the core CLI via pip or with optional Hugging Face extras.",
    command: "pip install embedbench\n# or with local Hugging Face SentenceTransformers:\npip install \"embedbench[hf]\"",
    notes: "Requires Python 3.10+. If running Ollama, ensure your ollama daemon is running."
  },
  {
    id: "quickstart",
    title: "2. One-line Quickstart",
    description: "Run a benchmark directly against a folder of PDFs, Markdown, TXT, or Word files.",
    command: "embedbench run ./docs --models ollama/nomic-embed-text,ollama/bge-m3 --generate 30",
    notes: "--generate 30 creates 30 synthetic questions via your local Ollama LLM with deduplication and quality filters."
  },
  {
    id: "eval-set",
    title: "3. Bring Your Own Eval Set",
    description: "Use ground-truth question-chunk pairs in JSONL format for deterministic evaluations.",
    command: "embedbench run ./docs --models ollama/bge-large,all-MiniLM-L6-v2 --eval-set queries.jsonl",
    notes: "JSONL contains queries and positive chunk/doc IDs to compute Hit@k, MRR@k, and NDCG@k."
  },
  {
    id: "report",
    title: "4. Generate Standalone HTML Reports",
    description: "Produce interactive HTML dashboards with latency breakdown charts and recommendation badges.",
    command: "embedbench run ./docs --config embedbench.yaml --report-html ./reports/benchmark.html",
    notes: "Export options include terminal tables, machine-readable JSON, CSV, and self-contained HTML charts."
  }
];

export function QuickstartTabs() {
  const [activeStep, setActiveStep] = useState(0);
  const [copied, setCopied] = useState(false);

  const step = steps[activeStep];

  const handleCopy = () => {
    navigator.clipboard.writeText(step.command);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-6 shadow-2xl backdrop-blur-sm">
      <div className="flex flex-wrap gap-2 border-b border-slate-800 pb-4 mb-6">
        {steps.map((s, idx) => (
          <button
            key={s.id}
            onClick={() => {
              setActiveStep(idx);
              setCopied(false);
            }}
            className={`px-4 py-2 text-sm font-medium rounded-lg transition-all flex items-center gap-2 ${
              activeStep === idx
                ? "bg-indigo-600 text-white shadow-md shadow-indigo-600/30"
                : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60"
            }`}
          >
            <span>{s.title}</span>
          </button>
        ))}
      </div>

      <div className="space-y-4">
        <p className="text-slate-300 text-sm">{step.description}</p>

        <div className="relative group">
          <div className="absolute top-3 right-3 flex items-center gap-2">
            <button
              onClick={handleCopy}
              className="p-1.5 rounded-md bg-slate-800/80 hover:bg-slate-700 text-slate-300 transition-colors border border-slate-700/60 flex items-center gap-1.5 text-xs font-mono"
              title="Copy command"
            >
              {copied ? (
                <>
                  <Check className="w-3.5 h-3.5 text-emerald-400" />
                  <span className="text-emerald-400">Copied!</span>
                </>
              ) : (
                <>
                  <Copy className="w-3.5 h-3.5" />
                  <span>Copy</span>
                </>
              )}
            </button>
          </div>

          <pre className="p-4 bg-slate-950 rounded-xl border border-slate-800/90 text-indigo-300 font-mono text-xs sm:text-sm overflow-x-auto leading-relaxed">
            <code>{step.command}</code>
          </pre>
        </div>

        {step.notes && (
          <div className="flex items-start gap-2 text-xs text-slate-400 bg-slate-800/40 p-3 rounded-lg border border-slate-800/70">
            <CheckCircle2 className="w-4 h-4 text-indigo-400 mt-0.5 shrink-0" />
            <span>{step.notes}</span>
          </div>
        )}
      </div>
    </div>
  );
}
