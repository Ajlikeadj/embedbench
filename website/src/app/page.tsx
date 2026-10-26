"use client";

import React, { useState, useEffect } from "react";

/* ─── Inline SVG Icons (no lucide dependency issues) ────────────────── */

function IconGithub({ className = "" }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="currentColor">
      <path d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.53 1.032 1.53 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" />
    </svg>
  );
}

function IconArrow({ className = "" }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
      <path d="M7 17L17 7M17 7H7M17 7V17" />
    </svg>
  );
}

function IconCopy({ className = "", style }: { className?: string; style?: React.CSSProperties }) {
  return (
    <svg className={className} style={style} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
      <rect x="9" y="9" width="13" height="13" rx="2" />
      <path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1" />
    </svg>
  );
}

function IconCheck({ className = "", style }: { className?: string; style?: React.CSSProperties }) {
  return (
    <svg className={className} style={style} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M20 6L9 17l-5-5" />
    </svg>
  );
}

/* ─── Styling constants ─────────────────────────────────────────────── */

const C = {
  bg: "#08090b",
  raised: "#0e1012",
  surface: "#141618",
  border: "rgba(255,255,255,0.08)",
  borderHover: "rgba(255,255,255,0.16)",
  text: "#f0efe8",
  textSec: "#a09d93",
  textMuted: "#6b6860",
  accent: "#e28743",
  accentDim: "rgba(226,135,67,0.12)",
} as const;

/* ─── Data ──────────────────────────────────────────────────────────── */

interface Model {
  id: string;
  vol: string;
  name: string;
  backend: string;
  dim: number;
  ctx: number;
  hit5: number;
  mrr5: number;
  ndcg5: number;
  latency: number;
  ingestSec: number;
  ramMb: number;
  badge: string;
  quote: string;
  source: string;
  ring: string;
}

const MODELS: Model[] = [
  {
    id: "bge-m3",
    vol: "VOL. I",
    name: "BAAI / BGE-M3",
    backend: "Ollama · PyTorch",
    dim: 1024,
    ctx: 8192,
    hit5: 0.942,
    mrr5: 0.884,
    ndcg5: 0.913,
    latency: 22.4,
    ingestSec: 7.8,
    ramMb: 1140,
    badge: "PINNACLE RECALL",
    quote: "Unrivaled multi-lingual comprehension. Dense retrieval precision that surfaces edge cases other models miss entirely.",
    source: "Retrieval Journal, Vol. IV",
    ring: "#d97706",
  },
  {
    id: "minilm",
    vol: "VOL. II",
    name: "all-MiniLM-L6-v2",
    backend: "HuggingFace · ONNX",
    dim: 384,
    ctx: 512,
    hit5: 0.865,
    mrr5: 0.791,
    ndcg5: 0.824,
    latency: 5.6,
    ingestSec: 1.9,
    ramMb: 240,
    badge: "ULTRALIGHT SPEED",
    quote: "Featherweight champion. 4× faster indexing with near-instant CPU execution. Perfect for constrained hardware.",
    source: "Benchmark Digest",
    ring: "#10b981",
  },
  {
    id: "nomic",
    vol: "VOL. III",
    name: "Nomic Embed v1.5",
    backend: "Ollama · GGUF",
    dim: 768,
    ctx: 8192,
    hit5: 0.918,
    mrr5: 0.843,
    ndcg5: 0.879,
    latency: 13.8,
    ingestSec: 4.2,
    ramMb: 580,
    badge: "BALANCED PICK",
    quote: "The sweet spot of modern RAG. Extended 8k context, disciplined memory, and consistent ranking authority.",
    source: "Local AI Monograph",
    ring: "#3b82f6",
  },
  {
    id: "bge-large",
    vol: "VOL. IV",
    name: "BAAI / BGE-Large v1.5",
    backend: "Ollama · PyTorch",
    dim: 1024,
    ctx: 512,
    hit5: 0.951,
    mrr5: 0.899,
    ndcg5: 0.928,
    latency: 34.2,
    ingestSec: 11.5,
    ramMb: 1380,
    badge: "HIGH CAPACITY",
    quote: "Heavyweight semantic discrimination. Demands capable hardware but returns near-human relevancy scores.",
    source: "ArXiv Empirical Review",
    ring: "#a855f7",
  },
];

const STEPS = [
  {
    num: "01",
    label: "INSTALL",
    title: "Bootstrap",
    cmd: `pip install embedbench\n# or with HuggingFace extras:\npip install "embedbench[hf]"`,
    note: "Python 3.10+. Zero cloud accounts required.",
  },
  {
    num: "02",
    label: "BENCHMARK",
    title: "Point & Run",
    cmd: `embedbench run ./docs \\\n  --models ollama/bge-m3,all-MiniLM-L6-v2 \\\n  --generate 30`,
    note: "--generate creates synthetic questions via your local Ollama LLM.",
  },
  {
    num: "03",
    label: "EVALUATE",
    title: "Ground Truth",
    cmd: `embedbench run ./docs \\\n  --eval-set queries.jsonl \\\n  --top-k 1,3,5,10`,
    note: "JSONL with query + positive chunk IDs for deterministic metrics.",
  },
  {
    num: "04",
    label: "REPORT",
    title: "Publish",
    cmd: `embedbench run ./docs \\\n  --config embedbench.yaml \\\n  --report-html ./out/report.html`,
    note: "Self-contained HTML with charts. Also outputs JSON & CSV.",
  },
];

/* ─── Utility: copy to clipboard ────────────────────────────────────── */

function useCopy(timeout = 2000) {
  const [copied, setCopied] = useState(false);
  const copy = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), timeout);
  };
  return { copied, copy };
}

/* ─── Metric Bar ────────────────────────────────────────────────────── */

function MetricBar({
  label,
  value,
  max = 1,
  format,
  color = C.text,
}: {
  label: string;
  value: number;
  max?: number;
  format: string;
  color?: string;
}) {
  return (
    <div>
      <div className="flex justify-between mb-1" style={{ fontSize: 11, fontFamily: "var(--font-mono)" }}>
        <span style={{ color: C.textSec }}>{label}</span>
        <span style={{ color: C.text, fontWeight: 600 }}>{format}</span>
      </div>
      <div style={{ height: 2, background: "rgba(255,255,255,0.06)", borderRadius: 1 }}>
        <div
          style={{
            height: "100%",
            width: `${(value / max) * 100}%`,
            background: color,
            borderRadius: 1,
            transition: "width 0.6s ease",
          }}
        />
      </div>
    </div>
  );
}

/* ─── Disc Component ────────────────────────────────────────────────── */

function Disc({
  model,
  isActive,
  onClick,
}: {
  model: Model;
  isActive: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className="group relative flex flex-col items-center text-center"
      style={{
        transition: "transform 0.4s ease, opacity 0.3s ease",
        transform: isActive ? "scale(1)" : "scale(0.88)",
        opacity: isActive ? 1 : 0.5,
      }}
    >
      {/* Vinyl disc */}
      <div
        className={isActive ? "disc-spin" : ""}
        style={{
          width: isActive ? 200 : 160,
          height: isActive ? 200 : 160,
          borderRadius: "50%",
          background: `radial-gradient(circle at 50% 50%, #1a1a1a 30%, #0d0d0d 70%, #080808 100%)`,
          border: `2px solid ${isActive ? model.ring : "rgba(255,255,255,0.06)"}`,
          boxShadow: isActive
            ? `0 0 40px ${model.ring}20, 0 0 80px ${model.ring}10, inset 0 0 30px rgba(0,0,0,0.8)`
            : "inset 0 0 30px rgba(0,0,0,0.6)",
          position: "relative",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          transition: "all 0.5s ease",
          cursor: "pointer",
        }}
      >
        {/* Grooves */}
        {[16, 28, 40, 52].map((inset) => (
          <div
            key={inset}
            style={{
              position: "absolute",
              inset: inset,
              borderRadius: "50%",
              border: "1px solid rgba(255,255,255,0.04)",
              pointerEvents: "none",
            }}
          />
        ))}

        {/* Center label */}
        <div
          style={{
            width: 44,
            height: 44,
            borderRadius: "50%",
            background: isActive ? model.ring : "#1a1c20",
            border: "2px solid rgba(255,255,255,0.1)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            transition: "background 0.4s ease",
          }}
        >
          <div style={{ width: 8, height: 8, borderRadius: "50%", background: C.bg, border: "1px solid rgba(255,255,255,0.15)" }} />
        </div>
      </div>

      {/* Label below disc */}
      <div className="mt-5" style={{ transition: "opacity 0.3s", opacity: isActive ? 1 : 0.6 }}>
        <div style={{ fontSize: 10, fontFamily: "var(--font-mono)", color: C.accent, letterSpacing: "0.15em", marginBottom: 4 }}>
          {model.vol}
        </div>
        <div
          style={{
            fontFamily: "var(--font-display)",
            fontStyle: "italic",
            fontSize: isActive ? 18 : 14,
            color: C.text,
            transition: "font-size 0.4s ease",
          }}
        >
          {model.name}
        </div>
        <div style={{ fontSize: 10, fontFamily: "var(--font-mono)", color: C.textMuted, marginTop: 2 }}>
          {model.badge}
        </div>
      </div>
    </button>
  );
}

/* ═══════════════════════════════════════════════════════════════════════
   PAGE
   ═══════════════════════════════════════════════════════════════════════ */

export default function Home() {
  const [activeModel, setActiveModel] = useState(0);
  const [activeStep, setActiveStep] = useState(0);
  const heroClip = useCopy();
  const stepClip = useCopy();
  const m = MODELS[activeModel];
  const step = STEPS[activeStep];

  return (
    <div style={{ minHeight: "100vh", display: "flex", flexDirection: "column" }}>

      {/* ─── NAV ──────────────────────────────────────────────────────── */}
      <nav
        style={{
          position: "sticky",
          top: 0,
          zIndex: 100,
          backdropFilter: "blur(16px) saturate(1.2)",
          WebkitBackdropFilter: "blur(16px) saturate(1.2)",
          background: "rgba(8,9,11,0.85)",
          borderBottom: `1px solid ${C.border}`,
        }}
      >
        <div style={{ maxWidth: 1200, margin: "0 auto", padding: "0 24px", display: "flex", alignItems: "center", justifyContent: "space-between", height: 64 }}>
          <div style={{ display: "flex", alignItems: "baseline", gap: 10 }}>
            <span style={{ fontFamily: "var(--font-display)", fontStyle: "italic", fontSize: 24, color: C.text }}>
              embedbench
            </span>
            <span style={{ fontFamily: "var(--font-mono)", fontSize: 10, color: C.textMuted, letterSpacing: "0.15em" }}>
              v0.1.0
            </span>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: 32 }}>
            <div className="hidden md:flex" style={{ gap: 28 }}>
              {["#catalogue", "#manifesto", "#quickstart", "#architecture", "#cli"].map((href) => (
                <a
                  key={href}
                  href={href}
                  style={{
                    fontFamily: "var(--font-mono)",
                    fontSize: 11,
                    color: C.textSec,
                    textDecoration: "none",
                    letterSpacing: "0.08em",
                    textTransform: "uppercase",
                    transition: "color 0.2s",
                  }}
                  onMouseEnter={(e) => (e.currentTarget.style.color = C.text)}
                  onMouseLeave={(e) => (e.currentTarget.style.color = C.textSec)}
                >
                  {href.replace("#", "")}
                </a>
              ))}
            </div>

            <a
              href="https://github.com/Ajlikeadj/embedbench"
              target="_blank"
              rel="noreferrer"
              style={{
                display: "flex",
                alignItems: "center",
                gap: 6,
                padding: "6px 14px",
                border: `1px solid ${C.border}`,
                color: C.textSec,
                textDecoration: "none",
                fontFamily: "var(--font-mono)",
                fontSize: 11,
                transition: "all 0.2s",
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.borderColor = C.borderHover;
                e.currentTarget.style.color = C.text;
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.borderColor = C.border;
                e.currentTarget.style.color = C.textSec;
              }}
            >
              <IconGithub className="w-4 h-4" />
              <span>GitHub</span>
            </a>
          </div>
        </div>
      </nav>

      {/* ─── HERO ─────────────────────────────────────────────────────── */}
      <section style={{ padding: "100px 24px 120px", maxWidth: 1200, margin: "0 auto", width: "100%" }}>
        <div className="animate-fade-up" style={{ maxWidth: 820 }}>
          {/* Eyebrow */}
          <div style={{
            display: "inline-flex",
            alignItems: "center",
            gap: 8,
            marginBottom: 28,
            fontFamily: "var(--font-mono)",
            fontSize: 11,
            color: C.accent,
            letterSpacing: "0.2em",
            textTransform: "uppercase",
          }}>
            <span className="animate-pulse-glow" style={{ width: 6, height: 6, borderRadius: "50%", background: C.accent }} />
            100% Local · Zero Cloud APIs
          </div>

          {/* Title */}
          <h1 style={{
            fontFamily: "var(--font-display)",
            fontStyle: "italic",
            fontWeight: 400,
            fontSize: "clamp(36px, 6vw, 72px)",
            lineHeight: 1.05,
            color: C.text,
            letterSpacing: "-0.02em",
            marginBottom: 28,
          }}>
            Benchmark embedding models{" "}
            <span style={{ borderBottom: `1px solid ${C.accent}60`, paddingBottom: 4 }}>
              on your own documents
            </span>
          </h1>

          {/* Subtitle */}
          <p className="animate-fade-up-delay" style={{
            fontSize: "clamp(15px, 1.5vw, 18px)",
            lineHeight: 1.65,
            color: C.textSec,
            maxWidth: 600,
            marginBottom: 40,
          }}>
            Public leaderboards rank on synthetic trivia. embedbench ingests your PDFs, manuals, and support tickets — measures recall with FAISS cosine indexing — and reveals the true champion for your domain.
          </p>

          {/* Install CTA */}
          <div className="animate-fade-up-delay-2" style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
            <button
              onClick={() => heroClip.copy("pip install embedbench")}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 10,
                padding: "12px 20px",
                background: C.surface,
                border: `1px solid ${C.border}`,
                color: C.text,
                fontFamily: "var(--font-mono)",
                fontSize: 13,
                cursor: "pointer",
                transition: "all 0.2s",
              }}
              onMouseEnter={(e) => (e.currentTarget.style.borderColor = C.borderHover)}
              onMouseLeave={(e) => (e.currentTarget.style.borderColor = C.border)}
            >
              <span style={{ color: C.textMuted }}>$</span>
              pip install embedbench
              {heroClip.copied ? (
                <IconCheck className="w-4 h-4" style={{ color: "#10b981" }} />
              ) : (
                <IconCopy className="w-4 h-4" style={{ color: C.textMuted }} />
              )}
            </button>

            <a
              href="#catalogue"
              style={{
                display: "flex",
                alignItems: "center",
                gap: 6,
                padding: "12px 24px",
                background: C.text,
                color: C.bg,
                fontFamily: "var(--font-mono)",
                fontSize: 12,
                fontWeight: 600,
                letterSpacing: "0.05em",
                textTransform: "uppercase",
                textDecoration: "none",
                transition: "opacity 0.2s",
              }}
              onMouseEnter={(e) => (e.currentTarget.style.opacity = "0.85")}
              onMouseLeave={(e) => (e.currentTarget.style.opacity = "1")}
            >
              Explore Models <IconArrow className="w-4 h-4" />
            </a>
          </div>
        </div>
      </section>

      {/* ─── MODEL CATALOGUE (Disc Crate) ─────────────────────────────── */}
      <section
        id="catalogue"
        style={{
          borderTop: `1px solid ${C.border}`,
          borderBottom: `1px solid ${C.border}`,
          background: C.raised,
          padding: "80px 24px 100px",
        }}
      >
        <div style={{ maxWidth: 1200, margin: "0 auto" }}>
          {/* Section header */}
          <div style={{ display: "flex", flexWrap: "wrap", justifyContent: "space-between", alignItems: "flex-end", marginBottom: 60, gap: 16 }}>
            <div>
              <div style={{ fontFamily: "var(--font-mono)", fontSize: 11, color: C.accent, letterSpacing: "0.2em", textTransform: "uppercase", marginBottom: 8 }}>
                Archive Catalogue
              </div>
              <h2 style={{ fontFamily: "var(--font-display)", fontStyle: "italic", fontWeight: 400, fontSize: "clamp(28px, 4vw, 48px)", color: C.text, letterSpacing: "-0.01em" }}>
                Local Embedding Editions
              </h2>
            </div>
            <p style={{ fontFamily: "var(--font-mono)", fontSize: 12, color: C.textMuted, maxWidth: 340 }}>
              Select a disc to inspect embedding dimensions, retrieval accuracy, and hardware cost.
            </p>
          </div>

          {/* Disc carousel */}
          <div style={{ display: "flex", justifyContent: "center", alignItems: "flex-end", gap: "clamp(24px, 4vw, 60px)", marginBottom: 64, flexWrap: "wrap" }}>
            {MODELS.map((model, i) => (
              <Disc key={model.id} model={model} isActive={activeModel === i} onClick={() => setActiveModel(i)} />
            ))}
          </div>

          {/* Active model detail panel */}
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "1fr",
              gap: 0,
              border: `1px solid ${C.border}`,
              background: C.bg,
            }}
            className="lg:grid-cols-2"
          >
            {/* Left: editorial quote */}
            <div style={{ padding: "clamp(24px, 4vw, 48px)", borderRight: `1px solid ${C.border}` }}>
              <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 24 }}>
                <span style={{ fontFamily: "var(--font-mono)", fontSize: 10, letterSpacing: "0.15em", padding: "3px 8px", border: `1px solid ${C.border}`, color: C.text }}>
                  {m.vol}
                </span>
                <span style={{ fontFamily: "var(--font-mono)", fontSize: 11, color: C.textMuted }}>
                  {m.backend}
                </span>
              </div>

              <blockquote style={{
                fontFamily: "var(--font-display)",
                fontStyle: "italic",
                fontWeight: 400,
                fontSize: "clamp(20px, 2.5vw, 28px)",
                lineHeight: 1.4,
                color: C.text,
                marginBottom: 20,
              }}>
                &ldquo;{m.quote}&rdquo;
              </blockquote>

              <div style={{ fontFamily: "var(--font-mono)", fontSize: 11, color: C.textMuted }}>
                <span style={{ color: C.accent }}>★ ★ ★ ★ ★</span>
                {" "}— {m.source}
              </div>

              <div style={{ display: "flex", flexWrap: "wrap", gap: 12, marginTop: 28 }}>
                {[
                  { label: "Dimensions", value: m.dim.toString() },
                  { label: "Context", value: `${m.ctx.toLocaleString()} tok` },
                  { label: "RAM Peak", value: `${m.ramMb} MB` },
                ].map((stat) => (
                  <div key={stat.label} style={{ padding: "10px 14px", border: `1px solid ${C.border}`, background: C.raised }}>
                    <div style={{ fontFamily: "var(--font-mono)", fontSize: 9, color: C.textMuted, letterSpacing: "0.1em", textTransform: "uppercase" }}>
                      {stat.label}
                    </div>
                    <div style={{ fontFamily: "var(--font-mono)", fontSize: 14, color: C.text, fontWeight: 600, marginTop: 2 }}>
                      {stat.value}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Right: metrics */}
            <div style={{ padding: "clamp(24px, 4vw, 48px)" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", borderBottom: `1px solid ${C.border}`, paddingBottom: 12, marginBottom: 24 }}>
                <span style={{ fontFamily: "var(--font-mono)", fontSize: 11, letterSpacing: "0.1em", textTransform: "uppercase", color: C.textSec }}>
                  FAISS Cosine Retrieval (k=5)
                </span>
                <span style={{ fontFamily: "var(--font-mono)", fontSize: 10, color: C.accent }}>{m.badge}</span>
              </div>

              <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                <MetricBar label="Hit@5" value={m.hit5} format={`${(m.hit5 * 100).toFixed(1)}%`} color={m.ring} />
                <MetricBar label="MRR@5" value={m.mrr5} format={m.mrr5.toFixed(3)} color={C.accent} />
                <MetricBar label="NDCG@5" value={m.ndcg5} format={m.ndcg5.toFixed(3)} color={C.textSec} />
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginTop: 28, paddingTop: 20, borderTop: `1px solid ${C.border}` }}>
                <div>
                  <div style={{ fontFamily: "var(--font-mono)", fontSize: 9, color: C.textMuted, letterSpacing: "0.1em", textTransform: "uppercase" }}>
                    Query p95
                  </div>
                  <div style={{ fontFamily: "var(--font-mono)", fontSize: 20, color: C.text, fontWeight: 600 }}>
                    {m.latency}<span style={{ fontSize: 12, color: C.textMuted }}> ms</span>
                  </div>
                </div>
                <div>
                  <div style={{ fontFamily: "var(--font-mono)", fontSize: 9, color: C.textMuted, letterSpacing: "0.1em", textTransform: "uppercase" }}>
                    Corpus Ingest
                  </div>
                  <div style={{ fontFamily: "var(--font-mono)", fontSize: 20, color: C.text, fontWeight: 600 }}>
                    {m.ingestSec}<span style={{ fontSize: 12, color: C.textMuted }}> sec</span>
                  </div>
                </div>
              </div>

              <div style={{ marginTop: 24, padding: "10px 14px", background: C.raised, border: `1px solid ${C.border}`, fontFamily: "var(--font-mono)", fontSize: 12, color: C.textSec }}>
                <span style={{ color: C.textMuted }}>$</span>{" "}
                <span style={{ color: C.text }}>embedbench run ./docs --models {m.id}</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ─── MANIFESTO ────────────────────────────────────────────────── */}
      <section id="manifesto" style={{ padding: "100px 24px", maxWidth: 1200, margin: "0 auto" }}>
        <div style={{ maxWidth: 600, marginBottom: 64 }}>
          <div style={{ fontFamily: "var(--font-mono)", fontSize: 11, color: C.accent, letterSpacing: "0.2em", textTransform: "uppercase", marginBottom: 8 }}>
            Manifesto
          </div>
          <h2 style={{ fontFamily: "var(--font-display)", fontStyle: "italic", fontWeight: 400, fontSize: "clamp(28px, 4vw, 44px)", color: C.text, lineHeight: 1.15 }}>
            Why public rankings fail in production
          </h2>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 1, background: C.border }}>
          {[
            {
              num: "01",
              title: "Domain Mismatch",
              body: "MTEB scores against Wikipedia and public QA. Your medical transcripts, codebase symbols, and internal acronyms occupy a completely different embedding geometry.",
            },
            {
              num: "02",
              title: "Hardware Blindness",
              body: "Leaderboards ignore cold-start latency, 8GB VRAM limits, and your application's p95 SLA. embedbench profiles real memory and timing on your target hardware.",
            },
            {
              num: "03",
              title: "Air-Gapped Privacy",
              body: "Cloud evaluation APIs leak proprietary IP. embedbench runs 100% locally — zero telemetry, zero internet requests, zero third-party tokens.",
            },
          ].map((item) => (
            <div key={item.num} style={{ background: C.bg, padding: "clamp(28px, 3vw, 48px)" }}>
              <div style={{ fontFamily: "var(--font-mono)", fontSize: 11, color: C.accent, marginBottom: 16 }}>
                {item.num}
              </div>
              <h3 style={{ fontFamily: "var(--font-display)", fontStyle: "italic", fontSize: 20, color: C.text, marginBottom: 12 }}>
                {item.title}
              </h3>
              <p style={{ fontSize: 13, lineHeight: 1.7, color: C.textSec }}>
                {item.body}
              </p>
            </div>
          ))}
        </div>
      </section>

      {/* ─── QUICKSTART ───────────────────────────────────────────────── */}
      <section
        id="quickstart"
        style={{
          borderTop: `1px solid ${C.border}`,
          borderBottom: `1px solid ${C.border}`,
          background: C.raised,
          padding: "80px 24px 100px",
        }}
      >
        <div style={{ maxWidth: 1200, margin: "0 auto" }}>
          <div style={{ marginBottom: 48 }}>
            <div style={{ fontFamily: "var(--font-mono)", fontSize: 11, color: C.accent, letterSpacing: "0.2em", textTransform: "uppercase", marginBottom: 8 }}>
              Execution Guide
            </div>
            <h2 style={{ fontFamily: "var(--font-display)", fontStyle: "italic", fontWeight: 400, fontSize: "clamp(28px, 4vw, 44px)", color: C.text }}>
              Four steps to your first benchmark
            </h2>
          </div>

          {/* Step tabs */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", borderBottom: `1px solid ${C.border}` }}>
            {STEPS.map((s, i) => (
              <button
                key={s.num}
                onClick={() => { setActiveStep(i); stepClip.copy(""); }}
                style={{
                  padding: "16px 20px",
                  textAlign: "left",
                  background: activeStep === i ? C.bg : "transparent",
                  borderBottom: activeStep === i ? `2px solid ${C.accent}` : "2px solid transparent",
                  cursor: "pointer",
                  border: "none",
                  borderRight: `1px solid ${C.border}`,
                  transition: "all 0.2s",
                  fontFamily: "var(--font-mono)",
                }}
              >
                <div style={{ fontSize: 10, color: activeStep === i ? C.accent : C.textMuted, letterSpacing: "0.15em", marginBottom: 4 }}>
                  {s.num} / {s.label}
                </div>
                <div style={{
                  fontFamily: "var(--font-display)",
                  fontStyle: "italic",
                  fontSize: 16,
                  color: activeStep === i ? C.text : C.textSec,
                }}>
                  {s.title}
                </div>
              </button>
            ))}
          </div>

          {/* Step content */}
          <div style={{ background: C.bg, border: `1px solid ${C.border}`, borderTop: "none", padding: "clamp(24px, 3vw, 40px)" }}>
            <div style={{ position: "relative" }}>
              <button
                onClick={() => stepClip.copy(step.cmd)}
                style={{
                  position: "absolute",
                  top: 12,
                  right: 12,
                  display: "flex",
                  alignItems: "center",
                  gap: 6,
                  padding: "5px 12px",
                  background: C.surface,
                  border: `1px solid ${C.border}`,
                  color: stepClip.copied ? "#10b981" : C.textSec,
                  fontFamily: "var(--font-mono)",
                  fontSize: 11,
                  cursor: "pointer",
                  transition: "all 0.2s",
                }}
              >
                {stepClip.copied ? <><IconCheck className="w-3 h-3" /> Copied</> : <><IconCopy className="w-3 h-3" /> Copy</>}
              </button>

              <pre style={{
                fontFamily: "var(--font-mono)",
                fontSize: 13,
                color: C.text,
                lineHeight: 1.7,
                padding: "20px 0",
                margin: 0,
                overflow: "auto",
                background: "transparent",
              }}>
                <code>{step.cmd}</code>
              </pre>
            </div>

            <div style={{
              marginTop: 16,
              paddingTop: 16,
              borderTop: `1px solid ${C.border}`,
              fontFamily: "var(--font-mono)",
              fontSize: 12,
              color: C.textMuted,
              display: "flex",
              alignItems: "center",
              gap: 8,
            }}>
              <span style={{ color: C.accent }}>→</span>
              {step.note}
            </div>
          </div>
        </div>
      </section>

      {/* ─── ARCHITECTURE ─────────────────────────────────────────────── */}
      <section id="architecture" style={{ padding: "100px 24px", maxWidth: 1200, margin: "0 auto" }}>
        <div className="grid grid-cols-1 lg:grid-cols-2" style={{ gap: 48, alignItems: "start" }}>
          <div>
            <div style={{ fontFamily: "var(--font-mono)", fontSize: 11, color: C.accent, letterSpacing: "0.2em", textTransform: "uppercase", marginBottom: 8 }}>
              Architecture
            </div>
            <h2 style={{ fontFamily: "var(--font-display)", fontStyle: "italic", fontWeight: 400, fontSize: "clamp(28px, 3vw, 40px)", color: C.text, marginBottom: 20, lineHeight: 1.15 }}>
              Deterministic evaluation pipeline
            </h2>
            <p style={{ fontSize: 14, lineHeight: 1.7, color: C.textSec, marginBottom: 32 }}>
              A single reproducible loop: parse → chunk → embed → FAISS index → query → score. Every run snapshots the exact environment, commit hash, and configuration for later comparison.
            </p>

            {[
              "Document Parsers — .txt .md .pdf .docx .html",
              "Chunking — Fixed, Sentence-aware, Recursive",
              "FAISS Cosine Index — Hit@k, MRR@k, NDCG@k",
              "Environment Snapshot — Full reproducibility log",
            ].map((line, i) => (
              <div
                key={i}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 12,
                  padding: "12px 16px",
                  border: `1px solid ${C.border}`,
                  background: C.raised,
                  marginBottom: 8,
                  fontFamily: "var(--font-mono)",
                  fontSize: 12,
                  color: C.textSec,
                }}
              >
                <span style={{ color: C.accent, fontWeight: 600, minWidth: 20 }}>
                  {String(i + 1).padStart(2, "0")}
                </span>
                {line}
              </div>
            ))}
          </div>

          <div style={{ border: `1px solid ${C.border}`, background: C.raised }}>
            <div style={{ display: "flex", justifyContent: "space-between", padding: "12px 20px", borderBottom: `1px solid ${C.border}`, fontFamily: "var(--font-mono)", fontSize: 11, color: C.textMuted }}>
              <span>embedbench.yaml</span>
              <span>Configuration</span>
            </div>
            <pre style={{
              fontFamily: "var(--font-mono)",
              fontSize: 12,
              lineHeight: 1.65,
              color: C.textSec,
              padding: "24px 20px",
              margin: 0,
              overflow: "auto",
            }}>
{`corpus:
  path: "./technical_manuals"
  include_patterns: ["*.md", "*.pdf"]
  chunking:
    strategy: "recursive"
    chunk_size: 512
    chunk_overlap: 64

evaluation:
  mode: "synthetic"
  synthetic_count: 50
  ollama_generator_model: "llama3.2"
  top_k: [1, 3, 5, 10]

models:
  - id: "ollama/bge-m3"
    batch_size: 16
  - id: "all-MiniLM-L6-v2"
    device: "cpu"
  - id: "ollama/nomic-embed-text"
    batch_size: 32

output:
  formats: ["console", "json", "html"]
  report_path: "./reports/benchmark.html"`}
            </pre>
          </div>
        </div>
      </section>

      {/* ─── CLI REFERENCE ────────────────────────────────────────────── */}
      <section
        id="cli"
        style={{
          borderTop: `1px solid ${C.border}`,
          background: C.raised,
          padding: "80px 24px 100px",
        }}
      >
        <div style={{ maxWidth: 1200, margin: "0 auto" }}>
          <div style={{ marginBottom: 48 }}>
            <div style={{ fontFamily: "var(--font-mono)", fontSize: 11, color: C.accent, letterSpacing: "0.2em", textTransform: "uppercase", marginBottom: 8 }}>
              Reference
            </div>
            <h2 style={{ fontFamily: "var(--font-display)", fontStyle: "italic", fontWeight: 400, fontSize: "clamp(28px, 4vw, 40px)", color: C.text }}>
              CLI Commands
            </h2>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 1, background: C.border }}>
            {[
              {
                cmd: "embedbench run <DIR>",
                tag: "Primary",
                desc: "Full benchmark pipeline.",
                flags: ["--models <str>", "--eval-set <file>", "--generate <int>", "--top-k <ints>", "--report-html <path>"],
              },
              {
                cmd: "embedbench generate",
                tag: "Synthetic",
                desc: "Create eval question sets via local LLM.",
                flags: ["--corpus <dir>", "--count <int>", "--model <name>", "--out <file.jsonl>"],
              },
              {
                cmd: "embedbench list-models",
                tag: "Discovery",
                desc: "Detect available local embedding models.",
                flags: ["--source ollama", "--source hf"],
              },
              {
                cmd: "embedbench report",
                tag: "Render",
                desc: "Re-render reports from previous JSON runs.",
                flags: ["--input <run.json>", "--output <out.html>"],
              },
            ].map((item) => (
              <div key={item.cmd} style={{ background: C.bg, padding: "clamp(20px, 3vw, 32px)" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12, paddingBottom: 10, borderBottom: `1px solid ${C.border}` }}>
                  <span style={{ fontFamily: "var(--font-mono)", fontSize: 12, fontWeight: 600, color: C.accent }}>{item.cmd}</span>
                  <span style={{ fontFamily: "var(--font-mono)", fontSize: 9, color: C.textMuted, letterSpacing: "0.1em", textTransform: "uppercase" }}>{item.tag}</span>
                </div>
                <p style={{ fontFamily: "var(--font-mono)", fontSize: 11, color: C.textSec, marginBottom: 12 }}>{item.desc}</p>
                <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                  {item.flags.map((flag) => (
                    <div key={flag} style={{ fontFamily: "var(--font-mono)", fontSize: 11, color: C.textMuted }}>
                      <code style={{ color: C.text }}>{flag.split(" ")[0]}</code> {flag.split(" ").slice(1).join(" ")}
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ─── FOOTER ───────────────────────────────────────────────────── */}
      <footer style={{ borderTop: `1px solid ${C.border}`, padding: "48px 24px", background: C.bg }}>
        <div style={{ maxWidth: 1200, margin: "0 auto", display: "flex", flexWrap: "wrap", justifyContent: "space-between", alignItems: "center", gap: 20 }}>
          <div>
            <span style={{ fontFamily: "var(--font-display)", fontStyle: "italic", fontSize: 18, color: C.text }}>
              embedbench
            </span>
            <span style={{ fontFamily: "var(--font-mono)", fontSize: 11, color: C.textMuted, marginLeft: 12 }}>
              Local RAG retrieval benchmark · MIT License
            </span>
          </div>
          <div style={{ display: "flex", gap: 24, fontFamily: "var(--font-mono)", fontSize: 11 }}>
            <a href="https://github.com/Ajlikeadj/embedbench" target="_blank" rel="noreferrer"
              style={{ color: C.textSec, textDecoration: "none", display: "flex", alignItems: "center", gap: 4 }}
            >
              <IconGithub className="w-3.5 h-3.5" /> Repository
            </a>
            <a href="https://github.com/Ajlikeadj/embedbench/blob/main/LICENSE" target="_blank" rel="noreferrer"
              style={{ color: C.textSec, textDecoration: "none" }}
            >
              License
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
}
