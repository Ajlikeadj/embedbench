import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "embedbench — Local Embedding Benchmarks for RAG",
  description:
    "Benchmark local embedding models for RAG retrieval on your own documents. Per-model FAISS indexes, speed/accuracy/cost comparisons, zero cloud API keys.",
  openGraph: {
    title: "embedbench — Local Embedding Benchmarks for RAG",
    description:
      "Benchmark embedding models on your own documents locally without cloud API keys.",
    url: "https://embedbench-opal.vercel.app",
    siteName: "embedbench",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400..900;1,400..900&family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="noise-bg">
        {children}
      </body>
    </html>
  );
}
