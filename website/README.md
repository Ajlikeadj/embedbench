# embedbench Documentation Website

This directory contains the Next.js and Tailwind CSS documentation & interactive showcase website for **[embedbench](https://github.com/Ajlikeadj/embedbench)**.

It includes:
- **Interactive Retrieval Playground**: Preview comparisons of Hit@k, MRR@k, latency, and memory across models.
- **Quickstart & Integration Walkthrough**: Copy-paste CLI commands and config templates.
- **Architecture & Workflow Explanations**: Visual breakdown of chunking, Ollama/HF embedding, FAISS indexing, and metrics generation.
- **CLI & Parameter Reference**: Detailed usage commands and flags.

---

## Local Development

```bash
cd website
npm install
npm run dev
```

Visit [http://localhost:3000](http://localhost:3000) in your browser.

## Deploying to Vercel

1. Go to [vercel.com/new](https://vercel.com/new).
2. Choose your repository: `Ajlikeadj/embedbench`.
3. In the project settings, set:
   - **Framework Preset**: `Next.js`
   - **Root Directory**: `website`
4. Click **Deploy**.

Vercel will automatically build the site and deploy updates whenever you push commits to GitHub.
