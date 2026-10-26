import React from "react";
import { Server, Terminal, Shield, ArrowRight, ExternalLink } from "lucide-react";

export function VercelDeploymentGuide() {
  return (
    <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 sm:p-8 backdrop-blur-md">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-10 h-10 rounded-xl bg-black border border-slate-700 flex items-center justify-center font-bold text-white shadow-lg">
          ▲
        </div>
        <div>
          <h3 className="text-xl font-bold text-white">How to Deploy This Website to Vercel</h3>
          <p className="text-slate-400 text-sm">Step-by-step instructions to link this GitHub repository directly to Vercel.</p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-slate-950/70 border border-slate-800/80 rounded-xl p-5 relative">
          <div className="w-7 h-7 rounded-full bg-indigo-500/20 text-indigo-400 flex items-center justify-center text-xs font-bold mb-3 border border-indigo-500/30">
            1
          </div>
          <h4 className="font-semibold text-slate-200 text-sm mb-2">Push Repo to GitHub</h4>
          <p className="text-slate-400 text-xs leading-relaxed mb-3">
            Commit the newly added <code className="text-indigo-300">website/</code> directory and push to your GitHub repository:
          </p>
          <pre className="p-2.5 bg-slate-900 rounded-lg text-indigo-300 font-mono text-[11px] overflow-x-auto">
            <code>{`git add website\ngit commit -m "Add docs site"\ngit push origin main`}</code>
          </pre>
        </div>

        <div className="bg-slate-950/70 border border-slate-800/80 rounded-xl p-5 relative">
          <div className="w-7 h-7 rounded-full bg-indigo-500/20 text-indigo-400 flex items-center justify-center text-xs font-bold mb-3 border border-indigo-500/30">
            2
          </div>
          <h4 className="font-semibold text-slate-200 text-sm mb-2">Import Project on Vercel</h4>
          <p className="text-slate-400 text-xs leading-relaxed mb-3">
            Go to <a href="https://vercel.com/new" target="_blank" rel="noreferrer" className="text-indigo-400 hover:underline inline-flex items-center gap-0.5">vercel.com/new <ExternalLink className="w-3 h-3" /></a> and select <strong>Ajlikeadj/embedbench</strong> from your GitHub repositories.
          </p>
          <div className="text-xs text-amber-300/90 bg-amber-500/10 border border-amber-500/20 p-2.5 rounded-lg">
            <strong>Key Setting:</strong> Set <strong>Root Directory</strong> to <code className="bg-amber-950/40 px-1 py-0.5 rounded">website</code>.
          </div>
        </div>

        <div className="bg-slate-950/70 border border-slate-800/80 rounded-xl p-5 relative">
          <div className="w-7 h-7 rounded-full bg-indigo-500/20 text-indigo-400 flex items-center justify-center text-xs font-bold mb-3 border border-indigo-500/30">
            3
          </div>
          <h4 className="font-semibold text-slate-200 text-sm mb-2">Automatic Deployments</h4>
          <p className="text-slate-400 text-xs leading-relaxed mb-3">
            Click <strong>Deploy</strong>. Vercel automatically runs <code className="text-indigo-300">npm run build</code> and serves your fast static and dynamic docs worldwide with instant preview links on every pull request.
          </p>
          <div className="text-[11px] text-emerald-400 flex items-center gap-1.5 font-medium">
            ✓ Zero server configuration required
          </div>
        </div>
      </div>
    </div>
  );
}
