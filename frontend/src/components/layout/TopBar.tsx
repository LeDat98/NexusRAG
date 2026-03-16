import { memo, useEffect, useState } from "react";
import { useLocation } from "react-router-dom";
import { ChevronRight, Cpu, Database } from "lucide-react";
import { cn } from "@/lib/utils";
import { api } from "@/lib/api";

interface ConfigStatus {
  llm_provider: string;
  llm_model: string;
  kg_embedding_provider: string;
  kg_embedding_model: string;
  kg_embedding_dimension: number;
  nexusrag_embedding_model: string;
  nexusrag_reranker_model: string;
}

interface TopBarProps {
  actions?: React.ReactNode;
  className?: string;
}

export const TopBar = memo(function TopBar({ actions, className }: TopBarProps) {
  const location = useLocation();
  const [config, setConfig] = useState<ConfigStatus | null>(null);

  useEffect(() => {
    api.get<ConfigStatus>("/config/status").then(setConfig).catch(() => {});
  }, []);

  const segments: { label: string; active: boolean }[] = [
    { label: "JobNexus", active: false },
  ];

  if (location.pathname === "/") {
    segments.push({ label: "Home", active: true });
  } else if (location.pathname === "/candidate") {
    segments.push({ label: "Find Jobs", active: true });
  } else if (location.pathname === "/recruiter") {
    segments.push({ label: "Find Talent", active: true });
  } else if (location.pathname === "/chat") {
    segments.push({ label: "Chat", active: true });
  } else if (location.pathname.startsWith("/knowledge-bases/")) {
    segments.push({ label: "Workspace", active: true });
  }

  return (
    <div
      className={cn(
        "h-12 flex items-center justify-between px-4 border-b border-border flex-shrink-0 bg-background",
        className
      )}
    >
      {/* Breadcrumbs */}
      <div className="flex items-center gap-1.5 text-sm min-w-0">
        {segments.map((seg, i) => (
          <div key={i} className="flex items-center gap-1.5 min-w-0">
            {i > 0 && <ChevronRight className="w-3.5 h-3.5 text-muted-foreground flex-shrink-0" />}
            <span
              className={cn(
                "truncate",
                seg.active ? "font-medium text-foreground" : "text-muted-foreground"
              )}
            >
              {seg.label}
            </span>
          </div>
        ))}
      </div>

      {/* Right-side: model badges + actions */}
      <div className="flex items-center gap-2 flex-shrink-0">
        {config && (
          <div className="flex items-center gap-1.5">
            <div
              className={cn(
                "flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium",
                config.llm_provider === "ollama"
                  ? "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400"
                  : "bg-blue-500/10 text-blue-600 dark:text-blue-400"
              )}
              title={`LLM: ${config.llm_provider} / ${config.llm_model}`}
            >
              <Cpu className="w-3 h-3" />
              <span>{config.llm_model}</span>
            </div>
            <div
              className="flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-purple-500/10 text-purple-600 dark:text-purple-400"
              title={`KG Embedding: ${config.kg_embedding_provider} / ${config.kg_embedding_model} (${config.kg_embedding_dimension}d)`}
            >
              <Database className="w-3 h-3" />
              <span>{config.kg_embedding_model}</span>
            </div>
          </div>
        )}
        {actions}
      </div>
    </div>
  );
});
