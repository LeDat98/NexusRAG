import { useState, useCallback } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { motion, AnimatePresence } from "framer-motion";
import {
  Briefcase,
  Star,
  Loader2,
  ChevronRight,
  User,
  MapPin,
  DollarSign,
  Zap,
} from "lucide-react";
import { toast } from "sonner";
import { cn } from "@/lib/utils";
import { api } from "@/lib/api";
import type { Candidate, CandidateCV, MatchResult } from "@/types";

// ---------------------------------------------------------------------------
// API hooks
// ---------------------------------------------------------------------------

function useCandidates() {
  return useQuery({
    queryKey: ["candidates"],
    queryFn: () => api.get<Candidate[]>("/candidates"),
  });
}

function useCandidateDetail(id: number | null) {
  return useQuery({
    queryKey: ["candidate", id],
    queryFn: () =>
      api.get<Candidate & { cvs: CandidateCV[] }>(`/candidates/${id}`),
    enabled: !!id,
  });
}

// ---------------------------------------------------------------------------
// Score bar component
// ---------------------------------------------------------------------------

function ScoreBar({ score, label }: { score: number; label: string }) {
  const pct = Math.round(score * 100);
  const color =
    pct >= 70
      ? "bg-emerald-500"
      : pct >= 40
        ? "bg-amber-500"
        : "bg-red-400";
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="w-20 text-muted-foreground truncate">{label}</span>
      <div className="flex-1 h-1.5 rounded-full bg-muted overflow-hidden">
        <motion.div
          className={cn("h-full rounded-full", color)}
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.6 }}
        />
      </div>
      <span className="w-8 text-right font-medium">{pct}%</span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Match Card
// ---------------------------------------------------------------------------

function MatchCard({ match }: { match: MatchResult }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <motion.div
      layout
      className="rounded-lg border bg-card p-4 hover:shadow-md transition-shadow cursor-pointer"
      onClick={() => setExpanded(!expanded)}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <Briefcase className="w-4 h-4 text-primary flex-shrink-0" />
            <span className="font-semibold text-sm truncate">
              Job #{match.job_id}
            </span>
          </div>
          <div className="mt-1 text-xs text-muted-foreground">
            {match.explanation}
          </div>
        </div>
        <div className="flex flex-col items-end gap-1">
          <span
            className={cn(
              "text-lg font-bold",
              match.overall_score >= 0.7
                ? "text-emerald-500"
                : match.overall_score >= 0.4
                  ? "text-amber-500"
                  : "text-red-400"
            )}
          >
            {Math.round(match.overall_score * 100)}%
          </span>
          <ChevronRight
            className={cn(
              "w-4 h-4 transition-transform",
              expanded && "rotate-90"
            )}
          />
        </div>
      </div>

      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="overflow-hidden"
          >
            <div className="mt-3 pt-3 border-t space-y-1.5">
              <ScoreBar score={match.semantic_score} label="Semantic" />
              <ScoreBar score={match.skill_match_score} label="Skills" />
              <ScoreBar score={match.experience_score} label="Experience" />
              <ScoreBar score={match.location_score} label="Location" />
              <ScoreBar score={match.salary_score} label="Salary" />
            </div>
            {match.matched_skills && match.matched_skills.length > 0 && (
              <div className="mt-2 flex flex-wrap gap-1">
                {match.matched_skills.map((s) => (
                  <span
                    key={s}
                    className="px-1.5 py-0.5 text-[10px] rounded bg-emerald-400/15 text-emerald-600 dark:text-emerald-400"
                  >
                    {s}
                  </span>
                ))}
              </div>
            )}
            {match.missing_skills && match.missing_skills.length > 0 && (
              <div className="mt-1 flex flex-wrap gap-1">
                {match.missing_skills.map((s) => (
                  <span
                    key={s}
                    className="px-1.5 py-0.5 text-[10px] rounded bg-red-400/15 text-red-600 dark:text-red-400"
                  >
                    {s}
                  </span>
                ))}
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

// ---------------------------------------------------------------------------
// Main Dashboard
// ---------------------------------------------------------------------------

export function CandidateDashboard() {
  const qc = useQueryClient();
  const { data: candidates, isLoading: loadingCandidates } = useCandidates();
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const { data: detail } = useCandidateDetail(selectedId);
  const [matches, setMatches] = useState<MatchResult[]>([]);
  const [matchLoading, setMatchLoading] = useState(false);

  const createCandidate = useMutation({
    mutationFn: (data: { name: string; email?: string }) =>
      api.post<Candidate>("/candidates", data),
    onSuccess: (c) => {
      qc.invalidateQueries({ queryKey: ["candidates"] });
      setSelectedId(c.id);
      toast.success(`Created candidate: ${c.name}`);
    },
  });

  const uploadCV = useMutation({
    mutationFn: (file: File) =>
      api.uploadFile(`/candidates/${selectedId}/upload-cv`, file),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["candidate", selectedId] });
      toast.success("CV uploaded");
    },
  });

  const processCV = useMutation({
    mutationFn: (cvId: number) =>
      api.post(`/candidates/${selectedId}/process/${cvId}`, {}),
    onSuccess: () => {
      toast.success("CV processing started");
      setTimeout(
        () => qc.invalidateQueries({ queryKey: ["candidate", selectedId] }),
        3000
      );
    },
  });

  const handleFindJobs = useCallback(async () => {
    if (!selectedId) return;
    setMatchLoading(true);
    try {
      const res = await api.get<{
        candidate_id: number;
        total: number;
        matches: MatchResult[];
      }>(`/candidates/${selectedId}/recommendations?top_k=10`);
      setMatches(res.matches);
    } catch (e: unknown) {
      toast.error(e instanceof Error ? e.message : "Failed to find jobs");
    } finally {
      setMatchLoading(false);
    }
  }, [selectedId]);

  const handleQuickCreate = () => {
    const name = prompt("Candidate name:");
    if (name) createCandidate.mutate({ name });
  };

  return (
    <div className="h-full overflow-hidden grid grid-cols-[280px_1fr_1fr] gap-0">
      {/* Column 1: Candidate List */}
      <div className="border-r flex flex-col overflow-hidden">
        <div className="p-3 border-b flex items-center justify-between">
          <h2 className="font-semibold text-sm flex items-center gap-1.5">
            <User className="w-4 h-4" /> Candidates
          </h2>
          <button
            onClick={handleQuickCreate}
            className="text-xs px-2 py-1 rounded bg-primary text-primary-foreground hover:bg-primary/90"
          >
            + New
          </button>
        </div>
        <div className="flex-1 overflow-y-auto p-2 space-y-1">
          {loadingCandidates && (
            <div className="flex justify-center py-8">
              <Loader2 className="w-5 h-5 animate-spin text-muted-foreground" />
            </div>
          )}
          {candidates?.map((c) => (
            <button
              key={c.id}
              onClick={() => {
                setSelectedId(c.id);
                setMatches([]);
              }}
              className={cn(
                "w-full text-left p-2 rounded-md text-sm transition-colors",
                selectedId === c.id
                  ? "bg-primary/10 text-primary"
                  : "hover:bg-muted"
              )}
            >
              <div className="font-medium truncate">{c.name}</div>
              <div className="text-xs text-muted-foreground truncate">
                {c.desired_role || c.email || "No details"}
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Column 2: Candidate Detail + CV */}
      <div className="border-r flex flex-col overflow-hidden">
        {detail ? (
          <>
            <div className="p-4 border-b space-y-2">
              <h3 className="font-semibold">{detail.name}</h3>
              {detail.desired_role && (
                <div className="flex items-center gap-1 text-xs text-muted-foreground">
                  <Briefcase className="w-3 h-3" /> {detail.desired_role}
                </div>
              )}
              {detail.location && (
                <div className="flex items-center gap-1 text-xs text-muted-foreground">
                  <MapPin className="w-3 h-3" /> {detail.location}
                </div>
              )}
              {(detail.desired_salary_min || detail.desired_salary_max) && (
                <div className="flex items-center gap-1 text-xs text-muted-foreground">
                  <DollarSign className="w-3 h-3" />
                  {detail.desired_salary_min}–{detail.desired_salary_max}
                </div>
              )}
            </div>

            <div className="p-3 border-b">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-medium">CVs</span>
                <label className="text-xs px-2 py-1 rounded bg-primary/10 text-primary cursor-pointer hover:bg-primary/20">
                  Upload CV
                  <input
                    type="file"
                    className="hidden"
                    accept=".pdf,.docx,.txt,.md"
                    onChange={(e) => {
                      const f = e.target.files?.[0];
                      if (f) uploadCV.mutate(f);
                    }}
                  />
                </label>
              </div>
              {detail.cvs?.map((cv: CandidateCV) => (
                <div
                  key={cv.id}
                  className="p-2 rounded border text-xs mb-1 flex items-center justify-between"
                >
                  <div>
                    <div className="font-medium">{cv.original_filename}</div>
                    <div className="text-muted-foreground">
                      {cv.status} · {cv.chunk_count} chunks
                    </div>
                    {cv.skills_extracted && cv.skills_extracted.length > 0 && (
                      <div className="flex flex-wrap gap-1 mt-1">
                        {cv.skills_extracted.slice(0, 8).map((s: string) => (
                          <span
                            key={s}
                            className="px-1 py-0.5 rounded bg-primary/10 text-[10px]"
                          >
                            {s}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                  {cv.status === "pending" && (
                    <button
                      onClick={() => processCV.mutate(cv.id)}
                      className="text-[10px] px-2 py-1 rounded bg-primary text-primary-foreground"
                    >
                      Process
                    </button>
                  )}
                </div>
              ))}
            </div>

            <div className="p-3 flex-1 flex flex-col items-center justify-center">
              <button
                onClick={handleFindJobs}
                disabled={matchLoading || !detail.cvs?.some((cv: CandidateCV) => cv.status === "indexed")}
                className="flex items-center gap-2 px-4 py-2 rounded-lg bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {matchLoading ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Zap className="w-4 h-4" />
                )}
                Find Matching Jobs
              </button>
              {!detail.cvs?.some((cv: CandidateCV) => cv.status === "indexed") && (
                <p className="text-xs text-muted-foreground mt-2">
                  Upload and process a CV first
                </p>
              )}
            </div>
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center text-muted-foreground text-sm">
            Select a candidate
          </div>
        )}
      </div>

      {/* Column 3: Recommendations */}
      <div className="flex flex-col overflow-hidden">
        <div className="p-3 border-b">
          <h3 className="font-semibold text-sm flex items-center gap-1.5">
            <Star className="w-4 h-4" /> Job Recommendations
          </h3>
        </div>
        <div className="flex-1 overflow-y-auto p-3 space-y-2">
          {matches.length === 0 && !matchLoading && (
            <div className="flex flex-col items-center justify-center h-full text-muted-foreground text-sm">
              <Briefcase className="w-8 h-8 mb-2 opacity-30" />
              Click "Find Matching Jobs" to get recommendations
            </div>
          )}
          {matchLoading && (
            <div className="flex justify-center py-8">
              <Loader2 className="w-6 h-6 animate-spin text-primary" />
            </div>
          )}
          {matches.map((m) => (
            <MatchCard key={m.id} match={m} />
          ))}
        </div>
      </div>
    </div>
  );
}
