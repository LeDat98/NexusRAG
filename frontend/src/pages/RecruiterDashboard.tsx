import { useState, useCallback } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { motion, AnimatePresence } from "framer-motion";
import {
  Building2,
  Users,
  Loader2,
  ChevronRight,
  MapPin,
  DollarSign,
  Zap,
  Star,
  Briefcase,
} from "lucide-react";
import { toast } from "sonner";
import { cn } from "@/lib/utils";
import { api } from "@/lib/api";
import type {
  Company,
  JobPosting,
  MatchResult,
} from "@/types";

// ---------------------------------------------------------------------------
// API hooks
// ---------------------------------------------------------------------------

function useCompanies() {
  return useQuery({
    queryKey: ["companies"],
    queryFn: () => api.get<Company[]>("/jobs/companies"),
  });
}

function useJobs() {
  return useQuery({
    queryKey: ["jobs"],
    queryFn: () => api.get<JobPosting[]>("/jobs"),
  });
}

function useJobDetail(id: number | null) {
  return useQuery({
    queryKey: ["job", id],
    queryFn: () => api.get<JobPosting>(`/jobs/${id}`),
    enabled: !!id,
  });
}

// ---------------------------------------------------------------------------
// Score bar
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
// Candidate Match Card (recruiter-facing)
// ---------------------------------------------------------------------------

function CandidateMatchCard({ match }: { match: MatchResult }) {
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
            <Users className="w-4 h-4 text-primary flex-shrink-0" />
            <span className="font-semibold text-sm">
              Candidate #{match.candidate_id}
            </span>
          </div>
          <div className="mt-1 text-xs text-muted-foreground truncate">
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
// Main Recruiter Dashboard
// ---------------------------------------------------------------------------

export function RecruiterDashboard() {
  const qc = useQueryClient();
  const { data: companies } = useCompanies();
  const { data: jobs, isLoading: loadingJobs } = useJobs();
  const [selectedJobId, setSelectedJobId] = useState<number | null>(null);
  const { data: jobDetail } = useJobDetail(selectedJobId);
  const [candidates, setCandidates] = useState<MatchResult[]>([]);
  const [matchLoading, setMatchLoading] = useState(false);

  const createCompany = useMutation({
    mutationFn: (data: { name: string }) =>
      api.post<Company>("/jobs/companies", data),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["companies"] });
      toast.success("Company created");
    },
  });

  const createJob = useMutation({
    mutationFn: (data: {
      company_id: number;
      title: string;
      description_text?: string;
      skills_required?: string[];
    }) => api.post<JobPosting>("/jobs", data),
    onSuccess: (j) => {
      qc.invalidateQueries({ queryKey: ["jobs"] });
      setSelectedJobId(j.id);
      toast.success(`Job created: ${j.title}`);
    },
  });

  const processJob = useMutation({
    mutationFn: (jobId: number) => api.post(`/jobs/${jobId}/process`, {}),
    onSuccess: () => {
      toast.success("JD processing started");
      setTimeout(
        () => qc.invalidateQueries({ queryKey: ["jobs"] }),
        3000
      );
    },
  });

  const handleFindCandidates = useCallback(async () => {
    if (!selectedJobId) return;
    setMatchLoading(true);
    try {
      const res = await api.get<{
        job_id: number;
        total: number;
        matches: MatchResult[];
      }>(`/jobs/${selectedJobId}/candidates?top_k=10`);
      setCandidates(res.matches);
    } catch (e: unknown) {
      toast.error(
        e instanceof Error ? e.message : "Failed to find candidates"
      );
    } finally {
      setMatchLoading(false);
    }
  }, [selectedJobId]);

  const handleQuickCreateJob = () => {
    if (!companies || companies.length === 0) {
      toast.error("Create a company first");
      return;
    }
    const title = prompt("Job title:");
    if (title) {
      createJob.mutate({
        company_id: companies[0].id,
        title,
      });
    }
  };

  const handleQuickCreateCompany = () => {
    const name = prompt("Company name:");
    if (name) createCompany.mutate({ name });
  };

  return (
    <div className="h-full overflow-hidden grid grid-cols-[280px_1fr_1fr] gap-0">
      {/* Column 1: Jobs List */}
      <div className="border-r flex flex-col overflow-hidden">
        <div className="p-3 border-b flex items-center justify-between">
          <h2 className="font-semibold text-sm flex items-center gap-1.5">
            <Briefcase className="w-4 h-4" /> Jobs
          </h2>
          <div className="flex gap-1">
            <button
              onClick={handleQuickCreateCompany}
              className="text-xs px-2 py-1 rounded bg-muted hover:bg-muted/80"
              title="New Company"
            >
              <Building2 className="w-3 h-3" />
            </button>
            <button
              onClick={handleQuickCreateJob}
              className="text-xs px-2 py-1 rounded bg-primary text-primary-foreground hover:bg-primary/90"
            >
              + Job
            </button>
          </div>
        </div>
        <div className="flex-1 overflow-y-auto p-2 space-y-1">
          {loadingJobs && (
            <div className="flex justify-center py-8">
              <Loader2 className="w-5 h-5 animate-spin text-muted-foreground" />
            </div>
          )}
          {jobs?.map((j) => (
            <button
              key={j.id}
              onClick={() => {
                setSelectedJobId(j.id);
                setCandidates([]);
              }}
              className={cn(
                "w-full text-left p-2 rounded-md text-sm transition-colors",
                selectedJobId === j.id
                  ? "bg-primary/10 text-primary"
                  : "hover:bg-muted"
              )}
            >
              <div className="font-medium truncate">{j.title}</div>
              <div className="text-xs text-muted-foreground truncate">
                {j.status} · {j.location || "No location"}
              </div>
              {j.skills_required && j.skills_required.length > 0 && (
                <div className="flex flex-wrap gap-1 mt-1">
                  {j.skills_required.slice(0, 4).map((s) => (
                    <span
                      key={s}
                      className="px-1 py-0.5 rounded bg-primary/10 text-[10px]"
                    >
                      {s}
                    </span>
                  ))}
                </div>
              )}
            </button>
          ))}
        </div>
      </div>

      {/* Column 2: Job Detail */}
      <div className="border-r flex flex-col overflow-hidden">
        {jobDetail ? (
          <>
            <div className="p-4 border-b space-y-2">
              <h3 className="font-semibold">{jobDetail.title}</h3>
              {jobDetail.company && (
                <div className="flex items-center gap-1 text-xs text-muted-foreground">
                  <Building2 className="w-3 h-3" /> {jobDetail.company.name}
                </div>
              )}
              {jobDetail.location && (
                <div className="flex items-center gap-1 text-xs text-muted-foreground">
                  <MapPin className="w-3 h-3" /> {jobDetail.location}
                </div>
              )}
              {(jobDetail.salary_min || jobDetail.salary_max) && (
                <div className="flex items-center gap-1 text-xs text-muted-foreground">
                  <DollarSign className="w-3 h-3" />
                  {jobDetail.salary_min}–{jobDetail.salary_max}
                </div>
              )}
              <div className="text-xs">
                <span className="text-muted-foreground">Status:</span>{" "}
                <span className="font-medium">{jobDetail.status}</span>
                {" · "}
                <span className="text-muted-foreground">Chunks:</span>{" "}
                {jobDetail.chunk_count}
              </div>
            </div>

            {jobDetail.skills_required && jobDetail.skills_required.length > 0 && (
              <div className="p-3 border-b">
                <div className="text-xs font-medium mb-1">Required Skills</div>
                <div className="flex flex-wrap gap-1">
                  {jobDetail.skills_required.map((s) => (
                    <span
                      key={s}
                      className="px-1.5 py-0.5 text-[10px] rounded bg-blue-400/15 text-blue-600 dark:text-blue-400"
                    >
                      {s}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {jobDetail.description_text && (
              <div className="p-3 border-b max-h-40 overflow-y-auto">
                <div className="text-xs font-medium mb-1">Description</div>
                <p className="text-xs text-muted-foreground whitespace-pre-wrap">
                  {jobDetail.description_text}
                </p>
              </div>
            )}

            <div className="p-3 flex flex-col gap-2">
              {jobDetail.status === "draft" && (
                <button
                  onClick={() => processJob.mutate(jobDetail.id)}
                  className="flex items-center gap-2 px-3 py-1.5 rounded bg-muted hover:bg-muted/80 text-xs"
                >
                  Process JD
                </button>
              )}
              <button
                onClick={handleFindCandidates}
                disabled={matchLoading || jobDetail.chunk_count === 0}
                className="flex items-center gap-2 px-4 py-2 rounded-lg bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {matchLoading ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Zap className="w-4 h-4" />
                )}
                Find Matching Candidates
              </button>
            </div>
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center text-muted-foreground text-sm">
            Select a job posting
          </div>
        )}
      </div>

      {/* Column 3: Candidate Recommendations */}
      <div className="flex flex-col overflow-hidden">
        <div className="p-3 border-b">
          <h3 className="font-semibold text-sm flex items-center gap-1.5">
            <Star className="w-4 h-4" /> Candidate Matches
          </h3>
        </div>
        <div className="flex-1 overflow-y-auto p-3 space-y-2">
          {candidates.length === 0 && !matchLoading && (
            <div className="flex flex-col items-center justify-center h-full text-muted-foreground text-sm">
              <Users className="w-8 h-8 mb-2 opacity-30" />
              Click "Find Matching Candidates" to discover talent
            </div>
          )}
          {matchLoading && (
            <div className="flex justify-center py-8">
              <Loader2 className="w-6 h-6 animate-spin text-primary" />
            </div>
          )}
          {candidates.map((m) => (
            <CandidateMatchCard key={m.id} match={m} />
          ))}
        </div>
      </div>
    </div>
  );
}
