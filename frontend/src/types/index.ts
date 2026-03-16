// Knowledge Base (Document Workspace)
export interface KnowledgeBase {
  id: number;
  name: string;
  description: string | null;
  system_prompt: string | null;
  document_count: number;
  indexed_count: number;
  created_at: string;
  updated_at: string;
}

export interface CreateWorkspace {
  name: string;
  description?: string;
}

export interface UpdateWorkspace {
  name?: string;
  description?: string;
  system_prompt?: string | null;
}

export interface WorkspaceSummary {
  id: number;
  name: string;
  document_count: number;
}

export interface Document {
  id: number;
  workspace_id: number;
  filename: string;
  original_filename: string;
  file_type: string;
  file_size: number;
  status: DocumentStatus;
  chunk_count: number;
  error_message: string | null;
  created_at: string;
  updated_at: string;
  page_count?: number;
  image_count?: number;
  table_count?: number;
  parser_version?: string;
  processing_time_ms?: number;
}

// RAG Types
export type DocumentStatus = "pending" | "parsing" | "indexing" | "processing" | "indexed" | "failed";

export type RAGQueryMode = "hybrid" | "vector_only" | "naive" | "local" | "global";

export interface RAGQueryRequest {
  question: string;
  top_k?: number;
  document_ids?: number[];
  mode?: RAGQueryMode;
}

export interface Citation {
  source_file: string;
  document_id: number | null;
  page_no: number | null;
  heading_path: string[];
  formatted: string;
}

export interface DocumentImage {
  image_id: string;
  document_id: number;
  page_no: number;
  caption: string;
  width: number;
  height: number;
  url: string;
}

export interface RetrievedChunk {
  content: string;
  chunk_id: string;
  score: number;
  metadata: Record<string, unknown>;
  citation?: Citation;
}

export interface RAGQueryResponse {
  query: string;
  chunks: RetrievedChunk[];
  context: string;
  total_chunks: number;
  knowledge_graph_summary?: string;
  citations?: Citation[];
  image_refs?: DocumentImage[];
}

export interface RAGStats {
  workspace_id: number;
  total_documents: number;
  indexed_documents: number;
  total_chunks: number;
  image_count?: number;
  nexusrag_documents?: number;
}

// Knowledge Graph Types
export interface KGEntity {
  name: string;
  entity_type: string;
  description: string;
  degree: number;
}

export interface KGRelationship {
  source: string;
  target: string;
  description: string;
  keywords: string;
  weight: number;
}

export interface KGGraphNode {
  id: string;
  label: string;
  entity_type: string;
  degree: number;
}

export interface KGGraphEdge {
  source: string;
  target: string;
  label: string;
  weight: number;
}

export interface KGGraphData {
  nodes: KGGraphNode[];
  edges: KGGraphEdge[];
  is_truncated: boolean;
}

export interface KGAnalytics {
  entity_count: number;
  relationship_count: number;
  entity_types: Record<string, number>;
  top_entities: KGEntity[];
  avg_degree: number;
}

export interface DocumentBreakdown {
  document_id: number;
  filename: string;
  chunk_count: number;
  image_count: number;
  page_count: number;
  file_size: number;
  status: string;
}

export interface ProjectAnalytics {
  stats: RAGStats;
  kg_analytics: KGAnalytics | null;
  document_breakdown: DocumentBreakdown[];
}

// Chat Types
export interface ChatImageRef {
  ref_id?: string;  // 4-char alphanumeric ID, e.g. "p4f2"
  image_id: string;
  document_id: number;
  page_no: number;
  caption: string;
  url: string;
  width: number;
  height: number;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: ChatSourceChunk[];
  relatedEntities?: string[];
  imageRefs?: ChatImageRef[];
  thinking?: string | null;
  timestamp: string;
  isStreaming?: boolean;
  agentSteps?: AgentStep[];
}

export interface ChatSourceChunk {
  index: number | string;  // number for legacy, string for new [a3x9] format
  chunk_id: string;
  content: string;
  document_id: number;
  page_no: number;
  heading_path: string[];
  score: number;
  source_type?: "vector" | "kg";
}

export interface ChatResponseData {
  answer: string;
  sources: ChatSourceChunk[];
  related_entities: string[];
  kg_summary: string | null;
  image_refs: ChatImageRef[];
  thinking: string | null;
}

export interface PersistedChatMessage {
  id: number;
  message_id: string;
  role: "user" | "assistant";
  content: string;
  sources?: ChatSourceChunk[] | null;
  related_entities?: string[] | null;
  image_refs?: ChatImageRef[] | null;
  thinking?: string | null;
  agent_steps?: AgentStep[] | null;
  created_at: string;
}

export interface ChatHistoryResponse {
  workspace_id: number;
  messages: PersistedChatMessage[];
  total: number;
}

export interface LLMCapabilities {
  provider: string;
  model: string;
  supports_thinking: boolean;
  supports_vision: boolean;
  thinking_default: boolean;
}

// SSE Streaming Types
export type ChatStreamStatus = "idle" | "analyzing" | "retrieving" | "generating" | "error";

// Agent Step Types (ThinkingTimeline)
export type AgentStepType =
  | "analyzing"
  | "understood"
  | "retrieving"
  | "sources_found"
  | "generating"
  | "done"
  | "error";

export type AgentStepStatus = "active" | "completed" | "error";

export interface AgentStep {
  id: string;
  step: AgentStepType;
  detail: string;
  status: AgentStepStatus;
  timestamp: number;
  durationMs?: number;
  thinkingText?: string;
  sourceBadges?: string[];
  sourceCount?: number;
  imageCount?: number;
}

// -----------------------------------------------------------------------
// JobNexus Types
// -----------------------------------------------------------------------

export type CandidateStatus = "active" | "inactive";
export type CVStatus = "pending" | "parsing" | "indexing" | "indexed" | "failed";
export type JobStatus = "open" | "closed" | "draft";
export type EmploymentType = "full_time" | "part_time" | "contract" | "internship" | "remote";
export type MatchStatus = "pending" | "accepted" | "rejected";

export interface Candidate {
  id: number;
  name: string;
  email: string | null;
  phone: string | null;
  location: string | null;
  desired_role: string | null;
  desired_salary_min: number | null;
  desired_salary_max: number | null;
  experience_years: number | null;
  education_level: string | null;
  status: CandidateStatus;
  created_at: string;
  updated_at: string;
}

export interface CandidateCV {
  id: number;
  candidate_id: number;
  original_filename: string;
  file_type: string;
  file_size: number;
  status: CVStatus;
  chunk_count: number;
  page_count: number;
  skills_extracted: string[] | null;
  experience_extracted: unknown[] | null;
  education_extracted: unknown[] | null;
  summary_extracted: string | null;
  processing_time_ms: number;
  error_message: string | null;
  created_at: string;
}

export interface Company {
  id: number;
  name: string;
  industry: string | null;
  location: string | null;
  size: string | null;
  description: string | null;
  created_at: string;
}

export interface JobPosting {
  id: number;
  company_id: number;
  title: string;
  description_text: string | null;
  location: string | null;
  salary_min: number | null;
  salary_max: number | null;
  experience_required: number | null;
  skills_required: string[] | null;
  skills_nice_to_have: string[] | null;
  employment_type: EmploymentType;
  status: JobStatus;
  chunk_count: number;
  processing_time_ms: number;
  original_filename: string | null;
  created_at: string;
  company?: Company;
  markdown_content?: string | null;
}

export interface ScoreBreakdown {
  semantic_score: number;
  skill_match_score: number;
  experience_score: number;
  location_score: number;
  salary_score: number;
}

export interface MatchResult {
  id: number;
  candidate_id: number;
  job_id: number;
  overall_score: number;
  semantic_score: number;
  skill_match_score: number;
  experience_score: number;
  location_score: number;
  salary_score: number;
  matched_skills: string[] | null;
  missing_skills: string[] | null;
  explanation: string | null;
  status: MatchStatus;
}
