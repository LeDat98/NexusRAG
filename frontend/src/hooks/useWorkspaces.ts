import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type {
  KnowledgeBase,
  CreateWorkspace,
  UpdateWorkspace,
  WorkspaceSummary,
} from "@/types";

export function useWorkspaces() {
  return useQuery({
    queryKey: ["workspaces"],
    queryFn: () => api.get<KnowledgeBase[]>("/workspaces"),
    staleTime: 30_000,
  });
}

export function useWorkspace(workspaceId: number | null) {
  return useQuery({
    queryKey: ["workspaces", workspaceId],
    queryFn: () => api.get<KnowledgeBase>(`/workspaces/${workspaceId}`),
    enabled: !!workspaceId,
  });
}

export function useWorkspaceSummaries() {
  return useQuery({
    queryKey: ["workspaces", "summary"],
    queryFn: () => api.get<WorkspaceSummary[]>("/workspaces/summary"),
  });
}

export function useCreateWorkspace() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: CreateWorkspace) =>
      api.post<KnowledgeBase>("/workspaces", data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["workspaces"] });
    },
  });
}

export function useUpdateWorkspace() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ id, data }: { id: number; data: UpdateWorkspace }) =>
      api.put<KnowledgeBase>(`/workspaces/${id}`, data),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ["workspaces"] });
      queryClient.invalidateQueries({ queryKey: ["workspaces", variables.id] });
    },
  });
}

export function useDeleteWorkspace() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: number) => api.delete(`/workspaces/${id}`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["workspaces"] });
    },
  });
}
