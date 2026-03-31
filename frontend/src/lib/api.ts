const BASE_URL = import.meta.env.VITE_API_URL || "/api/v1";

class ApiClient {
  private async request<T>(path: string, options?: RequestInit): Promise<T> {
    const response = await fetch(`${BASE_URL}${path}`, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...options?.headers,
      },
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: "Unknown error" }));
      throw new Error(error.detail || `API Error: ${response.status}`);
    }

    if (response.status === 204) {
      return undefined as T;
    }

    return response.json();
  }

  get<T>(path: string) {
    return this.request<T>(path, { method: "GET" });
  }

  /** Fetch a plain-text (or markdown) response as a string. */
  async getText(path: string): Promise<string> {
    const response = await fetch(`${BASE_URL}${path}`, { method: "GET" });
    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: "Unknown error" }));
      throw new Error(error.detail || `API Error: ${response.status}`);
    }
    return response.text();
  }

  post<T>(path: string, data?: unknown) {
    return this.request<T>(path, {
      method: "POST",
      body: data ? JSON.stringify(data) : undefined,
    });
  }

  put<T>(path: string, data?: unknown) {
    return this.request<T>(path, {
      method: "PUT",
      body: data ? JSON.stringify(data) : undefined,
    });
  }

  patch<T>(path: string, data: unknown) {
    return this.request<T>(path, {
      method: "PATCH",
      body: JSON.stringify(data),
    });
  }

  delete(path: string) {
    return this.request(path, { method: "DELETE" });
  }

  async downloadFile(path: string, filename: string): Promise<void> {
    const response = await fetch(`${BASE_URL}${path}`);

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: "Download failed" }));
      throw new Error(error.detail || `Download Error: ${response.status}`);
    }

    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  async uploadFile<T>(path: string, file: File, customMetadata?: {key: string, value: string}[]): Promise<T> {
    const formData = new FormData();
    formData.append("file", file);
    
    if (customMetadata && customMetadata.length > 0) {
      formData.append("custom_metadata", JSON.stringify(customMetadata));
    }

    const response = await fetch(`${BASE_URL}${path}`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: "Upload failed" }));
      throw new Error(error.detail);
    }

    return response.json();
  }
}

export const api = new ApiClient();
