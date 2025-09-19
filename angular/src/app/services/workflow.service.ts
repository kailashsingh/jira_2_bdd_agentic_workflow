import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface WorkflowRequest {
  sprint_id?: number;
  jira_keys?: string[];
}

export interface WorkflowResponse {
  status: string;
  run_id: string;
  started_at: string;
  message: string;
}

export interface WorkflowStatus {
  status: string;
  started_at: string;
  sprint_id?: number;
  completed_at?: string;
  result?: any;
  error?: string;
}

@Injectable({
  providedIn: 'root'
})
export class WorkflowService {
  private baseUrl = 'http://localhost:8000';

  constructor(private http: HttpClient) { }

  triggerWorkflow(request: WorkflowRequest): Observable<WorkflowResponse> {
    return this.http.post<WorkflowResponse>(`${this.baseUrl}/workflow/trigger`, request);
  }

  getWorkflowStatus(runId: string): Observable<WorkflowStatus> {
    return this.http.get<WorkflowStatus>(`${this.baseUrl}/workflow/status/${runId}`);
  }

  getAllRuns(): Observable<WorkflowStatus[]> {
    return this.http.get<WorkflowStatus[]>(`${this.baseUrl}/workflow/runs`);
  }

  healthCheck(): Observable<any> {
    return this.http.get(`${this.baseUrl}/health`);
  }
}
