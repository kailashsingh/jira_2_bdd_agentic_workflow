import { Component, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Subscription } from 'rxjs';
import { MatTableModule } from '@angular/material/table';
import { MatButtonModule } from '@angular/material/button';
import { MatChipsModule } from '@angular/material/chips';
import { MatIconModule } from '@angular/material/icon';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatSelectModule } from '@angular/material/select';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatTooltipModule } from '@angular/material/tooltip';
import { MatCardModule } from '@angular/material/card';
import { BackendService, WorkflowStatus } from '../../service/backend-service';
import { JobRefreshService } from '../../service/job-refresh.service';

export interface JobStatusEntry {
  run_id: string;
  started_at: string;
  status: 'running' | 'completed' | 'failed' | 'pending';
  pull_requests: string[];
  message: string;
  sprint_id?: number;
  completed_at?: string;
  error?: string;
}

@Component({
  selector: 'app-job-status-panel',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    MatTableModule,
    MatButtonModule,
    MatChipsModule,
    MatIconModule,
    MatFormFieldModule,
    MatInputModule,
    MatSelectModule,
    MatProgressSpinnerModule,
    MatTooltipModule,
    MatCardModule
  ],
  template: `
    <div class="space-y-6">
      <!-- Controls Section -->
      <div class="px-6">
      <!-- Job Status Grid with Title -->
      <div class="px-6">
        <div class="border rounded-lg overflow-hidden">
           <!-- Title and Controls Section -->
           <div class="border-b bg-card px-6 py-4">
             <div class="flex items-center justify-between">
               <div class="flex items-center gap-2">
                 <h2 class="text-xl font-semibold">Jobs</h2>
                 <mat-icon class="text-blue-600" style="font-size: 16px;" title="View and manage workflow job statuses">info</mat-icon>
               </div>
               
               <div class="flex items-center gap-2">
                 <mat-form-field class="max-w-sm" appearance="outline">
                   <mat-icon matPrefix>search</mat-icon>
                   <input matInput placeholder="Search jobs..." [(ngModel)]="searchTerm" (input)="applyFilter()">
                 </mat-form-field>
                 
                 <mat-form-field class="w-130" appearance="outline">
                   <mat-select placeholder="Filter" [(ngModel)]="statusFilter" (selectionChange)="applyFilter()">
                     <mat-option value="all">All Jobs</mat-option>
                     <mat-option value="running">Running</mat-option>
                     <mat-option value="completed">Completed</mat-option>
                     <mat-option value="failed">Failed</mat-option>
                     <mat-option value="pending">Pending</mat-option>
                   </mat-select>
                 </mat-form-field>
                 
                 <button mat-stroked-button (click)="refreshJobs()" [disabled]="isLoading">
                   <mat-icon>refresh</mat-icon>
                   Refresh
                 </button>
               </div>
             </div>
           </div>
          <table mat-table [dataSource]="filteredDataSource" class="w-full">
            
            <!-- Run ID Column -->
            <ng-container matColumnDef="run_id">
              <th mat-header-cell *matHeaderCellDef>Run ID</th>
              <td mat-cell *matCellDef="let element">
                <div class="font-mono text-sm">{{ element.run_id }}</div>
              </td>
            </ng-container>

            <!-- Started At Column -->
            <ng-container matColumnDef="started_at">
              <th mat-header-cell *matHeaderCellDef>Started At</th>
              <td mat-cell *matCellDef="let element">
                <div class="text-sm">{{ formatDateTime(element.started_at) }}</div>
              </td>
            </ng-container>

            <!-- Status Column -->
            <ng-container matColumnDef="status">
              <th mat-header-cell *matHeaderCellDef>Status</th>
              <td mat-cell *matCellDef="let element">
                <div class="flex items-center gap-2">
                  <mat-icon [ngClass]="getStatusIconClass(element.status)">
                    {{ getStatusIcon(element.status) }}
                  </mat-icon>
                  <span class="capitalize text-sm">{{ element.status }}</span>
                  <mat-spinner *ngIf="element.status === 'running'" diameter="16"></mat-spinner>
                </div>
              </td>
            </ng-container>

            <!-- Pull Requests Column -->
            <ng-container matColumnDef="pull_requests">
              <th mat-header-cell *matHeaderCellDef>Pull Requests</th>
              <td mat-cell *matCellDef="let element">
                <mat-chip-set>
                  <mat-chip *ngFor="let pr of element.pull_requests" color="accent">
                    {{ pr }}
                  </mat-chip>
                  <mat-chip *ngIf="element.pull_requests.length === 0" disabled>
                    No PRs
                  </mat-chip>
                </mat-chip-set>
              </td>
            </ng-container>

            <!-- Message Column -->
            <ng-container matColumnDef="message">
              <th mat-header-cell *matHeaderCellDef>Message</th>
              <td mat-cell *matCellDef="let element">
                <div class="text-sm max-w-xs truncate" [title]="element.message">
                  {{ element.message || 'No message' }}
                </div>
              </td>
            </ng-container>

            <!-- Action Column -->
            <ng-container matColumnDef="action">
              <th mat-header-cell *matHeaderCellDef class="w-100">Action</th>
              <td mat-cell *matCellDef="let element">
                <button mat-icon-button 
                        (click)="retriggerJob(element)" 
                        [disabled]="element.status === 'running'"
                        matTooltip="Retrigger Job">
                  <mat-icon>refresh</mat-icon>
                </button>
              </td>
            </ng-container>

            <tr mat-header-row *matHeaderRowDef="displayedColumns"></tr>
            <tr mat-row *matRowDef="let row; columns: displayedColumns;"></tr>
          </table>
        </div>

        <!-- Loading State -->
        <div *ngIf="isLoading" class="flex justify-center py-8">
          <mat-spinner></mat-spinner>
        </div>

        <!-- Empty State -->
        <div *ngIf="!isLoading && filteredDataSource.length === 0" class="text-center py-8 text-muted-foreground">
          <mat-icon class="text-4xl mb-2">inbox</mat-icon>
          <p>No jobs found</p>
        </div>
        </div>
      </div>
    </div>
  `,
  styles: [`
    .border { border: 1px solid var(--border, rgba(0, 0, 0, 0.1)); }
    .rounded-lg { border-radius: 0.5rem; }
    .overflow-hidden { overflow: hidden; }
    .w-full { width: 100%; }
    .w-100 { width: 100px; }
    .flex { display: flex; }
    .items-center { align-items: center; }
    .justify-center { justify-content: center; }
    .justify-between { justify-content: space-between; }
    .gap-2 { gap: 0.5rem; }
    .gap-4 { gap: 1rem; }
    .space-y-6 > * + * { margin-top: 1.5rem; }
    .text-4xl { font-size: 2.25rem; line-height: 2.5rem; }
    .text-sm { font-size: 0.875rem; }
    .text-muted-foreground { color: var(--muted-foreground, #717182); }
    .px-6 { padding-left: 1.5rem; padding-right: 1.5rem; }
    .py-8 { padding-top: 2rem; padding-bottom: 2rem; }
    .mb-2 { margin-bottom: 0.5rem; }
    .capitalize { text-transform: capitalize; }
    .font-mono { font-family: ui-monospace, SFMono-Regular, "SF Mono", Consolas, "Liberation Mono", Menlo, monospace; }
    .max-w-sm { max-width: 24rem; }
    .max-w-xs { max-width: 20rem; }
    .w-130 { width: 130px; }
    .truncate { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .flex-1 { flex: 1 1 0%; }
    .flex-col { flex-direction: column; }
    .items-start { align-items: flex-start; }
    .text-center { text-align: center; }
    
    .text-green-600 { color: #059669; }
    .text-yellow-600 { color: #d97706; }
    .text-red-600 { color: #dc2626; }
    .text-blue-600 { color: #2563eb; }
    
    .flex { display: flex; }
    .items-center { align-items: center; }
    .gap-2 { gap: 0.5rem; }
    .text-xl { font-size: 1.25rem; line-height: 1.75rem; }
    .font-semibold { font-weight: 600; }
    .border-b { border-bottom: 1px solid var(--border, rgba(0, 0, 0, 0.1)); }
    .bg-card { background-color: var(--card, #ffffff); }
    .px-6 { padding-left: 1.5rem; padding-right: 1.5rem; }
    .py-4 { padding-top: 1rem; padding-bottom: 1rem; }
    
    mat-icon { margin-right: 0.5rem; }
    mat-icon.h-4 { font-size: 1rem; height: 1rem; width: 1rem; }
    
    @media (min-width: 640px) {
      .sm\\:flex-row { flex-direction: row; }
      .sm\\:items-center { align-items: center; }
    }
  `]
})
export class JobStatusPanelComponent implements OnInit, OnDestroy {
  displayedColumns: string[] = ['run_id', 'started_at', 'status', 'pull_requests', 'message', 'action'];
  
  dataSource: JobStatusEntry[] = [];
  filteredDataSource: JobStatusEntry[] = [];
  searchTerm: string = '';
  statusFilter: string = 'all';
  isLoading: boolean = false;
  private refreshSubscription?: Subscription;

  constructor(
    private backendService: BackendService,
    private jobRefreshService: JobRefreshService
  ) {}

  ngOnInit() {
    this.loadJobs();
    // Subscribe to refresh triggers from other components
    this.refreshSubscription = this.jobRefreshService.refreshTriggered$.subscribe(() => {
      console.log('Refresh triggered from external component');
      this.loadJobs();
    });
  }

  ngOnDestroy() {
    // Clean up subscription to prevent memory leaks
    if (this.refreshSubscription) {
      this.refreshSubscription.unsubscribe();
    }
  }

  loadJobs() {
    this.isLoading = true;
    this.backendService.getAllRuns().subscribe({
      next: (runs: WorkflowStatus[]) => {
        console.log('Received runs from backend:', runs);
        this.dataSource = runs.map((run, index) => this.mapWorkflowStatusToJobEntry(run, index));
        this.applyFilter();
        this.isLoading = false;
      },
      error: (error) => {
        console.error('Error loading jobs:', error);
        this.isLoading = false;
        // Fallback to mock data for development
        console.log('Falling back to mock data');
        //this.loadMockData();
      }
    });
  }

  loadMockData() {
    this.dataSource = [
      {
        run_id: 'run-001-2024-01-15',
        started_at: '2024-01-15T09:30:00Z',
        status: 'completed',
        pull_requests: ['PR-001', 'PR-002'],
        message: 'Workflow completed successfully',
        sprint_id: 123
      },
      {
        run_id: 'run-002-2024-01-15',
        started_at: '2024-01-15T14:15:00Z',
        status: 'running',
        pull_requests: ['PR-003'],
        message: 'Processing feature generation...',
        sprint_id: 124
      },
      {
        run_id: 'run-003-2024-01-14',
        started_at: '2024-01-14T11:45:00Z',
        status: 'failed',
        pull_requests: [],
        message: 'Error: Failed to generate BDD scenarios',
        error: 'Connection timeout to external service'
      },
      {
        run_id: 'run-004-2024-01-14',
        started_at: '2024-01-14T16:20:00Z',
        status: 'completed',
        pull_requests: ['PR-004', 'PR-005', 'PR-006'],
        message: 'All features generated and validated',
        sprint_id: 125
      },
      {
        run_id: 'run-005-2024-01-13',
        started_at: '2024-01-13T13:10:00Z',
        status: 'pending',
        pull_requests: ['PR-007'],
        message: 'Waiting for approval...'
      }
    ];
    this.applyFilter();
  }

  mapWorkflowStatusToJobEntry(run: WorkflowStatus, index: number): JobStatusEntry {
    // Generate a run_id if not provided by backend
    const runId = run.status || `run-${index + 1}-${new Date().toISOString().split('T')[0]}`;
    
    return {
      run_id: runId,
      started_at: run.started_at || new Date().toISOString(),
      status: this.mapStatus(run.status),
      pull_requests: [], // This would need to be extracted from the actual data
      message: run.error || 'Workflow in progress',
      sprint_id: run.sprint_id,
      completed_at: run.completed_at,
      error: run.error
    };
  }

  mapStatus(status: string): 'running' | 'completed' | 'failed' | 'pending' {
    if (!status) return 'pending';
    
    switch (status.toLowerCase()) {
      case 'completed':
      case 'success':
      case 'finished':
        return 'completed';
      case 'failed':
      case 'error':
      case 'cancelled':
        return 'failed';
      case 'running':
      case 'in_progress':
      case 'processing':
        return 'running';
      case 'pending':
      case 'queued':
      case 'waiting':
        return 'pending';
      default:
        console.warn('Unknown status:', status);
        return 'pending';
    }
  }

  applyFilter() {
    let filtered = [...this.dataSource];

    // Apply search filter
    if (this.searchTerm) {
      const searchLower = this.searchTerm.toLowerCase();
      filtered = filtered.filter(job => 
        job.run_id.toLowerCase().includes(searchLower) ||
        job.message.toLowerCase().includes(searchLower) ||
        job.pull_requests.some(pr => pr.toLowerCase().includes(searchLower))
      );
    }

    // Apply status filter
    if (this.statusFilter !== 'all') {
      filtered = filtered.filter(job => job.status === this.statusFilter);
    }

    this.filteredDataSource = filtered;
  }

  refreshJobs() {
    console.log('Refreshing jobs...');
    this.loadJobs();
  }

  triggerNewJob() {
    // This would trigger a new workflow job
    console.log('Triggering new job...');
    // Implementation would depend on your specific requirements
  }

  retriggerJob(job: JobStatusEntry) {
    console.log('Retriggering job:', job.run_id);
    // Implementation would depend on your specific requirements
    // This could call the backend service to retrigger the specific job
  }

  getStatusIcon(status: string): string {
    switch (status) {
      case 'completed':
        return 'check_circle';
      case 'running':
        return 'play_circle';
      case 'failed':
        return 'error';
      case 'pending':
        return 'schedule';
      default:
        return 'help';
    }
  }

  getStatusIconClass(status: string): string {
    switch (status) {
      case 'completed':
        return 'text-green-600';
      case 'running':
        return 'text-blue-600';
      case 'failed':
        return 'text-red-600';
      case 'pending':
        return 'text-yellow-600';
      default:
        return '';
    }
  }

  formatDateTime(dateString: string): string {
    const date = new Date(dateString);
    return date.toLocaleString();
  }
}
