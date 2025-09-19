import { Component, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatButtonModule } from '@angular/material/button';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatSelectModule } from '@angular/material/select';
import { MatIconModule } from '@angular/material/icon';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatSnackBarModule, MatSnackBar } from '@angular/material/snack-bar';
import { WorkflowService, WorkflowRequest } from '../../services/workflow.service';
import { Subscription } from 'rxjs';

@Component({
  selector: 'app-study-controls',
  standalone: true,
  imports: [
    CommonModule,
    MatButtonModule,
    MatFormFieldModule,
    MatInputModule,
    MatSelectModule,
    MatIconModule,
    MatProgressSpinnerModule,
    MatSnackBarModule
  ],
  template: `
    <div class="flex flex-col sm:flex-row gap-4 items-start sm:items-center justify-between">
      <div class="flex flex-1 gap-2 items-center">
        <mat-form-field class="flex-1 max-w-sm" appearance="outline">
          <mat-icon matPrefix>search</mat-icon>
          <input matInput placeholder="Search studies...">
        </mat-form-field>
        
        <mat-form-field class="w-130" appearance="outline">
          <mat-label>Filter</mat-label>
          <mat-select>
            <mat-option value="all">All Studies</mat-option>
            <mat-option value="role-a">Role A Only</mat-option>
            <mat-option value="role-b">Role B Only</mat-option>
            <mat-option value="active">Active Only</mat-option>
            <mat-option value="validated">Validated</mat-option>
            <mat-option value="pending">Pending</mat-option>
          </mat-select>
        </mat-form-field>
      </div>
      
      <div class="flex gap-2">
        <button mat-stroked-button>
          <mat-icon>refresh</mat-icon>
          Refresh
        </button>
        <button mat-flat-button color="primary" (click)="generateFeatureFile()" [disabled]="isGenerating">
          <mat-icon *ngIf="!isGenerating">auto_fix_high</mat-icon>
          <mat-spinner *ngIf="isGenerating" diameter="20"></mat-spinner>
          {{ isGenerating ? 'Generating...' : 'Generate Feature File' }}
        </button>
        <button mat-flat-button color="accent">
          <mat-icon>add</mat-icon>
          New Study
        </button>
      </div>
    </div>
  `,
  styles: [`
    .flex { display: flex; }
    .flex-col { flex-direction: column; }
    .flex-1 { flex: 1 1 0%; }
    .gap-2 { gap: 0.5rem; }
    .gap-4 { gap: 1rem; }
    .items-center { align-items: center; }
    .items-start { align-items: flex-start; }
    .justify-between { justify-content: space-between; }
    .max-w-sm { max-width: 24rem; }
    .w-130 { width: 130px; }
    mat-icon { margin-right: 0.5rem; }
    
    @media (min-width: 640px) {
      .sm\\:flex-row { flex-direction: row; }
      .sm\\:items-center { align-items: center; }
    }
  `]
})
export class StudyControlsComponent implements OnDestroy {
  isGenerating = false;
  currentRunId: string | null = null;
  private pollInterval: any;
  private subscriptions: Subscription[] = [];

  constructor(
    private workflowService: WorkflowService,
    private snackBar: MatSnackBar
  ) {}

  ngOnDestroy(): void {
    // Clean up polling interval
    if (this.pollInterval) {
      clearInterval(this.pollInterval);
    }
    
    // Unsubscribe from all subscriptions
    this.subscriptions.forEach(sub => sub.unsubscribe());
  }

  generateFeatureFile(): void {
    if (this.isGenerating) {
      return;
    }

    this.isGenerating = true;
    
    // You can customize this request based on your needs
    const request: WorkflowRequest = {
      sprint_id: undefined, // You can add a form field to capture this
      jira_keys: undefined  // You can add a form field to capture this
    };

    const subscription = this.workflowService.triggerWorkflow(request).subscribe({
      next: (response) => {
        this.currentRunId = response.run_id;
        this.snackBar.open(
          `Workflow started successfully! Run ID: ${response.run_id}`, 
          'Close', 
          { duration: 5000 }
        );
        
        // Start polling for status updates
        this.pollWorkflowStatus();
      },
      error: (error) => {
        console.error('Error triggering workflow:', error);
        this.snackBar.open(
          'Error starting workflow. Please check the backend connection.', 
          'Close', 
          { duration: 5000 }
        );
        this.isGenerating = false;
      }
    });
    
    this.subscriptions.push(subscription);
  }

  private pollWorkflowStatus(): void {
    if (!this.currentRunId) {
      this.isGenerating = false;
      return;
    }

    // Clear any existing polling interval
    if (this.pollInterval) {
      clearInterval(this.pollInterval);
    }

    this.pollInterval = setInterval(() => {
      const subscription = this.workflowService.getWorkflowStatus(this.currentRunId!).subscribe({
        next: (status) => {
          if (status.status === 'completed' || status.status === 'failed') {
            clearInterval(this.pollInterval);
            this.isGenerating = false;
            
            if (status.status === 'completed') {
              this.snackBar.open(
                'Feature file generation completed successfully!', 
                'Close', 
                { duration: 5000 }
              );
            } else {
              this.snackBar.open(
                `Workflow failed: ${status.error || 'Unknown error'}`, 
                'Close', 
                { duration: 5000 }
              );
            }
          }
        },
        error: (error) => {
          console.error('Error checking workflow status:', error);
          clearInterval(this.pollInterval);
          this.isGenerating = false;
        }
      });
      
      this.subscriptions.push(subscription);
    }, 2000); // Poll every 2 seconds
  }
}