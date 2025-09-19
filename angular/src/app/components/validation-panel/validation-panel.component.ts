import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { MatChipsModule } from '@angular/material/chips';
import { MatIconModule } from '@angular/material/icon';
import { MatDividerModule } from '@angular/material/divider';

@Component({
  selector: 'app-validation-panel',
  standalone: true,
  imports: [
    CommonModule,
    MatCardModule,
    MatButtonModule,
    MatChipsModule,
    MatIconModule,
    MatDividerModule
  ],
  template: `
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <mat-card>
        <mat-card-header>
          <mat-card-title class="flex items-center gap-2" title="">
            On-Demand Validation
            <mat-icon class="text-blue-600" style="font-size: 16px;" title="Insert JIRA ID and generate feature files on demand">info</mat-icon>
          
          </mat-card-title>
        </mat-card-header>
        <mat-card-content class="space-y-4">
 
          <mat-divider></mat-divider>
          <div class="flex items-center justify-between">
            <div class="flex items-center gap-1">
              <label class="text-sm font-medium">JIRA ID</label>
            </div>
            <input type="text" class="border rounded px-2 py-1 w-32" placeholder="Enter JIRA ID">
          </div>
          
          <div style="height: 150px; background-color: transparent;"></div>
          
          <button mat-stroked-button class="w-full" style="margin-top: 26px;">
            <span>Generate Feature File</span>
          </button>
          
        </mat-card-content>
      </mat-card>

      <mat-card>
        <mat-card-header>
          <mat-card-title class="flex items-center gap-2">
          Trigger Batch Job
          <mat-icon class="text-blue-600" style="font-size: 16px;" title="Insert Tribe, Team, Component, JS Iteration, Sprint, Status and Trigger Batch Job">info</mat-icon>
            
          </mat-card-title>
        </mat-card-header>
        <mat-card-content class="space-y-3">
        <mat-divider></mat-divider>
            <div class="flex items-center justify-between">
              <div class="flex items-center gap-1">
                <label class="text-sm font-medium">Tribe</label>
              </div>
              <input type="text" class="border rounded px-2 py-1 w-32" placeholder="Enter Tribe">
            </div>
          
           <div class="flex items-center justify-between">
            <div class="flex items-center gap-1">
              <label class="text-sm font-medium">Team Name</label>
              </div>
            <input type="text" class="border rounded px-2 py-1 w-32" placeholder="Enter Team">
          </div>

           <div class="flex items-center justify-between">
            <div class="flex items-center gap-1">
              <label class="text-sm font-medium">Component Name</label>
              </div>
            <input type="text" class="border rounded px-2 py-1 w-32" placeholder="Component Name">
          </div>

           <div class="flex items-center justify-between">
            <div class="flex items-center gap-1">
              <label class="text-sm font-medium">JS Iteration</label>
             </div>
            <input type="text" class="border rounded px-2 py-1 w-32" placeholder="JS Iteration">
          </div>

           <div class="flex items-center justify-between">
            <div class="flex items-center gap-1">
              <label class="text-sm font-medium">Sprint</label>
              </div>
            <input type="text" class="border rounded px-2 py-1 w-32" placeholder="Enter Sprint">
          </div>

          <div class="flex items-center justify-between">
            <div class="flex items-center gap-1">
              <label class="text-sm font-medium">Status</label>
            </div>
            <input type="text" class="border rounded px-2 py-1 w-32" placeholder="Enter Status">
          </div>
          
          <button mat-stroked-button class="w-full">
            <span>Generate Feature File</span>
          </button>
        </mat-card-content>
      </mat-card>
    </div>
  `,
  styles: [`
    .grid { display: grid; }
    .grid-cols-1 { grid-template-columns: repeat(1, minmax(0, 1fr)); }
    .gap-6 { gap: 1.5rem; }
    .flex { display: flex; }
    .items-center { align-items: center; }
    .justify-between { justify-content: space-between; }
    .gap-2 { gap: 0.5rem; }
    .space-y-4 > * + * { margin-top: 1rem; }
    .space-y-3 > * + * { margin-top: 0.75rem; }
    .space-y-2 > * + * { margin-top: 0.5rem; }
    .text-green-600 { color: #059669; }
    .text-yellow-600 { color: #d97706; }
    .text-red-600 { color: #dc2626; }
    .text-blue-600 { color: #2563eb; }
    .text-sm { font-size: 0.875rem; }
    .text-muted-foreground { color: var(--muted-foreground, #717182); }
    .h-4 { font-size: 1rem; height: 1rem; width: 1rem; }
    .w-full { width: 100%; }
    .pt-2 { padding-top: 0.5rem; }
    .mb-2 { margin-bottom: 0.5rem; }
    .font-bold { font-weight: 700; }
    .border { border: 1px solid var(--border, rgba(0, 0, 0, 0.1)); }
    .rounded { border-radius: 0.25rem; }
    .px-2 { padding-left: 0.5rem; padding-right: 0.5rem; }
    .py-1 { padding-top: 0.25rem; padding-bottom: 0.25rem; }
    .w-32 { width: 8rem; }
    .font-medium { font-weight: 500; }
    
    @media (min-width: 1024px) {
      .lg\\:grid-cols-2 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    
    .justify-between button {
      display: flex;
      justify-content: space-between;
      align-items: center;
      width: 100%;
    }
  `]
})
export class ValidationPanelComponent {}