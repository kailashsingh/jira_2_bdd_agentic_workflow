import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatButtonModule } from '@angular/material/button';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatSelectModule } from '@angular/material/select';
import { MatIconModule } from '@angular/material/icon';

@Component({
  selector: 'app-study-controls',
  standalone: true,
  imports: [
    CommonModule,
    MatButtonModule,
    MatFormFieldModule,
    MatInputModule,
    MatSelectModule,
    MatIconModule
  ],
  template: `
    <div class="flex flex-col sm:flex-row gap-4 items-start sm:items-center justify-between">
      <div class="flex flex-1 gap-2 items-center">
        <mat-form-field class="flex-1 max-w-sm" appearance="outline">
          <mat-icon matPrefix>search</mat-icon>
          <input matInput placeholder="Search studies...">
        </mat-form-field>
        
        <mat-form-field class="w-130" appearance="outline">
          <mat-select placeholder="Filter">
            <mat-icon matPrefix>filter_list</mat-icon>
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
        <button mat-flat-button color="primary">
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
export class StudyControlsComponent {}