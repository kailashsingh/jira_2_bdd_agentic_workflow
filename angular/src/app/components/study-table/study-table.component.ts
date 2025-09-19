import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatTableModule } from '@angular/material/table';
import { MatButtonModule } from '@angular/material/button';
import { MatChipsModule } from '@angular/material/chips';
import { MatIconModule } from '@angular/material/icon';

export interface StudyEntry {
  id: string;
  batch: string;
  date: string;
  time: string;
  status: 'validated' | 'pending' | 'error';
  pr: string;
}

@Component({
  selector: 'app-study-table',
  standalone: true,
  imports: [
    CommonModule,
    MatTableModule,
    MatButtonModule,
    MatChipsModule,
    MatIconModule
  ],
  template: `
    <div class="border rounded-lg overflow-hidden">
      <table mat-table [dataSource]="dataSource" class="w-full">
        
        <!-- Batch Column -->
        <ng-container matColumnDef="batch">
          <th mat-header-cell *matHeaderCellDef>Batch</th>
          <td mat-cell *matCellDef="let element">
            <mat-chip-set>
              <mat-chip>{{ element.batch }}</mat-chip>
            </mat-chip-set>
          </td>
        </ng-container>

        <!-- Date Column -->
        <ng-container matColumnDef="date">
          <th mat-header-cell *matHeaderCellDef>Date</th>
          <td mat-cell *matCellDef="let element">{{ element.date }}</td>
        </ng-container>

        <!-- Time Column -->
        <ng-container matColumnDef="time">
          <th mat-header-cell *matHeaderCellDef>Time</th>
          <td mat-cell *matCellDef="let element">{{ element.time }}</td>
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
            </div>
          </td>
        </ng-container>

        <!-- PR Column -->
        <ng-container matColumnDef="pr">
          <th mat-header-cell *matHeaderCellDef>PR</th>
          <td mat-cell *matCellDef="let element">
            <mat-chip-set>
              <mat-chip color="accent">{{ element.pr }}</mat-chip>
            </mat-chip-set>
          </td>
        </ng-container>

        <!-- Action Column -->
        <ng-container matColumnDef="action">
          <th mat-header-cell *matHeaderCellDef class="w-100">Action</th>
          <td mat-cell *matCellDef="let element">
            <button mat-icon-button>
              <mat-icon>chevron_right</mat-icon>
            </button>
          </td>
        </ng-container>

        <tr mat-header-row *matHeaderRowDef="displayedColumns"></tr>
        <tr mat-row *matRowDef="let row; columns: displayedColumns;"></tr>
      </table>
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
    .gap-2 { gap: 0.5rem; }
    .capitalize { text-transform: capitalize; }
    .text-sm { font-size: 0.875rem; }
    
    .text-green-600 { color: #059669; }
    .text-yellow-600 { color: #d97706; }
    .text-red-600 { color: #dc2626; }
    
    mat-icon.h-4 { font-size: 1rem; height: 1rem; width: 1rem; }
  `]
})
export class StudyTableComponent {
  displayedColumns: string[] = ['batch', 'date', 'time', 'status', 'pr', 'action'];
  
  dataSource: StudyEntry[] = [
    {
      id: '1',
      batch: 'BATCH-001',
      date: '2024-01-15',
      time: '09:30',
      status: 'validated',
      pr: 'PR-001'
    },
    {
      id: '2',
      batch: 'BATCH-002',
      date: '2024-01-15',
      time: '14:15',
      status: 'pending',
      pr: 'PR-002'
    },
    {
      id: '3',
      batch: 'BATCH-001',
      date: '2024-01-14',
      time: '11:45',
      status: 'error',
      pr: 'PR-003'
    },
    {
      id: '4',
      batch: 'BATCH-003',
      date: '2024-01-14',
      time: '16:20',
      status: 'validated',
      pr: 'PR-004'
    },
    {
      id: '5',
      batch: 'BATCH-002',
      date: '2024-01-13',
      time: '13:10',
      status: 'pending',
      pr: 'PR-005'
    }
  ];

  getStatusIcon(status: string): string {
    switch (status) {
      case 'validated':
        return 'check_circle';
      case 'pending':
        return 'schedule';
      case 'error':
        return 'error';
      default:
        return 'help';
    }
  }

  getStatusIconClass(status: string): string {
    switch (status) {
      case 'validated':
        return 'text-green-600';
      case 'pending':
        return 'text-yellow-600';
      case 'error':
        return 'text-red-600';
      default:
        return '';
    }
  }
}