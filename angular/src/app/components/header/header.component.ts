import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';

@Component({
  selector: 'app-header',
  standalone: true,
  imports: [
    CommonModule,
    MatButtonModule,
    MatIconModule
  ],
  template: `
    <!-- Header Section -->
    <div class="border-b bg-card px-6 py-4">
      <div class="flex items-center justify-between">
        <div class="space-y-1">
          <h1 class="text-2xl">BDD Agentic Workflow</h1>
          <p class="text-muted-foreground">Behavior-driven development with intelligent automation</p>
        </div>
        <div class="flex items-center gap-3">
          <button mat-stroked-button>
            <mat-icon>settings</mat-icon>
            Settings
          </button>
        </div>
      </div>
    </div>
  `,
  styles: [`
    .border-b { border-bottom: 1px solid var(--border, rgba(0, 0, 0, 0.1)); }
    .bg-card { background-color: var(--card, #ffffff); }
    .px-6 { padding-left: 1.5rem; padding-right: 1.5rem; }
    .py-4 { padding-top: 1rem; padding-bottom: 1rem; }
    .flex { display: flex; }
    .items-center { align-items: center; }
    .justify-between { justify-content: space-between; }
    .gap-3 { gap: 0.75rem; }
    .space-y-1 > * + * { margin-top: 0.25rem; }
    .text-2xl { font-size: 1.5rem; line-height: 2rem; }
    .text-muted-foreground { color: var(--muted-foreground, #717182); }
    
    mat-icon { margin-right: 0.5rem; }
  `]
})
export class HeaderComponent {
}
