import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { JobStatusPanelComponent } from './components/job-status-panel/job-status-panel.component';
import { FeatureGenerationJobComponent } from './components/feature-generation-job/feature-generation-job.component';
import { HeaderComponent } from './components/header/header.component';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [
    CommonModule,
    HeaderComponent,
    JobStatusPanelComponent,
    FeatureGenerationJobComponent
  ],
  template: `
    <div class="min-h-screen bg-background">
      <app-header></app-header>
      <div class="container mx-auto space-y-8">
        <app-feature-generation-job></app-feature-generation-job>
        <app-job-status-panel></app-job-status-panel>
      </div>
    </div>
  `,
  styles: [`
    .min-h-screen { min-height: 100vh; }
    .bg-background { background-color: var(--background, #ffffff); }
    .container { max-width: 1200px; }
    .mx-auto { margin-left: auto; margin-right: auto; }
    .space-y-8 > * + * { margin-top: 2rem; }
  `]
})
export class AppComponent {
  title = 'BDD Agentic Workflow';
}