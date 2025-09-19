import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { StudyHeaderComponent } from './components/study-header/study-header.component';
import { StudyControlsComponent } from './components/study-controls/study-controls.component';
import { StudyTableComponent } from './components/study-table/study-table.component';
import { ValidationPanelComponent } from './components/validation-panel/validation-panel.component';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [
    CommonModule,
    StudyHeaderComponent,
    StudyControlsComponent,
    StudyTableComponent,
    ValidationPanelComponent
  ],
  template: `
    <div class="min-h-screen bg-background">
      <app-study-header></app-study-header>
      <div class="container mx-auto px-6 py-8 space-y-8">
        <app-validation-panel></app-validation-panel>
        <div class="space-y-4">
          <div class="flex items-center justify-between">
            <h2>Study Entries</h2>
          </div>
          <app-study-controls></app-study-controls>
          <app-study-table></app-study-table>
        </div>
      </div>
    </div>
  `,
  styles: [`
    .min-h-screen { min-height: 100vh; }
    .bg-background { background-color: var(--background, #ffffff); }
    .container { max-width: 1200px; }
    .mx-auto { margin-left: auto; margin-right: auto; }
    .px-6 { padding-left: 1.5rem; padding-right: 1.5rem; }
    .py-8 { padding-top: 2rem; padding-bottom: 2rem; }
    .space-y-8 > * + * { margin-top: 2rem; }
    .space-y-4 > * + * { margin-top: 1rem; }
    .flex { display: flex; }
    .items-center { align-items: center; }
    .justify-between { justify-content: space-between; }
  `]
})
export class AppComponent {
  title = 'BDD Agentic Workflow';
}