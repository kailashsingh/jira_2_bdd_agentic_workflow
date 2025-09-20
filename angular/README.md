# BDD Agentic Workflow - Angular

This is an Angular conversion of the React BDD (Behavior-Driven Development) Agentic Workflow application. The application provides intelligent automation for behavior-driven development workflows with comprehensive study management and batch processing features.

## Features

- **On-Demand Validation**: JIRA ID-based feature file generation with real-time validation
- **Batch Job Processing**: Comprehensive form-based batch job triggering with multiple parameters
- **Study Management**: Track and manage study entries with batch processing and status tracking
- **Interactive Controls**: Search, filter, and manage studies efficiently
- **Status Tracking**: Visual indicators for validated, pending, and error states
- **Information Tooltips**: Contextual help and guidance throughout the interface
- **Responsive Design**: Mobile-first design with Material Design components

## Tech Stack

- **Angular 17** - Modern Angular with standalone components
- **Angular Material** - Material Design components for UI
- **TypeScript** - Type-safe development
- **SCSS** - Enhanced styling capabilities
- **Material Icons** - Beautiful icons for UI elements

## Project Structure

```
src/
├── app/
│   ├── components/
│   │   ├── job-status-panel/      # Combined job status panel with grid
│   │   └── feature-generation-job/ # Feature generation job panel
│   └── app.component.ts           # Root component
├── styles.scss                    # Global styles and CSS variables
├── main.ts                        # Application bootstrap
└── index.html                     # HTML entry point
```

## Getting Started

### Prerequisites

- Node.js (v16 or higher)
- Angular CLI (`npm install -g @angular/cli`)

### Installation

1. Clone or extract the project files
2. Navigate to the project directory:
   ```bash
   cd bdd-agentic-workflow-angular
   ```

3. Install dependencies:
   ```bash
   npm install
   ```

4. Start the development server:
   ```bash
   npm start
   # or
   ng serve
   ```

5. Open your browser and navigate to `http://localhost:4200`

### Building for Production

```bash
npm run build
# or
ng build --configuration production
```

## Key Differences from React Version

### Architecture
- **Standalone Components**: Uses Angular 17's standalone components instead of NgModules
- **Dependency Injection**: Leverages Angular's DI system for services
- **Material Design**: Uses Angular Material instead of shadcn/ui components

### Component Structure
- **Template-driven**: Uses Angular templates with structural directives
- **Two-way Binding**: Utilizes Angular's powerful data binding features
- **Lifecycle Hooks**: Implements Angular component lifecycle methods

### Styling
- **SCSS Support**: Enhanced styling with Sass preprocessing
- **CSS Custom Properties**: Maintains the same design system variables
- **Material Theming**: Integrates with Angular Material's theming system

## Components Overview

### JobStatusPanelComponent
Combined job status panel featuring:
- **Header Section**: Application title and settings controls
- **Search & Filter**: Search by run ID, message, or PRs with status filtering
- **Job Status Grid**: Comprehensive table with columns for:
  - Run ID: Unique job identifier
  - Started At: Job initiation timestamp
  - Status: Visual status indicators (running, completed, failed, pending)
  - Pull Requests: PR chips for each job
  - Message: Job status messages and error details
  - Action: Retrigger button for individual jobs
- **Real-time Updates**: Loading states and status indicators

### FeatureGenerationJobComponent
Two-panel layout featuring:

#### On-Demand Validation Panel
- JIRA ID input field with tooltip guidance
- Generate Feature File button
- Information icon with contextual help

#### Trigger Batch Job Panel
- Comprehensive form with multiple input fields:
  - Tribe (with tooltip)
  - Team Name (with tooltip)
  - Component Name (with tooltip)
  - JS Iteration (with tooltip)
  - Sprint (with tooltip)
  - Status (with tooltip)
- Generate Feature File action button
- Information icon with detailed guidance

## Customization

### Theming
The application uses CSS custom properties for theming. Modify the variables in `src/styles.scss`:

```scss
:root {
  --primary: #your-color;
  --background: #your-background;
  // ... other variables
}
```

### Adding New Components
Generate new components using Angular CLI:
```bash
ng generate component components/your-component --standalone
```

## Development Commands

- `npm start` or `ng serve` - Start development server (runs on http://localhost:4200)
- `npm run build` - Build for production
- `npm run watch` - Build and watch for changes
- `npm test` - Run unit tests
- `npm run lint` - Run linting
- `ng generate component <name>` - Generate new component
- `ng generate service <name>` - Generate new service

## Recent Updates

### Version 1.0.0 Features
- ✅ Complete Angular 17 migration with standalone components
- ✅ Material Design implementation with Angular Material
- ✅ On-demand validation with JIRA ID integration
- ✅ Comprehensive batch job processing form
- ✅ Information tooltips and contextual help
- ✅ Responsive design with mobile-first approach
- ✅ Clean, professional UI with consistent styling
- ✅ Study management with status tracking
- ✅ Interactive search and filtering capabilities

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Original Design

This Angular project is based on the original Figma design available at: https://www.figma.com/design/P5H2vwVE6fumxW8S2UHuc4/Create-UX-Design

## Support

For issues and questions, please create an issue in the repository or refer to the Angular documentation at https://angular.io/docs
