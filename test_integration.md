# Testing the Integration

## How to Test the Generate Feature File Button

### Prerequisites
1. Make sure the FastAPI backend is running on port 8000
2. Make sure the Angular frontend is running on port 4200

### Steps to Test

1. **Start the Backend Server:**
   ```bash
   cd backend
   python main.py
   ```
   The server should start on `http://localhost:8000`

2. **Start the Angular Frontend:**
   ```bash
   cd angular
   npm start
   ```
   The frontend should start on `http://localhost:4200`

3. **Test the Integration:**
   - Open your browser and navigate to `http://localhost:4200`
   - Look for the "Generate Feature File" button in the study controls section
   - Click the button
   - You should see:
     - The button shows a loading spinner and "Generating..." text
     - A success notification appears with a run ID
     - The button returns to normal state when the workflow completes

### Expected Behavior

1. **Button Click:**
   - Button becomes disabled and shows loading spinner
   - Text changes to "Generating..."

2. **API Call:**
   - Makes POST request to `http://localhost:8000/workflow/trigger`
   - Sends empty request body (sprint_id and jira_keys are undefined)

3. **Response Handling:**
   - Shows success notification with run ID
   - Starts polling for workflow status every 2 seconds

4. **Status Polling:**
   - Polls `http://localhost:8000/workflow/status/{run_id}` every 2 seconds
   - Shows completion notification when workflow finishes
   - Shows error notification if workflow fails

### Troubleshooting

1. **CORS Issues:**
   - If you see CORS errors, make sure the FastAPI backend has CORS middleware enabled
   - Add this to your FastAPI app:
   ```python
   from fastapi.middleware.cors import CORSMiddleware
   
   app.add_middleware(
       CORSMiddleware,
       allow_origins=["http://localhost:4200"],
       allow_credentials=True,
       allow_methods=["*"],
       allow_headers=["*"],
   )
   ```

2. **Connection Refused:**
   - Make sure the backend is running on port 8000
   - Check the console for any error messages

3. **Button Not Working:**
   - Check browser console for JavaScript errors
   - Verify all imports are correct in the Angular component

### API Endpoints Used

- `POST /workflow/trigger` - Triggers the workflow
- `GET /workflow/status/{run_id}` - Gets workflow status
- `GET /health` - Health check (optional)

### Customization

You can customize the workflow request by modifying the `generateFeatureFile()` method in `study-controls.component.ts`:

```typescript
const request: WorkflowRequest = {
  sprint_id: 123, // Add your sprint ID
  jira_keys: ['JIRA-123', 'JIRA-456'] // Add your Jira keys
};
```
