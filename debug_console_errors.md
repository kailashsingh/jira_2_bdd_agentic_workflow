# Debug Console Errors Guide

## Common Console Errors and Solutions

### 1. **Missing Module Errors**
**Error:** `Error: Can't resolve all parameters for StudyControlsComponent`
**Solution:** Make sure all required modules are imported in `main.ts`

### 2. **MatOptionModule Missing**
**Error:** `mat-option is not a known element`
**Solution:** Added `MatOptionModule` to imports in `main.ts`

### 3. **CORS Errors**
**Error:** `Access to fetch at 'http://localhost:8000' from origin 'http://localhost:4200' has been blocked by CORS policy`
**Solution:** 
- Backend has CORS middleware configured
- Make sure backend is running on port 8000
- Check browser network tab for actual error

### 4. **HTTP Client Not Available**
**Error:** `NullInjectorError: No provider for HttpClient`
**Solution:** `HttpClientModule` is imported in `main.ts`

### 5. **Memory Leaks**
**Error:** Console warnings about memory leaks
**Solution:** Added proper cleanup in `ngOnDestroy()` method

## Debugging Steps

### Step 1: Check Browser Console
1. Open Developer Tools (F12)
2. Go to Console tab
3. Look for red error messages
4. Check Network tab for failed requests

### Step 2: Verify Backend is Running
```bash
# In backend directory
python main.py
```
Should show: `Uvicorn running on http://0.0.0.0:8000`

### Step 3: Test Backend Directly
Open `test_backend_connection.html` in browser and click "Test Backend Connection"

### Step 4: Check Angular Build
```bash
# In angular directory
npm run build
```
Look for TypeScript compilation errors

### Step 5: Check Network Requests
1. Open Developer Tools → Network tab
2. Click "Generate Feature File" button
3. Look for:
   - POST request to `/workflow/trigger`
   - GET requests to `/workflow/status/{run_id}`
   - Any failed requests (red status)

## Common Fixes

### Fix 1: Clear Browser Cache
- Hard refresh: Ctrl+Shift+R (Windows) or Cmd+Shift+R (Mac)
- Or clear browser cache completely

### Fix 2: Restart Development Servers
```bash
# Stop both servers (Ctrl+C)
# Then restart:
cd backend && python main.py
cd angular && npm start
```

### Fix 3: Check Port Conflicts
- Backend should run on port 8000
- Angular should run on port 4200
- Check if ports are already in use

### Fix 4: Verify Dependencies
```bash
cd angular
npm install
```

## Expected Console Output (Success)

When working correctly, you should see:
1. No red error messages in console
2. Network requests showing 200 status
3. Success notifications in the UI
4. Polling requests every 2 seconds

## Troubleshooting Specific Errors

### "Cannot find module" errors
- Run `npm install` in angular directory
- Check if all imports are correct

### "Expression has changed after it was checked" errors
- This is usually a change detection issue
- The current implementation should handle this correctly

### "Maximum call stack exceeded" errors
- Usually caused by infinite loops in polling
- The current implementation has proper cleanup

## Testing Checklist

- [ ] Backend starts without errors
- [ ] Angular starts without errors
- [ ] No console errors on page load
- [ ] "Generate Feature File" button is visible
- [ ] Button click triggers API call
- [ ] Success notification appears
- [ ] Polling starts and stops correctly
- [ ] No memory leaks (check console for warnings)
