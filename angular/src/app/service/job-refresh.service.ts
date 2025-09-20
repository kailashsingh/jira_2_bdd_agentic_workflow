import { Injectable } from '@angular/core';
import { Subject } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class JobRefreshService {
  private refreshTrigger = new Subject<void>();
  
  // Observable that components can subscribe to
  public refreshTriggered$ = this.refreshTrigger.asObservable();
  
  // Method to trigger a refresh
  triggerRefresh(): void {
    this.refreshTrigger.next();
  }
}
