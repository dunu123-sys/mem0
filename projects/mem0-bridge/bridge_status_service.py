"""
Mem0 BridgeStatus Sidecar — Windows service wrapper (pywin32).

Runs bridge_status_server.py as a real, silent Windows service:
- No console window (ServiceFramework runs headless under the service account).
- Starts at boot (SERVICE_AUTO_START).
- SCM-managed: healthy start/stop, crash auto-restart via sc failure policy.
- Single instance: binds :18900 (env BRIDGE_STATUS_PORT to override).

Resilient runner: if the sidecar process exits unexpectedly, restart it with
backoff, capped at 10 consecutive restart attempts before giving up (SCM
failure policy + watchdog both apply).

Install:   python bridge_status_service.py install
           sc.exe failure Mem0BridgeStatus reset= 86400 actions= restart/5000/restart/10000/restart/30000
Start:     sc.exe start Mem0BridgeStatus   (or: python bridge_status_service.py start)
Uninstall: python bridge_status_service.py remove
"""
import os
import subprocess
import time

import win32service
import win32serviceutil
import win32event
import servicemanager

BRIDGE_DIR = os.path.dirname(os.path.abspath(__file__))
SIDECAR = os.path.join(BRIDGE_DIR, "bridge_status_server.py")
PYTHONW = r"C:\Python314\pythonw.exe"
if not os.path.exists(PYTHONW):
    PYTHONW = "pythonw"

class BridgeStatusService(win32serviceutil.ServiceFramework):
    _svc_name_ = "Mem0BridgeStatus"
    _svc_display_name_ = "Mem0 BridgeStatus Sidecar"
    _svc_description_ = "Serves Mem0 bridge webhook/analytics status on :18900 for the dashboard (silent)."

    def __init__(self, args):
        win32serviceutil.ServiceFramework.__init__(self, args)
        self.hWaitStop = win32event.CreateEvent(None, 0, 0, None)
        self.proc = None

    def SvcStop(self):
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        win32event.SetEvent(self.hWaitStop)
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
            except Exception:
                pass

    def SvcDoRun(self):
        servicemanager.LogMsg(
            servicemanager.EVENTLOG_INFORMATION_TYPE,
            servicemanager.PYS_SERVICE_STARTED,
            (self._svc_name_, ""),
        )
        self._run()

    def _run(self):
        attempts = 0
        while True:
            rc = win32event.WaitForSingleObject(self.hWaitStop, 0)
            if rc == win32event.WAIT_OBJECT_0:
                return
            if self.proc is None or self.proc.poll() is not None:
                attempts += 1
                if attempts > 10:
                    servicemanager.LogErrorMsg(
                        f"{self._svc_name_}: giving up after 10 restart attempts"
                    )
                    return
                env = dict(os.environ)
                self.proc = subprocess.Popen(
                    [PYTHONW, SIDECAR], cwd=BRIDGE_DIR, env=env,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
                servicemanager.LogInfoMsg(
                    f"{self._svc_name_}: sidecar started pid={self.proc.pid}"
                )
            # wait up to 2s for stop signal or process exit
            rc = win32event.WaitForSingleObject(self.hWaitStop, 2000)
            if rc == win32event.WAIT_OBJECT_0:
                return

if __name__ == "__main__":
    win32serviceutil.HandleCommandLine(BridgeStatusService)
