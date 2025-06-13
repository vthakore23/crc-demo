#!/usr/bin/env python3
"""
Real-time Training Monitor for CRC Analysis Platform
Monitors training progress across all phases with live updates
"""

import os, sys, time, json, subprocess
from pathlib import Path
from datetime import datetime
import argparse

class TrainingMonitor:
    """Real-time monitor for training pipeline"""
    
    def __init__(self, log_dir="logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
    def get_active_processes(self):
        """Get list of active training processes"""
        try:
            result = subprocess.run(
                ["ps", "aux"], 
                capture_output=True, 
                text=True
            )
            processes = []
            for line in result.stdout.split('\n'):
                if 'train' in line and ('python' in line or 'pytorch' in line):
                    parts = line.split()
                    if len(parts) > 10:
                        processes.append({
                            'pid': parts[1],
                            'cpu': parts[2],
                            'mem': parts[3],
                            'command': ' '.join(parts[10:])
                        })
            return processes
        except:
            return []
    
    def monitor_log_file(self, log_file, follow=True):
        """Monitor a specific log file"""
        if not log_file.exists():
            print(f"⏳ Waiting for {log_file.name} to be created...")
            while not log_file.exists():
                time.sleep(1)
        
        print(f"📊 Monitoring {log_file.name}")
        print("=" * 60)
        
        with open(log_file, 'r') as f:
            # Read existing content
            content = f.read()
            if content:
                print(content)
            
            if follow:
                # Follow new content
                while True:
                    line = f.readline()
                    if line:
                        # Color code important lines
                        if 'Epoch' in line:
                            print(f"🔄 {line.strip()}")
                        elif 'TrainAcc' in line or 'ValAcc' in line:
                            print(f"📈 {line.strip()}")
                        elif '✓' in line or 'saved' in line:
                            print(f"✅ {line.strip()}")
                        elif 'error' in line.lower() or 'failed' in line.lower():
                            print(f"❌ {line.strip()}")
                        else:
                            print(line.strip())
                    else:
                        time.sleep(0.1)
    
    def show_dashboard(self):
        """Show training dashboard"""
        os.system('clear' if os.name == 'posix' else 'cls')
        print("🔬 CRC Analysis Platform - Training Monitor")
        print("=" * 60)
        print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Check for active processes
        processes = self.get_active_processes()
        if processes:
            print("🏃 Active Training Processes:")
            for proc in processes:
                print(f"  PID: {proc['pid']} | CPU: {proc['cpu']}% | MEM: {proc['mem']}%")
                print(f"      {proc['command'][:80]}...")
            print()
        
        # Check log files
        log_files = list(self.log_dir.glob("*.log"))
        if log_files:
            print("📋 Recent Log Files:")
            for log_file in sorted(log_files, key=lambda x: x.stat().st_mtime, reverse=True):
                size_kb = round(log_file.stat().st_size / 1024, 1)
                mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                print(f"  📄 {log_file.name} ({size_kb} KB) - {mtime.strftime('%H:%M:%S')}")
            print()
        
        # Check model files
        models_dir = Path("models")
        if models_dir.exists():
            model_files = list(models_dir.rglob("*.pth")) + list(models_dir.rglob("*.pkl"))
            if model_files:
                print("🤖 Available Models:")
                for model_file in model_files:
                    size_mb = round(model_file.stat().st_size / 1024 / 1024, 1)
                    mtime = datetime.fromtimestamp(model_file.stat().st_mtime)
                    print(f"  ✅ {model_file.relative_to('.')} ({size_mb} MB) - {mtime.strftime('%H:%M')}")
                print()
        
        # Usage instructions
        print("📖 Commands:")
        print("  python3 scripts/monitor_training.py --follow <log_file>  # Follow specific log")
        print("  python3 scripts/monitor_training.py --dashboard           # Show this dashboard")
        print("  python3 scripts/monitor_training.py --list               # List available logs")
        
    def list_logs(self):
        """List available log files"""
        log_files = list(self.log_dir.glob("*.log"))
        if not log_files:
            print("No log files found")
            return
            
        print("Available log files:")
        for i, log_file in enumerate(log_files, 1):
            size_kb = round(log_file.stat().st_size / 1024, 1)
            mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
            print(f"{i:2d}. {log_file.name} ({size_kb} KB) - {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
    
    def follow_latest(self):
        """Follow the most recently modified log file"""
        log_files = list(self.log_dir.glob("*.log"))
        if not log_files:
            print("No log files found")
            return
            
        latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
        print(f"Following latest log: {latest_log.name}")
        self.monitor_log_file(latest_log, follow=True)

def main():
    parser = argparse.ArgumentParser(description="CRC Training Monitor")
    parser.add_argument("--follow", type=str, help="Follow specific log file")
    parser.add_argument("--dashboard", action="store_true", help="Show training dashboard")
    parser.add_argument("--list", action="store_true", help="List available log files")
    parser.add_argument("--latest", action="store_true", help="Follow latest log file")
    parser.add_argument("--log-dir", default="logs", help="Log directory (default: logs)")
    
    args = parser.parse_args()
    
    monitor = TrainingMonitor(args.log_dir)
    
    if args.follow:
        log_file = Path(args.log_dir) / args.follow
        monitor.monitor_log_file(log_file, follow=True)
    elif args.dashboard:
        monitor.show_dashboard()
    elif args.list:
        monitor.list_logs()
    elif args.latest:
        monitor.follow_latest()
    else:
        # Default: show dashboard
        monitor.show_dashboard()

if __name__ == "__main__":
    main() 