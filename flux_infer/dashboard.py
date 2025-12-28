import time
import random
import sys
import os
from datetime import datetime

# ANSI colors
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def draw_bar(percentage, width=30):
    fill = int(percentage * width)
    bar = "█" * fill + "░" * (width - fill)
    color = Colors.GREEN
    if percentage > 0.7: color = Colors.WARNING
    if percentage > 0.9: color = Colors.FAIL
    return f"{color}{bar}{Colors.ENDC}"

def dashboard_loop():
    try:
        start_time = time.time()
        requests_processed = 0
        total_tokens = 0
        
        while True:
            clear_screen()
            uptime = time.time() - start_time
            requests_processed += random.randint(1, 5)
            new_tokens = random.randint(50, 200)
            total_tokens += new_tokens
            
            gpu_util = random.uniform(0.85, 0.99)
            vram_util = random.uniform(0.40, 0.55) # Efficient due to PagedAttention
            kv_cache_frag = random.uniform(0.01, 0.05)
            
            # Header
            print(f"{Colors.HEADER}{Colors.BOLD}FluxInfer Real-Time Optimization Engine{Colors.ENDC}")
            print(f"Status: {Colors.GREEN}● ONLINE{Colors.ENDC} | Mode: {Colors.CYAN}O3 (Speculative){Colors.ENDC} | Uptime: {uptime:.1f}s")
            print("=" * 60)
            
            # Key Metrics
            print(f"\n{Colors.BOLD}Cluster Telemetry (Simulated H100 Node){Colors.ENDC}")
            print(f"GPU Utilization  : {draw_bar(gpu_util)} {gpu_util*100:.1f}%")
            print(f"VRAM Usage       : {draw_bar(vram_util)} {vram_util*100:.1f}% (Optimized)")
            print(f"KV Cache Frag.   : {draw_bar(kv_cache_frag)} {kv_cache_frag*100:.2f}%")
            
            # Throughput
            curr_tput = random.uniform(400, 700)
            print(f"\n{Colors.BOLD}Throughput{Colors.ENDC}")
            print(f"Current Speed    : {Colors.GREEN}{curr_tput:.1f} tok/s{Colors.ENDC}")
            print(f"Total Tokens     : {total_tokens:,}")
            print(f"Requests/sec     : {random.uniform(15, 25):.1f}")

            # Cost Savings
            # Assume $2.50 per 1M tokens vs $0.35
            savings = (total_tokens / 1_000_000) * (2.50 - 0.35)
            print(f"\n{Colors.BOLD}Financial Impact{Colors.ENDC}")
            print(f"Est. Cost Savings: {Colors.GREEN}${savings:.4f}{Colors.ENDC}")
            
            print("\n" + "=" * 60)
            print(f"{Colors.CYAN}[Ctrl+C] to Exit Dashboard{Colors.ENDC}")
            
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\nStopping dashboard...")

if __name__ == "__main__":
    dashboard_loop()
