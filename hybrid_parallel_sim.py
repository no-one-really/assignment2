import time
import random
import queue
import threading

class Device:
    def __init__(self, rank, stage, world_size, pipeline_depth):
        self.rank = rank
        self.stage = stage # Pipeline stage (0 to pipeline_depth - 1)
        self.world_size = world_size
        self.pipeline_depth = pipeline_depth
        self.memory = 0 # Track memory usage (simplified)
        self.compute_queue = queue.Queue()
        self.comm_queue = queue.Queue()
        self.log = []

    def log_event(self, event, duration):
        start_time = time.time()
        self.log.append({
            "rank": self.rank,
            "event": event,
            "start": start_time,
            "duration": duration
        })
        time.sleep(duration) # Simulate work

    def forward(self, micro_batch_id):
        print(f"[Rank {self.rank}] Starting Forward on MB {micro_batch_id}")
        self.log_event(f"Forward MB {micro_batch_id}", 0.05) # simulate 50ms compute
        self.memory += 10 # specific activation memory
        print(f"[Rank {self.rank}] Finished Forward on MB {micro_batch_id}")

    def backward(self, micro_batch_id):
        print(f"[Rank {self.rank}] Starting Backward on MB {micro_batch_id}")
        self.log_event(f"Backward MB {micro_batch_id}", 0.08) # simulate 80ms compute (backward is usually costlier)
        self.memory -= 10 # release activation memory
        print(f"[Rank {self.rank}] Finished Backward on MB {micro_batch_id}")

    def all_reduce(self):
        print(f"[Rank {self.rank}] Starting Ring All-Reduce")
        # Simulate Ring All-Reduce: 2 * (N-1) steps
        # For N=2 (Data Parallel size), it's 2 steps.
        step_duration = 0.02
        steps = 2 * (2 - 1) # hardcoded for DP size 2 for this demo
        for s in range(steps):
            self.log_event(f"All-Reduce Step {s+1}", step_duration)
        print(f"[Rank {self.rank}] Finished Ring All-Reduce")

class HybridTrainer:
    def __init__(self):
        # Topology: 4 Devices
        # Data Parallel Size = 2 (Group A: {0, 2}, Group B: {1, 3}) ? No.
        # Let's map: 
        # Rank 0: Stage 0, DP Group 0
        # Rank 1: Stage 1, DP Group 0
        # Rank 2: Stage 0, DP Group 1
        # Rank 3: Stage 1, DP Group 1
        # Pipeline: 0->1 and 2->3
        # Data Parallel Groups: {0, 2} and {1, 3}
        
        self.devices = [
            Device(0, 0, 4, 2),
            Device(1, 1, 4, 2),
            Device(2, 0, 4, 2),
            Device(3, 1, 4, 2)
        ]
        self.micro_batches = 8 # Total micro-batches per global batch
        
    def run_1f1b_schedule(self, rank):
        device = self.devices[rank]
        # 1F1B Logic for 2 stages
        # Warmup: Stage 0 does F, sends to Stage 1. 
        # Detailed 1F1B schedule is complex, simplified version here:
        
        if device.stage == 0:
            # Stage 0 (First Stage)
            # Schedule: F, F, F, F ... then B, B, B, B interleaved? 
            # 1F1B Steady state: F, B, F, B
            
            # Simple schedule for demo:
            # F(0), F(1), F(2) ... F(7) -> This is GPipe
            # 1F1B:
            # warmup_steps = pipeline_depth - stage - 1 = 1
            
            # F(0)
            device.forward(0)
            # Send to next stage (simulated)
            
            # F(1)
            device.forward(1)
            
            # Receive Grad from next stage for 0 (simulated wait)
            # B(0)
            device.backward(0)
            
            # F(2)
            device.forward(2)
            # B(1)
            device.backward(1)

            # ... continue pattern ...
            for i in range(2, self.micro_batches): # 2 to 7
                if i < self.micro_batches: device.forward(i) # Already done F(2) above? wait, loop logic fix
                if i-1 >= 0: device.backward(i-1)
                
            # Cooldown
            device.backward(self.micro_batches - 1)
            
        elif device.stage == 1:
            # Stage 1 (Last Stage)
            # Wait for F(0)
            # F(0)
            device.forward(0)
            # B(0)
            device.backward(0)
            
            # F(1)
            device.forward(1)
            # B(1)
            device.backward(1)
            
            # ... Loop
            for i in range(2, self.micro_batches):
                device.forward(i)
                device.backward(i)
        
        # After all micro-batches, do All-Reduce
        device.all_reduce()

    def run(self):
        threads = []
        start_time = time.time()
        for i in range(4):
            t = threading.Thread(target=self.run_1f1b_schedule, args=(i,))
            threads.append(t)
            t.start()
            
        for t in threads:
            t.join()
        end_time = time.time()
        
        print(f"Total Training Time: {end_time - start_time:.4f}s")
        self.generate_timeline()

    def generate_timeline(self):
        # Collect logs and print a simple ASCII timeline
        print("\n--- Timeline Execution Log ---")
        all_events = []
        for d in self.devices:
            for e in d.log:
                all_events.append(e)
        
        # Sort by start time
        all_events.sort(key=lambda x: x['start'])
        
        # Normalize time
        if not all_events: return
        t0 = all_events[0]['start']
        
        print(f"{'Time (s)':<10} | {'Rank 0 (Stage 0)':<25} | {'Rank 1 (Stage 1)':<25} | {'Rank 2 (Stage 0)':<25} | {'Rank 3 (Stage 1)':<25}")
        print("-" * 120)
        
        # Visualization is tricky in text, let's just list significant events roughly in order
        # Or just print the sorted list with column alignment
        
        for e in all_events:
            t_rel = e['start'] - t0
            col_width = 25
            cols = [""] * 4
            cols[e['rank']] = e['event']
            print(f"{t_rel:<10.3f} | {cols[0]:<{col_width}} | {cols[1]:<{col_width}} | {cols[2]:<{col_width}} | {cols[3]:<{col_width}}")

if __name__ == "__main__":
    trainer = HybridTrainer()
    trainer.run()
