"""Extracts key log events for TRK_104 and TRK_109 from the audit log."""
import sys

log_path = r'C:\Users\Vijayakumari K\.gemini\antigravity-ide\brain\88970f11-55a8-439c-a628-916bcfd59c9a\.system_generated\tasks\task-38.log'
with open(log_path, encoding='utf-8', errors='replace') as f:
    lines = f.readlines()

for t in ['TRK_104','TRK_109']:
    hits = [l.strip() for l in lines if t in l and any(k in l for k in ['OCR_ATTEMPT','OBSERVATION_CREATED','CONFIRMED','VALIDATION_ACCEPTED','fast_path','MAX_ATTEMPTS'])]
    print('=== ' + t + ' (' + str(len(hits)) + ' events) ===')
    for h in hits[:25]:
        print('  ' + h)
    print()
sys.stdout.flush()
