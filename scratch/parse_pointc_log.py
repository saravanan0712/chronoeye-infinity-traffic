"""Extracts key log events for specific tracks from the audit log."""
import sys

log_path = r'C:\Users\Vijayakumari K\.gemini\antigravity-ide\brain\88970f11-55a8-439c-a628-916bcfd59c9a\.system_generated\tasks\task-38.log'
with open(log_path, encoding='utf-8', errors='replace') as f:
    lines = f.readlines()

tracks = ['TRK_104','TRK_109','TRK_114','TRK_116','TRK_117','TRK_125','TRK_127','TRK_106']
keywords = ['OCR_ATTEMPT','OBSERVATION_CREATED','CONFIRMED','VALIDATION_ACCEPTED','fast_path','MAX_ATTEMPTS','SKIP']

for t in tracks:
    hits = [l.strip() for l in lines if t in l and any(k in l for k in keywords)]
    print('=== ' + t + ' (' + str(len(hits)) + ' events) ===')
    # Show OCR attempts and key state changes, max 20
    key = [h for h in hits if 'OCR_ATTEMPT' in h or 'OBSERVATION' in h or 'CONFIRMED' in h or 'fast_path' in h or 'MAX_ATTEMPTS' in h]
    for h in key[:20]:
        print('  ' + h)
    print()

sys.stdout.flush()
