import React from 'react';
import { Camera, Eye, Globe, Route, FileCheck, Clock, ArrowRightLeft, AlertTriangle } from 'lucide-react';
import { IntelligenceKpis } from '../types/dashboard';

interface IntelligenceKpiPanelProps {
    kpis: IntelligenceKpis;
    isDemoMode: boolean;
}

const KPI_DEFS = [
    { key: 'activeCameras' as const, label: 'ACTIVE CAMERAS', icon: Camera, color: 'var(--accent-cyan)' },
    { key: 'observedVehicles' as const, label: 'OBSERVED VEHICLES', icon: Eye, color: 'var(--accent-purple)' },
    { key: 'globalVehicles' as const, label: 'GLOBAL VEHICLES', icon: Globe, color: 'var(--accent-green)' },
    { key: 'activeJourneys' as const, label: 'ACTIVE JOURNEYS', icon: Route, color: '#00aaff' },
    { key: 'confirmedPlates' as const, label: 'CONFIRMED PLATES', icon: FileCheck, color: 'var(--accent-green)' },
    { key: 'pendingOcr' as const, label: 'PENDING OCR', icon: Clock, color: 'var(--accent-amber)' },
    { key: 'unknownPlates' as const, label: 'UNKNOWN PLATES', icon: AlertTriangle, color: 'var(--text-muted)' },
    { key: 'probableTransitions' as const, label: 'PROBABLE TRANSITIONS', icon: ArrowRightLeft, color: 'var(--accent-amber)' },
    { key: 'unobservedGaps' as const, label: 'UNOBSERVED GAPS', icon: AlertTriangle, color: 'var(--accent-rose)' },
];

export const IntelligenceKpiPanel: React.FC<IntelligenceKpiPanelProps> = ({ kpis, isDemoMode }) => {
    return (
        <div style={{
            display: 'grid',
            gridTemplateColumns: `repeat(${KPI_DEFS.length}, 1fr)`,
            gap: '0.5rem',
            marginBottom: '0.75rem',
        }}>
            {KPI_DEFS.map(def => {
                const Icon = def.icon;
                const val = kpis[def.key];
                return (
                    <div key={def.key} className="glass-panel" style={{
                        padding: '0.65rem 0.75rem',
                        position: 'relative',
                    }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', marginBottom: '0.35rem' }}>
                            <Icon size={12} color={def.color} />
                            <span style={{
                                fontSize: '0.52rem', fontWeight: 700, color: 'var(--text-muted)',
                                letterSpacing: '0.06em', textTransform: 'uppercase',
                            }}>
                                {def.label}
                            </span>
                        </div>
                        <div style={{
                            fontSize: '1.4rem', fontWeight: 800, fontFamily: 'var(--font-mono)',
                            color: def.color,
                            textShadow: `0 0 12px ${def.color}40`,
                        }}>
                            {val.toString().padStart(2, '0')}
                        </div>
                        {isDemoMode && (
                            <div style={{
                                position: 'absolute', top: '4px', right: '6px',
                                fontSize: '0.42rem', color: 'var(--accent-amber)', fontWeight: 700, opacity: 0.7,
                            }}>
                                DEMO
                            </div>
                        )}
                    </div>
                );
            })}
        </div>
    );
};
