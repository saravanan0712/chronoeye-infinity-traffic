import React, { useState, useEffect } from 'react';
import { AlertTriangle, Activity, ShieldAlert, Zap, FileWarning, HelpCircle } from 'lucide-react';
import { IncidentEvent } from '../types/dashboard';

interface IncidentPanelProps {
    incidents?: IncidentEvent[];
    isDemoMode?: boolean;
}

// Vehicle-intelligence specific incident anomalies
const DEMO_INTELLIGENCE_INCIDENTS: IncidentEvent[] = [
    {
        incident_id: 'INC_REID_001',
        incident_type: 'IMPOSSIBLE_TRAVEL_TIME',
        severity: 'HIGH',
        segment_id: 'CAM_A_EAST → CAM_C_NORTH',
        confidence: 0.94,
        status: 'ACTIVE',
        detected_at: Date.now() / 1000 - 95,
        evidence: { observed_dt_s: 14, min_feasible_dt_s: 45, implied_speed_kmh: 184 },
    },
    {
        incident_id: 'INC_OCR_002',
        incident_type: 'OCR_CONSENSUS_CONFLICT',
        severity: 'MEDIUM',
        segment_id: 'CAM_B_WEST [TRK_209]',
        confidence: 0.82,
        status: 'VERIFIED',
        detected_at: Date.now() / 1000 - 240,
        evidence: { candidates: ['TN09AB1234', 'TN09AR1234'], char_agreement: 0.72 },
    },
    {
        incident_id: 'INC_REID_003',
        incident_type: 'UNCERTAIN_ASSOCIATION',
        severity: 'MEDIUM',
        segment_id: 'CAM_C_NORTH → CAM_D_SOUTH',
        confidence: 0.68,
        status: 'RESOLVING',
        detected_at: Date.now() / 1000 - 410,
        evidence: { decision: 'MATCH_PROBABLE', visual_score: 0.64, plate_status: 'UNKNOWN' },
    },
    {
        incident_id: 'INC_GAP_004',
        incident_type: 'UNOBSERVED_CORRIDOR_GAP',
        severity: 'LOW',
        segment_id: 'CAM_A_EAST → CAM_B_WEST',
        confidence: 0.91,
        status: 'RESOLVED',
        detected_at: Date.now() / 1000 - 750,
        evidence: { blind_spot_distance_m: 850, estimated_delay_s: 42 },
    },
];

const SEVERITY_CONFIG = {
    HIGH: { color: 'var(--accent-rose)', bg: 'rgba(255,24,96,0.08)', border: 'rgba(255,24,96,0.3)' },
    MEDIUM: { color: 'var(--accent-amber)', bg: 'rgba(255,184,0,0.08)', border: 'rgba(255,184,0,0.3)' },
    LOW: { color: 'var(--accent-cyan)', bg: 'rgba(0,212,255,0.08)', border: 'rgba(0,212,255,0.25)' },
};

const INCIDENT_TYPE_CONFIG: Record<string, { icon: any; title: string }> = {
    IMPOSSIBLE_TRAVEL_TIME: { icon: Zap, title: 'IMPOSSIBLE TRAVEL KINEMATICS' },
    OCR_CONSENSUS_CONFLICT: { icon: FileWarning, title: 'OCR CHARACTER CONFLICT' },
    UNCERTAIN_ASSOCIATION: { icon: HelpCircle, title: 'PROBABLE REID ASSOCIATION' },
    UNOBSERVED_CORRIDOR_GAP: { icon: AlertTriangle, title: 'UNOBSERVED CORRIDOR BLIND SPOT' },
    ABNORMAL_SPEED_DROP: { icon: Activity, title: 'CORRIDOR BOTTLENECK DETECTED' },
    SUDDEN_QUEUE_GROWTH: { icon: ShieldAlert, title: 'TRAFFIC GRAPH QUEUE SPIKE' },
};

export const IncidentPanel: React.FC<IncidentPanelProps> = ({
    incidents = [],
    isDemoMode = false,
}) => {
    const [filter, setFilter] = useState<'ALL' | 'HIGH' | 'MEDIUM' | 'LOW'>('ALL');
    const [pulse, setPulse] = useState(false);

    const displayIncidents = incidents.length > 0 ? incidents : DEMO_INTELLIGENCE_INCIDENTS;
    const isDemo = isDemoMode || incidents.length === 0;

    const filtered = filter === 'ALL'
        ? displayIncidents
        : displayIncidents.filter(i => i.severity === filter);

    useEffect(() => {
        if (incidents.length > 0) {
            setPulse(true);
            const t = setTimeout(() => setPulse(false), 1000);
            return () => clearTimeout(t);
        }
    }, [incidents.length]);

    const timeSince = (ts: number) => {
        const diff = Math.floor(Date.now() / 1000 - ts);
        if (diff < 60) return `${diff}s ago`;
        if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
        return `${Math.floor(diff / 3600)}h ago`;
    };

    return (
        <div
            className="glass-panel"
            style={{
                padding: '1.25rem',
                marginBottom: '1rem',
                borderColor: pulse ? 'rgba(255,24,96,0.5)' : 'var(--border-color)',
                transition: 'border-color 0.5s ease',
            }}
        >
            {/* Header */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.85rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <ShieldAlert size={18} color="var(--accent-rose)" />
                    <div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                            <h2 style={{
                                fontFamily: 'var(--font-display)',
                                fontSize: '0.72rem',
                                fontWeight: 700,
                                color: 'var(--text-primary)',
                                letterSpacing: '0.1em',
                            }}>
                                REID ANOMALIES & INTELLIGENCE ALERTS
                            </h2>
                            {isDemo && (
                                <span className="demo-tag" style={{ fontSize: '0.55rem', padding: '1px 5px', borderRadius: '3px' }}>
                                    DEMO DATA
                                </span>
                            )}
                        </div>
                        <p style={{ fontSize: '0.62rem', color: 'var(--text-secondary)', marginTop: '2px' }}>
                            Spatial-Temporal Constraint Auditing · OCR Ambiguity & Physics Anomaly Sentinel
                        </p>
                    </div>
                </div>

                {/* Filters */}
                <div style={{ display: 'flex', gap: '0.3rem' }}>
                    {(['ALL', 'HIGH', 'MEDIUM', 'LOW'] as const).map(f => (
                        <button
                            key={f}
                            onClick={() => setFilter(f)}
                            style={{
                                background: filter === f ? 'rgba(0,212,255,0.15)' : 'transparent',
                                border: `1px solid ${filter === f ? 'var(--accent-cyan)' : 'var(--border-color)'}`,
                                color: filter === f ? 'var(--accent-cyan)' : 'var(--text-muted)',
                                fontSize: '0.58rem',
                                fontWeight: 700,
                                padding: '2px 7px',
                                borderRadius: '4px',
                                cursor: 'pointer',
                                transition: 'all 0.15s ease',
                            }}
                        >
                            {f}
                        </button>
                    ))}
                </div>
            </div>

            {/* Status Alert Banner */}
            <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
                padding: '0.45rem 0.75rem',
                borderRadius: '6px',
                marginBottom: '0.75rem',
                background: displayIncidents.length > 0 ? 'rgba(255,24,96,0.06)' : 'rgba(0,255,157,0.06)',
                border: `1px solid ${displayIncidents.length > 0 ? 'rgba(255,24,96,0.2)' : 'rgba(0,255,157,0.2)'}`,
            }}>
                <Activity size={12} color={displayIncidents.length > 0 ? 'var(--accent-rose)' : 'var(--accent-green)'} />
                <span style={{ fontSize: '0.68rem', fontWeight: 600, color: displayIncidents.length > 0 ? 'var(--accent-rose)' : 'var(--accent-green)' }}>
                    {displayIncidents.length > 0
                        ? `${displayIncidents.length} INTELLIGENCE ALERTS DETECTED — PIPELINE AUDITING ACTIVE`
                        : 'ALL REID PIPELINES NOMINAL — ZERO IRREGULARITIES'}
                </span>
            </div>

            {/* Incidents List */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem', maxHeight: '240px', overflowY: 'auto', paddingRight: '2px' }}>
                {filtered.map((inc, i) => {
                    const sev = SEVERITY_CONFIG[inc.severity as keyof typeof SEVERITY_CONFIG] || SEVERITY_CONFIG.LOW;
                    const typeConfig = INCIDENT_TYPE_CONFIG[inc.incident_type] || { icon: AlertTriangle, title: inc.incident_type.replace(/_/g, ' ') };
                    const IconComp = typeConfig.icon;

                    return (
                        <div
                            key={i}
                            className="incident-alert"
                            style={{
                                background: sev.bg,
                                border: `1px solid ${sev.border}`,
                                borderRadius: '8px',
                                padding: '0.75rem',
                                animationDelay: `${i * 80}ms`,
                            }}
                        >
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                                <div style={{ display: 'flex', gap: '0.65rem', alignItems: 'flex-start' }}>
                                    <div style={{
                                        padding: '4px',
                                        borderRadius: '6px',
                                        background: 'rgba(0,0,0,0.3)',
                                        color: sev.color,
                                    }}>
                                        <IconComp size={15} />
                                    </div>
                                    <div>
                                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '3px' }}>
                                            <span style={{ fontSize: '0.75rem', fontWeight: 700, color: sev.color }}>
                                                {typeConfig.title}
                                            </span>
                                            <span className={`badge badge-${inc.severity === 'HIGH' ? 'rose' : inc.severity === 'MEDIUM' ? 'amber' : 'green'}`}>
                                                {inc.severity}
                                            </span>
                                        </div>
                                        <div style={{ fontSize: '0.68rem', color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>
                                            {inc.segment_id} · CONF: {(inc.confidence * 100).toFixed(0)}%
                                        </div>
                                    </div>
                                </div>
                                <div style={{ textAlign: 'right', flexShrink: 0 }}>
                                    <div style={{ fontSize: '0.68rem', fontWeight: 600, color: 'var(--accent-green)', fontFamily: 'var(--font-mono)' }}>
                                        {inc.status}
                                    </div>
                                    <div style={{ fontSize: '0.6rem', color: 'var(--text-muted)', marginTop: '2px', fontFamily: 'var(--font-mono)' }}>
                                        {timeSince(inc.detected_at)}
                                    </div>
                                </div>
                            </div>

                            {/* Confidence Progress Bar */}
                            <div className="progress-bar-track" style={{ marginTop: '0.5rem' }}>
                                <div
                                    className="progress-bar-fill"
                                    style={{
                                        width: `${inc.confidence * 100}%`,
                                        background: `linear-gradient(90deg, ${sev.color}60, ${sev.color})`,
                                    }}
                                />
                            </div>
                        </div>
                    );
                })}
            </div>
        </div>
    );
};
