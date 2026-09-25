import React, { useState } from 'react';
import { Route, CheckCircle2, HelpCircle, XCircle } from 'lucide-react';
import { VehicleJourney, MatchDecision } from '../types/dashboard';

interface JourneyGraphProps {
    journey: VehicleJourney | null;
    selectedSegmentIndex?: number;
    onSelectSegment?: (index: number) => void;
    isDemoMode?: boolean;
}

export const JourneyGraph: React.FC<JourneyGraphProps> = ({
    journey,
    selectedSegmentIndex,
    onSelectSegment,
    isDemoMode = false,
}) => {
    const [hoveredNode, setHoveredNode] = useState<number | null>(null);
    const [hoveredEdge, setHoveredEdge] = useState<number | null>(null);

    if (!journey || !journey.segments || journey.segments.length === 0) {
        return (
            <div className="glass-panel" style={{ padding: '2rem', textAlign: 'center', marginBottom: '1rem' }}>
                <Route size={32} color="var(--accent-cyan)" style={{ opacity: 0.5, margin: '0 auto 0.75rem' }} />
                <h3 style={{ fontSize: '0.9rem', color: 'var(--text-primary)', marginBottom: '0.25rem' }}>
                    NO VEHICLE JOURNEY SELECTED
                </h3>
                <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                    Select an active global vehicle or search by plate to reconstruct the multi-camera journey graph.
                </p>
            </div>
        );
    }

    const segments = journey.segments;
    const isDemo = isDemoMode || (journey as any)._demo;

    const getNodeX = (index: number, total: number) => {
        if (total <= 1) return 400;
        const padding = 100;
        const availableWidth = 800 - padding * 2;
        return padding + (index / (total - 1)) * availableWidth;
    };

    const getNodeY = (index: number) => {
        return index % 2 === 0 ? 110 : 150;
    };

    const getDecisionBadge = (decision?: MatchDecision | string | null) => {
        switch (decision) {
            case MatchDecision.MATCH_CONFIRMED:
            case 'MATCH_CONFIRMED':
                return { label: 'CONFIRMED', color: 'var(--accent-green)', bg: 'rgba(0,255,157,0.15)', border: 'rgba(0,255,157,0.4)', icon: CheckCircle2 };
            case MatchDecision.MATCH_PROBABLE:
            case 'MATCH_PROBABLE':
                return { label: 'PROBABLE', color: 'var(--accent-amber)', bg: 'rgba(255,184,0,0.15)', border: 'rgba(255,184,0,0.4)', icon: HelpCircle };
            case MatchDecision.MATCH_REJECTED:
            case 'MATCH_REJECTED':
                return { label: 'REJECTED', color: 'var(--accent-rose)', bg: 'rgba(255,24,96,0.15)', border: 'rgba(255,24,96,0.4)', icon: XCircle };
            default:
                return { label: 'INSUFFICIENT', color: 'var(--text-muted)', bg: 'rgba(255,255,255,0.05)', border: 'rgba(255,255,255,0.15)', icon: HelpCircle };
        }
    };

    const getPlateStatusStyle = (status?: string | null) => {
        switch (status) {
            case 'CONFIRMED':
                return { bg: 'rgba(0,255,157,0.15)', color: 'var(--accent-green)', border: 'rgba(0,255,157,0.3)' };
            case 'PENDING':
                return { bg: 'rgba(255,184,0,0.15)', color: 'var(--accent-amber)', border: 'rgba(255,184,0,0.3)' };
            case 'UNKNOWN':
            default:
                return { bg: 'rgba(255,24,96,0.15)', color: 'var(--accent-rose)', border: 'rgba(255,24,96,0.3)' };
        }
    };

    return (
        <div className="glass-panel" style={{ padding: '1.25rem', marginBottom: '1rem', position: 'relative' }}>
            {/* Header */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                    <Route size={18} color="var(--accent-cyan)" />
                    <div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                            <h2 style={{
                                fontFamily: 'var(--font-display)',
                                fontSize: '0.75rem',
                                fontWeight: 700,
                                color: 'var(--text-primary)',
                                letterSpacing: '0.1em',
                            }}>
                                OBSERVED JOURNEY RECONSTRUCTION GRAPH
                            </h2>
                            {isDemo && (
                                <span className="demo-tag" style={{ fontSize: '0.55rem', padding: '1px 5px', borderRadius: '3px' }}>
                                    DEMO DATA
                                </span>
                            )}
                        </div>
                        <p style={{ fontSize: '0.62rem', color: 'var(--text-secondary)', marginTop: '2px' }}>
                            Spatial-Temporal Camera Transitions & Association Evidence (Module 2 Pipeline)
                        </p>
                    </div>
                </div>

                {/* Journey Summary Stats */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', fontSize: '0.7rem', fontFamily: 'var(--font-mono)' }}>
                    <div style={{ padding: '3px 8px', borderRadius: '4px', background: 'var(--bg-inset)', border: '1px solid var(--border-color)' }}>
                        <span style={{ color: 'var(--text-muted)', fontSize: '0.6rem' }}>VEHICLE: </span>
                        <span style={{ color: 'var(--accent-cyan)', fontWeight: 700 }}>{journey.global_vehicle_id}</span>
                    </div>
                    <div style={{ padding: '3px 8px', borderRadius: '4px', background: 'var(--bg-inset)', border: '1px solid var(--border-color)' }}>
                        <span style={{ color: 'var(--text-muted)', fontSize: '0.6rem' }}>PLATE: </span>
                        <span style={{ color: 'var(--accent-green)', fontWeight: 700 }}>{journey.plate_number || 'UNKNOWN'}</span>
                    </div>
                    <div style={{ padding: '3px 8px', borderRadius: '4px', background: 'var(--bg-inset)', border: '1px solid var(--border-color)' }}>
                        <span style={{ color: 'var(--text-muted)', fontSize: '0.6rem' }}>HOPS: </span>
                        <span style={{ color: 'var(--text-primary)', fontWeight: 700 }}>{segments.length} CAMERAS</span>
                    </div>
                    <div style={{ padding: '3px 8px', borderRadius: '4px', background: 'var(--bg-inset)', border: '1px solid var(--border-color)' }}>
                        <span style={{ color: 'var(--text-muted)', fontSize: '0.6rem' }}>CONFIDENCE: </span>
                        <span style={{ color: 'var(--accent-cyan)', fontWeight: 700 }}>
                            {Math.round((journey.overall_confidence ?? 0.85) * 100)}%
                        </span>
                    </div>
                </div>
            </div>

            {/* SVG Graph Canvas */}
            <div style={{
                background: 'linear-gradient(180deg, rgba(6,11,25,0.95), rgba(4,7,14,0.98))',
                borderRadius: '8px',
                border: '1px solid rgba(0,212,255,0.15)',
                overflowX: 'auto',
                position: 'relative',
                padding: '0.5rem 0',
            }}>
                <svg
                    viewBox="0 0 800 240"
                    style={{ width: '100%', minWidth: '700px', height: '240px', display: 'block' }}
                >
                    <defs>
                        <filter id="cyanGlow" x="-20%" y="-20%" width="140%" height="140%">
                            <feGaussianBlur stdDeviation="4" result="blur" />
                            <feComposite in="SourceGraphic" in2="blur" operator="over" />
                        </filter>
                        <filter id="amberGlow" x="-20%" y="-20%" width="140%" height="140%">
                            <feGaussianBlur stdDeviation="4" result="blur" />
                            <feComposite in="SourceGraphic" in2="blur" operator="over" />
                        </filter>
                        <linearGradient id="edgeGradCyan" x1="0%" y1="0%" x2="100%" y2="0%">
                            <stop offset="0%" stopColor="#00d4ff" stopOpacity="0.8" />
                            <stop offset="100%" stopColor="#00ff9d" stopOpacity="0.8" />
                        </linearGradient>
                    </defs>

                    {/* Background Grid */}
                    <g opacity="0.08">
                        {Array.from({ length: 16 }).map((_, i) => (
                            <line key={`grid-v-${i}`} x1={i * 50} y1="0" x2={i * 50} y2="240" stroke="#00d4ff" strokeWidth="1" />
                        ))}
                        {Array.from({ length: 5 }).map((_, i) => (
                            <line key={`grid-h-${i}`} x1="0" y1={i * 50} x2="800" y2={i * 50} stroke="#00d4ff" strokeWidth="1" />
                        ))}
                    </g>

                    {/* Edges */}
                    {segments.map((seg, idx) => {
                        if (idx === 0) return null;
                        const x1 = getNodeX(idx - 1, segments.length);
                        const y1 = getNodeY(idx - 1);
                        const x2 = getNodeX(idx, segments.length);
                        const y2 = getNodeY(idx);

                        const isSelected = selectedSegmentIndex === idx;
                        const isHovered = hoveredEdge === idx;
                        const hasGap = seg.has_unobserved_gap;
                        const score = Math.round((seg.transition_score ?? 0.88) * 100);
                        const decisionBadge = getDecisionBadge(seg.transition_decision);

                        const midX = (x1 + x2) / 2;
                        const midY = (y1 + y2) / 2;
                        const ctrlY = midY - (hasGap ? 25 : 12);
                        const pathData = `M ${x1} ${y1} Q ${midX} ${ctrlY} ${x2} ${y2}`;

                        return (
                            <g
                                key={`edge-${idx}`}
                                onClick={() => onSelectSegment && onSelectSegment(idx)}
                                onMouseEnter={() => setHoveredEdge(idx)}
                                onMouseLeave={() => setHoveredEdge(null)}
                                style={{ cursor: 'pointer' }}
                            >
                                <path d={pathData} fill="none" stroke="transparent" strokeWidth="24" />

                                {(isSelected || isHovered) && (
                                    <path
                                        d={pathData}
                                        fill="none"
                                        stroke={hasGap ? '#ffb800' : '#00d4ff'}
                                        strokeWidth="8"
                                        strokeOpacity="0.4"
                                        strokeLinecap="round"
                                        filter="url(#cyanGlow)"
                                    />
                                )}

                                <path
                                    d={pathData}
                                    fill="none"
                                    stroke={hasGap ? '#ffb800' : 'url(#edgeGradCyan)'}
                                    strokeWidth={isSelected ? 3.5 : 2}
                                    strokeDasharray={hasGap ? '6 4' : undefined}
                                    strokeLinecap="round"
                                />

                                <g transform={`translate(${midX}, ${ctrlY})`}>
                                    <rect
                                        x="-44"
                                        y="-13"
                                        width="88"
                                        height="26"
                                        rx="13"
                                        fill="rgba(4,7,14,0.92)"
                                        stroke={hasGap ? '#ffb800' : isSelected ? '#00d4ff' : 'rgba(0,212,255,0.4)'}
                                        strokeWidth={isSelected ? 1.8 : 1}
                                        filter="drop-shadow(0 2px 4px rgba(0,0,0,0.5))"
                                    />
                                    <text
                                        x="0"
                                        y="-1"
                                        textAnchor="middle"
                                        fill={decisionBadge.color}
                                        fontSize="8"
                                        fontWeight="700"
                                        fontFamily="var(--font-mono)"
                                    >
                                        {decisionBadge.label} {score}%
                                    </text>
                                    <text
                                        x="0"
                                        y="8"
                                        textAnchor="middle"
                                        fill={hasGap ? '#ffb800' : 'var(--text-secondary)'}
                                        fontSize="7"
                                        fontWeight="600"
                                        fontFamily="var(--font-mono)"
                                    >
                                        {hasGap ? 'UNOBSERVED' : '7-SIG FUSED'}
                                    </text>
                                </g>
                            </g>
                        );
                    })}

                    {/* Camera Nodes */}
                    {segments.map((seg, idx) => {
                        const x = getNodeX(idx, segments.length);
                        const y = getNodeY(idx);
                        const isSelected = selectedSegmentIndex === idx;
                        const isHovered = hoveredNode === idx;
                        const plateStyle = getPlateStatusStyle(seg.plate_status);
                        const timeStr = seg.timestamp ? new Date(seg.timestamp * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) : '--:--:--';

                        return (
                            <g
                                key={`node-${idx}`}
                                transform={`translate(${x}, ${y})`}
                                onClick={() => onSelectSegment && onSelectSegment(idx)}
                                onMouseEnter={() => setHoveredNode(idx)}
                                onMouseLeave={() => setHoveredNode(null)}
                                style={{ cursor: 'pointer' }}
                            >
                                {(isSelected || isHovered) && (
                                    <circle
                                        r="30"
                                        fill="none"
                                        stroke="var(--accent-cyan)"
                                        strokeWidth="2"
                                        strokeDasharray="4 2"
                                        opacity="0.8"
                                        filter="url(#cyanGlow)"
                                    />
                                )}

                                <circle
                                    r="22"
                                    fill="rgba(6,11,25,0.95)"
                                    stroke={isSelected ? 'var(--accent-cyan)' : 'rgba(0,212,255,0.4)'}
                                    strokeWidth={isSelected ? 2.5 : 1.5}
                                />

                                <circle
                                    r="14"
                                    fill={isSelected ? 'rgba(0,212,255,0.2)' : 'rgba(0,212,255,0.08)'}
                                />
                                <text
                                    x="0"
                                    y="4"
                                    textAnchor="middle"
                                    fill="var(--accent-cyan)"
                                    fontSize="10"
                                    fontWeight="bold"
                                    fontFamily="var(--font-mono)"
                                >
                                    C{idx + 1}
                                </text>

                                <g transform="translate(0, -32)">
                                    <rect
                                        x="-40"
                                        y="-9"
                                        width="80"
                                        height="18"
                                        rx="4"
                                        fill="rgba(4,7,14,0.9)"
                                        stroke="rgba(0,212,255,0.3)"
                                        strokeWidth="1"
                                    />
                                    <text
                                        x="0"
                                        y="3"
                                        textAnchor="middle"
                                        fill="var(--text-primary)"
                                        fontSize="8.5"
                                        fontWeight="700"
                                        fontFamily="var(--font-mono)"
                                    >
                                        {seg.camera_id}
                                    </text>
                                </g>

                                <g transform="translate(0, 32)">
                                    <rect
                                        x="-38"
                                        y="-7"
                                        width="76"
                                        height="15"
                                        rx="3"
                                        fill={plateStyle.bg}
                                        stroke={plateStyle.border}
                                        strokeWidth="1"
                                    />
                                    <text
                                        x="0"
                                        y="3.5"
                                        textAnchor="middle"
                                        fill={plateStyle.color}
                                        fontSize="7.5"
                                        fontWeight="700"
                                        fontFamily="var(--font-mono)"
                                    >
                                        {seg.plate_status || 'UNKNOWN'}: {seg.track_id}
                                    </text>

                                    <text
                                        x="0"
                                        y="18"
                                        textAnchor="middle"
                                        fill="var(--text-muted)"
                                        fontSize="7.5"
                                        fontFamily="var(--font-mono)"
                                    >
                                        {timeStr}
                                    </text>
                                </g>
                            </g>
                        );
                    })}
                </svg>
            </div>

            {/* Legend */}
            <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                marginTop: '0.75rem',
                fontSize: '0.65rem',
                color: 'var(--text-muted)',
                fontFamily: 'var(--font-mono)',
            }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                        <div style={{ width: '16px', height: '2px', background: 'var(--accent-green)' }} />
                        <span>CONFIRMED TRANSITION</span>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                        <div style={{ width: '16px', height: '2px', background: 'var(--accent-amber)', borderTop: '2px dashed #ffb800' }} />
                        <span>UNOBSERVED GAP / PROBABLE</span>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                        <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: 'var(--accent-cyan)' }} />
                        <span>CAMERA OBSERVATION NODE</span>
                    </div>
                </div>

                <div>
                    <span style={{ color: 'var(--accent-cyan)' }}>Tip:</span> Click any camera node or transition edge to inspect detailed 7-signal evidence.
                </div>
            </div>
        </div>
    );
};
