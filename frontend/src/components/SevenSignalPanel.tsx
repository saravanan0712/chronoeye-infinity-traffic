import React from 'react';
import { ReIDScoreBreakdown } from '../types/dashboard';
import { Fingerprint } from 'lucide-react';

interface SevenSignalPanelProps {
    breakdown: ReIDScoreBreakdown | null | undefined;
    isDemoMode: boolean;
    fromCamera?: string;
    toCamera?: string;
}

const SIGNAL_DEFS = [
    { key: 'plate_similarity' as const, label: 'LICENSE PLATE', color: '#00d4ff' },
    { key: 'appearance_similarity' as const, label: 'APPEARANCE', color: '#8b00ff' },
    { key: 'visual_features_similarity' as const, label: 'VISUAL FEATURES', color: '#b366ff' },
    { key: 'vehicle_type_similarity' as const, label: 'VEHICLE TYPE', color: '#00ff9d' },
    { key: 'temporal_compatibility' as const, label: 'TIMESTAMP / TEMPORAL', color: '#ffb800' },
    { key: 'spatial_compatibility' as const, label: 'CAMERA LOCATION / SPATIAL', color: '#00aaff' },
    { key: 'direction_similarity' as const, label: 'DIRECTION / TRAVEL FEASIBILITY', color: '#ff6b00' },
];

export const SevenSignalPanel: React.FC<SevenSignalPanelProps> = ({ breakdown, isDemoMode, fromCamera, toCamera }) => {
    const hasData = breakdown !== null && breakdown !== undefined;

    return (
        <div className="glass-panel" style={{ padding: '1rem' }}>
            {/* Header */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.85rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <Fingerprint size={16} color="var(--accent-cyan)" />
                    <div>
                        <h2 style={{
                            fontFamily: 'var(--font-display)', fontSize: '0.68rem', fontWeight: 700,
                            color: 'var(--text-primary)', letterSpacing: '0.1em',
                        }}>
                            SEVEN-SIGNAL ASSOCIATION
                        </h2>
                        {fromCamera && toCamera && (
                            <p style={{ fontSize: '0.58rem', color: 'var(--text-secondary)', marginTop: '1px' }}>
                                {fromCamera} → {toCamera}
                            </p>
                        )}
                    </div>
                </div>
                {isDemoMode && hasData && (
                    <span className="demo-mode-badge demo" style={{ fontSize: '0.5rem', padding: '1px 6px' }}>DEMO DATA</span>
                )}
            </div>

            {!hasData ? (
                <div className="evidence-unavailable">
                    <Fingerprint size={24} color="var(--text-muted)" />
                    <span>SIGNAL EVIDENCE UNAVAILABLE</span>
                    <span style={{ fontSize: '0.6rem' }}>Select a transition to view seven-signal breakdown</span>
                </div>
            ) : (
                <>
                    {/* Seven Signal Bars */}
                    {SIGNAL_DEFS.map(sig => {
                        const val = breakdown[sig.key];
                        const pct = Math.round(val * 100);
                        return (
                            <div key={sig.key} className="signal-bar-container">
                                <span className="signal-bar-label">{sig.label}</span>
                                <div className="signal-bar-track">
                                    <div className="signal-bar-fill" style={{
                                        width: `${pct}%`,
                                        background: `linear-gradient(90deg, ${sig.color}60, ${sig.color})`,
                                    }} />
                                </div>
                                <span className="signal-bar-value" style={{ color: sig.color }}>
                                    {pct}%
                                </span>
                            </div>
                        );
                    })}

                    {/* Overall + Decision */}
                    <div style={{
                        marginTop: '0.75rem', paddingTop: '0.65rem',
                        borderTop: '1px solid var(--border-color)',
                        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                    }}>
                        <div>
                            <span style={{ fontSize: '0.58rem', color: 'var(--text-muted)', letterSpacing: '0.06em' }}>
                                OVERALL SCORE
                            </span>
                            <div style={{
                                fontSize: '1.3rem', fontWeight: 800, fontFamily: 'var(--font-mono)',
                                color: 'var(--accent-cyan)',
                                textShadow: '0 0 12px rgba(0,212,255,0.4)',
                            }}>
                                {Math.round(breakdown.overall_score * 100)}%
                            </div>
                        </div>
                        <div style={{ textAlign: 'right' }}>
                            <span style={{ fontSize: '0.58rem', color: 'var(--text-muted)', letterSpacing: '0.06em' }}>
                                DECISION
                            </span>
                            <div>
                                <span className={`decision-badge decision-${breakdown.decision}`}>
                                    {breakdown.decision.replace(/_/g, ' ')}
                                </span>
                            </div>
                            {breakdown.rejection_reason && (
                                <div style={{ fontSize: '0.55rem', color: 'var(--accent-rose)', marginTop: '3px' }}>
                                    {breakdown.rejection_reason}
                                </div>
                            )}
                        </div>
                    </div>
                </>
            )}
        </div>
    );
};
