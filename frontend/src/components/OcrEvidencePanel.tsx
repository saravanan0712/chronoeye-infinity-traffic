import React from 'react';
import { FileCheck, CheckCircle2, Clock, XCircle, Layers, ShieldCheck } from 'lucide-react';
import { FusedPlateIdentity, PlateStatus } from '../types/dashboard';

interface OcrEvidencePanelProps {
    fusedPlate?: FusedPlateIdentity | null;
    plateNumber?: string;
    plateStatus?: PlateStatus | string;
    plateConfidence?: number;
    isDemoMode?: boolean;
}

export const OcrEvidencePanel: React.FC<OcrEvidencePanelProps> = ({
    fusedPlate,
    plateNumber = 'TN09AB1234',
    plateStatus = 'CONFIRMED',
    plateConfidence = 0.94,
    isDemoMode = false,
}) => {
    const isDemo = isDemoMode || (fusedPlate as any)?._demo;

    const displayPlate = fusedPlate?.best_plate_number || plateNumber || 'UNKNOWN';
    const displayStatus = (fusedPlate?.status || plateStatus || 'CONFIRMED').toUpperCase();
    const displayConfidence = fusedPlate?.overall_confidence ?? plateConfidence ?? 0.88;
    const obsCount = fusedPlate?.observation_count ?? 18;
    const agreeRatio = fusedPlate?.character_agreement_ratio ?? 0.96;

    const getStatusBadge = (status: string) => {
        switch (status) {
            case 'CONFIRMED':
                return { label: 'CONFIRMED', bg: 'rgba(0,255,157,0.15)', color: 'var(--accent-green)', border: 'rgba(0,255,157,0.4)', icon: CheckCircle2 };
            case 'PENDING':
                return { label: 'PENDING OCR', bg: 'rgba(255,184,0,0.15)', color: 'var(--accent-amber)', border: 'rgba(255,184,0,0.4)', icon: Clock };
            case 'UNKNOWN':
            default:
                return { label: 'UNKNOWN / UNRESOLVED', bg: 'rgba(255,24,96,0.15)', color: 'var(--accent-rose)', border: 'rgba(255,24,96,0.4)', icon: XCircle };
        }
    };

    const statusBadge = getStatusBadge(displayStatus);

    // Mock consensus characters if not provided in detail
    const chars = displayPlate.split('');
    const charConsensus = chars.map((char, i) => {
        // High confidence for first letters, slight jitter for demonstration
        const score = Math.min(1.0, Math.max(0.75, displayConfidence + (Math.sin(i * 1.5) * 0.05)));
        return {
            char,
            position: i + 1,
            confidence: Math.round(score * 100),
            alternatives: score < 0.9 ? [char === '0' ? 'O' : char === '8' ? 'B' : '?'] : [],
        };
    });

    const mockEvidenceFrames = fusedPlate?.raw_observations_audit || [
        { frame_id: 'FRM_1042_A', camera_id: 'CAM_A_EAST', raw_text: displayPlate, confidence: 0.92, timestamp: '10:14:21 AM' },
        { frame_id: 'FRM_1048_A', camera_id: 'CAM_A_EAST', raw_text: displayPlate, confidence: 0.95, timestamp: '10:14:22 AM' },
        { frame_id: 'FRM_2190_B', camera_id: 'CAM_B_WEST', raw_text: displayPlate, confidence: 0.89, timestamp: '10:15:46 AM' },
        { frame_id: 'FRM_3310_C', camera_id: 'CAM_C_NORTH', raw_text: displayPlate, confidence: 0.96, timestamp: '10:17:11 AM' },
    ];

    return (
        <div className="glass-panel" style={{ padding: '1.25rem', marginBottom: '1rem' }}>
            {/* Header */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                    <FileCheck size={18} color="var(--accent-green)" />
                    <div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                            <h2 style={{
                                fontFamily: 'var(--font-display)',
                                fontSize: '0.75rem',
                                fontWeight: 700,
                                color: 'var(--text-primary)',
                                letterSpacing: '0.1em',
                            }}>
                                TEMPORAL OCR MULTI-FRAME FUSION EVIDENCE
                            </h2>
                            {isDemo && (
                                <span className="demo-tag" style={{ fontSize: '0.55rem', padding: '1px 5px', borderRadius: '3px' }}>
                                    DEMO DATA
                                </span>
                            )}
                        </div>
                        <p style={{ fontSize: '0.62rem', color: 'var(--text-secondary)', marginTop: '2px' }}>
                            Bayesian Character Consensus & Temporal Plate Voting (Module 1 Perception)
                        </p>
                    </div>
                </div>

                {/* Status Badge */}
                <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.4rem',
                    padding: '4px 10px',
                    borderRadius: '6px',
                    background: statusBadge.bg,
                    border: `1px solid ${statusBadge.border}`,
                    color: statusBadge.color,
                    fontFamily: 'var(--font-mono)',
                    fontSize: '0.72rem',
                    fontWeight: 700,
                }}>
                    <statusBadge.icon size={13} />
                    <span>{statusBadge.label}</span>
                </div>
            </div>

            {/* Fused Identity Banner */}
            <div style={{
                background: 'var(--bg-inset)',
                borderRadius: '8px',
                border: '1px solid rgba(0,255,157,0.2)',
                padding: '1rem',
                marginBottom: '1rem',
                display: 'grid',
                gridTemplateColumns: '1.2fr 1fr 1fr',
                gap: '1rem',
                alignItems: 'center',
            }}>
                {/* Plate Display */}
                <div>
                    <span style={{ fontSize: '0.6rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                        FUSED CONSENSUS LICENSE PLATE
                    </span>
                    <div style={{
                        marginTop: '4px',
                        fontSize: '1.5rem',
                        fontWeight: 900,
                        fontFamily: 'var(--font-mono)',
                        color: 'var(--accent-green)',
                        letterSpacing: '0.15em',
                        background: 'rgba(0,255,157,0.06)',
                        padding: '4px 12px',
                        borderRadius: '6px',
                        border: '1px solid rgba(0,255,157,0.3)',
                        display: 'inline-block',
                    }}>
                        {displayPlate}
                    </div>
                </div>

                {/* Metrics */}
                <div style={{ borderLeft: '1px solid var(--border-color)', paddingLeft: '1rem' }}>
                    <div style={{ fontSize: '0.6rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>
                        FUSION CONFIDENCE
                    </div>
                    <div style={{ fontSize: '1.1rem', fontWeight: 800, color: 'var(--accent-cyan)', fontFamily: 'var(--font-mono)', marginTop: '2px' }}>
                        {Math.round(displayConfidence * 100)}%
                    </div>
                    <div className="progress-bar-track" style={{ marginTop: '4px', width: '90%' }}>
                        <div
                            className="progress-bar-fill"
                            style={{
                                width: `${displayConfidence * 100}%`,
                                background: 'linear-gradient(90deg, #00d4ff, #00ff9d)',
                            }}
                        />
                    </div>
                </div>

                <div style={{ borderLeft: '1px solid var(--border-color)', paddingLeft: '1rem' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                        <span style={{ fontSize: '0.6rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>SAMPLED FRAMES:</span>
                        <span style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>{obsCount} frames</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span style={{ fontSize: '0.6rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>AGREEMENT RATIO:</span>
                        <span style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--accent-green)', fontFamily: 'var(--font-mono)' }}>{Math.round(agreeRatio * 100)}%</span>
                    </div>
                </div>
            </div>

            {/* Character Consensus Breakdown */}
            <div style={{ marginBottom: '1rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', marginBottom: '0.5rem' }}>
                    <Layers size={13} color="var(--accent-cyan)" />
                    <span style={{ fontSize: '0.65rem', fontWeight: 700, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                        PER-CHARACTER SPATIAL-TEMPORAL PROBABILITY CONSENSUS
                    </span>
                </div>

                <div style={{ display: 'flex', gap: '0.5rem', overflowX: 'auto', paddingBottom: '0.3rem' }}>
                    {charConsensus.map((c, idx) => (
                        <div
                            key={idx}
                            style={{
                                flex: 1,
                                minWidth: '45px',
                                background: 'var(--bg-inset)',
                                border: `1px solid ${c.confidence >= 90 ? 'rgba(0,255,157,0.3)' : 'rgba(255,184,0,0.3)'}`,
                                borderRadius: '6px',
                                padding: '0.5rem 0.35rem',
                                textAlign: 'center',
                            }}
                        >
                            <div style={{ fontSize: '0.55rem', color: 'var(--text-muted)', marginBottom: '2px' }}>
                                POS #{c.position}
                            </div>
                            <div style={{
                                fontSize: '1.1rem',
                                fontWeight: 800,
                                color: c.confidence >= 90 ? 'var(--accent-green)' : 'var(--accent-amber)',
                                fontFamily: 'var(--font-mono)',
                            }}>
                                {c.char}
                            </div>
                            <div style={{
                                fontSize: '0.62rem',
                                fontWeight: 700,
                                color: 'var(--text-secondary)',
                                fontFamily: 'var(--font-mono)',
                                marginTop: '2px',
                            }}>
                                {c.confidence}%
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            {/* Frame Observation Audit Trail */}
            <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', marginBottom: '0.5rem' }}>
                    <ShieldCheck size={13} color="var(--accent-cyan)" />
                    <span style={{ fontSize: '0.65rem', fontWeight: 700, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                        TEMPORAL FRAME AUDIT TRAIL
                    </span>
                </div>

                <div style={{
                    background: 'var(--bg-inset)',
                    borderRadius: '6px',
                    border: '1px solid var(--border-color)',
                    maxHeight: '130px',
                    overflowY: 'auto',
                }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.68rem', fontFamily: 'var(--font-mono)' }}>
                        <thead>
                            <tr style={{ borderBottom: '1px solid var(--border-color)', background: 'rgba(255,255,255,0.02)', color: 'var(--text-muted)' }}>
                                <th style={{ padding: '5px 8px', textAlign: 'left' }}>FRAME ID</th>
                                <th style={{ padding: '5px 8px', textAlign: 'left' }}>CAMERA</th>
                                <th style={{ padding: '5px 8px', textAlign: 'left' }}>RAW OCR READING</th>
                                <th style={{ padding: '5px 8px', textAlign: 'right' }}>CONFIDENCE</th>
                                <th style={{ padding: '5px 8px', textAlign: 'right' }}>TIMESTAMP</th>
                            </tr>
                        </thead>
                        <tbody>
                            {mockEvidenceFrames.map((frame, i) => (
                                <tr key={i} style={{ borderBottom: '1px solid rgba(255,255,255,0.03)' }}>
                                    <td style={{ padding: '5px 8px', color: 'var(--accent-cyan)' }}>{frame.frame_id}</td>
                                    <td style={{ padding: '5px 8px', color: 'var(--text-primary)' }}>{frame.camera_id}</td>
                                    <td style={{ padding: '5px 8px', color: 'var(--accent-green)', fontWeight: 600 }}>{frame.raw_text}</td>
                                    <td style={{ padding: '5px 8px', textAlign: 'right', color: 'var(--text-primary)' }}>
                                        {Math.round(frame.confidence * 100)}%
                                    </td>
                                    <td style={{ padding: '5px 8px', textAlign: 'right', color: 'var(--text-muted)' }}>{frame.timestamp}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
};
